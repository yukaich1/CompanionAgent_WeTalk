from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime

from pydantic import BaseModel, Field

from knowledge.persona_shared import DISPLAY_KEYWORD_LIMIT, PERSONA_SUMMARY_SCHEMA, dedupe
from knowledge.persona_summary_refiners import (
    build_background_extraction_prompt,
    build_display_keywords_prompt,
    build_key_experience_selection_prompt,
    build_keyword_selection_prompt,
)
from knowledge.story_segmentation import segment_story_section
from llm import MistralEmbeddingCapacityError, MistralLLM, get_llm_settings
from persona_prompting import build_persona_summary_prompt
from rag.models import RAGChunk
from rag.processing import estimate_tokens, extract_keywords
from rag.tool import RAGTool


class StoryUnitModel(BaseModel):
    title: str = ""
    paragraph_ids: list[int] = Field(default_factory=list)


class StorySectionModel(BaseModel):
    section_type: str = "non_story"
    story_units: list[StoryUnitModel] = Field(default_factory=list)


KEYWORD_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords"],
}

KEYWORD_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords"],
}

VOICE_CARD_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"character_voice_card": {"type": "string"}},
    "required": ["character_voice_card"],
}

BACKGROUND_EXPERIENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "key_experiences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["key_experiences"],
}

KEY_EXPERIENCE_INDEX_SCHEMA = {
    "type": "object",
    "properties": {
        "indices": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["indices"],
}


class PersonaIngestService:
    def __init__(self, system):
        self.system = system
        self.rag = RAGTool(llm=self.system.model)

    def _summary_model_name(self) -> str:
        configured = str(os.getenv("LLM_SUMMARY_MODEL", "") or "").strip()
        if configured:
            return configured
        settings = get_llm_settings()
        return settings.chat_model

    def _summary_timeout(self) -> int:
        value = str(os.getenv("LLM_SUMMARY_TIMEOUT", "") or "").strip()
        try:
            return max(30, int(value)) if value else 120
        except ValueError:
            return 120

    def _summary_max_tokens(self) -> int:
        value = str(os.getenv("LLM_SUMMARY_MAX_TOKENS", "") or "").strip()
        try:
            return max(2000, int(value)) if value else 8000
        except ValueError:
            return 8000

    def _story_segmentation_timeout(self) -> int:
        value = str(os.getenv("LLM_STORY_SEGMENT_TIMEOUT", "") or "").strip()
        try:
            return max(20, int(value)) if value else 90
        except ValueError:
            return 90

    def _normalize_keyword(self, token: str, max_chars: int = 14) -> str:
        return self.system._normalize_display_keyword(token, max_chars=max_chars)

    def _refine_voice_card_against_source(self, summary: dict, source_text: str) -> str:
        voice_card = self.system._normalize_fact(summary.get("character_voice_card", ""), max_chars=600)
        if not voice_card:
            return ""

        prompt = f"""
你要把一段角色自述改写成“严格受原始资料约束”的版本。
要求：
1. 只保留原始资料里明确支持的身份、气质、说话方式、价值取向。
2. 删除任何原始资料里没有明确支持的新事实、新经历、新设定、新关系。
3. 尤其不要凭空补充职业、身世、过去经历、地理经历、关系史。
4. 如果原始资料没有出现某个具体名词或具体物件，就不要自己补出来。
5. 如果原始资料只支持高层倾向，就保持高层表达，不要把它具体化成新场景或新细节。
6. 保持第一人称、角色口吻和简短自述感。
7. 允许保留风格，但不允许保留无证据事实。
8. 不要加入原始资料里没有出现过的口头习惯或戏剧化收尾。
9. 长度控制在 40 到 120 字。
10. 只输出 JSON。

原始资料：
{source_text[:2200]}

待改写文本：
{voice_card}
""".strip()
        try:
            payload = self.system.model.generate(
                prompt,
                return_json=True,
                schema=VOICE_CARD_REWRITE_SCHEMA,
                temperature=0.0,
                max_tokens=180,
            )
            refined = self.system._normalize_fact((payload or {}).get("character_voice_card", ""), max_chars=220)
            return refined or voice_card
        except Exception:
            return voice_card

    def _looks_like_display_keyword(self, token: str) -> bool:
        value = self._normalize_keyword(token, max_chars=12)
        if not value:
            return False
        if len(value) < 2 or len(value) > 8:
            return False
        if "\n" in value or "  " in value:
            return False
        if re.search(r"[\"'“”‘’（）()\[\]{}<>:：；，。！？?\\|@#%^&*_+=~`]", value):
            return False
        if re.search(r"\d", value):
            return False
        return True

    def _keyword_label_sources(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        texts: list[str] = []

        voice_card = str(summary_dict.get("character_voice_card", "") or "").strip()
        if voice_card:
            texts.append(voice_card)

        for keyword in list(summary_dict.get("display_keywords", []) or []):
            clean = self._normalize_keyword(keyword, max_chars=12)
            if clean:
                texts.append(clean)

        base_template = summary_dict.get("base_template", {}) or {}
        for dim_data in base_template.values():
            if not isinstance(dim_data, dict):
                continue
            for item in dim_data.get("items", []) or []:
                if isinstance(item, dict):
                    clean = str(item.get("item", "") or "").strip()
                    if clean:
                        texts.append(clean)
            for item in dim_data.get("patterns", []) or []:
                if isinstance(item, dict):
                    clean = str(item.get("pattern", "") or "").strip()
                    if clean:
                        texts.append(clean)
        return texts

    def _keyword_evidence_sources(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        texts: list[str] = []

        voice_card = str(summary_dict.get("character_voice_card", "") or "").strip()
        if voice_card:
            texts.append(voice_card)

        base_template = summary_dict.get("base_template", {}) or {}
        for dim_data in base_template.values():
            if not isinstance(dim_data, dict):
                continue
            for field in ("rules", "key_experiences"):
                for item in dim_data.get(field, []) or []:
                    clean = str(item or "").strip()
                    if clean:
                        texts.append(clean)
            for item in dim_data.get("items", []) or []:
                if not isinstance(item, dict):
                    continue
                for key in ("item", "behavior", "level"):
                    clean = str(item.get(key, "") or "").strip()
                    if clean:
                        texts.append(clean)
            for item in dim_data.get("patterns", []) or []:
                if not isinstance(item, dict):
                    continue
                for key in ("pattern", "usage", "tone", "reason", "alternative"):
                    clean = str(item.get(key, "") or "").strip()
                    if clean:
                        texts.append(clean)

        for example in summary_dict.get("style_examples", []) or []:
            if isinstance(example, dict):
                clean = str(example.get("text", "") or "").strip()
                if clean:
                    texts.append(clean)

        return texts

    def _candidate_keyword_pool(self, summary: dict) -> list[str]:
        pool: list[str] = []
        for token in self._keyword_label_sources(summary):
            clean = self._normalize_keyword(token, max_chars=12)
            if clean and self._looks_like_display_keyword(clean):
                pool.append(clean)
        return dedupe(pool, limit=DISPLAY_KEYWORD_LIMIT)

    def _llm_keyword_candidates(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        label_text = "\n".join(self._keyword_label_sources(summary_dict))[:3000]
        evidence_text = "\n".join(self._keyword_evidence_sources(summary_dict))[:5000]
        if not evidence_text.strip():
            return []

        prompt = build_display_keywords_prompt(label_text, evidence_text)
        try:
            payload = self.system.model.generate(
                prompt,
                return_json=True,
                schema=KEYWORD_CANDIDATE_SCHEMA,
                temperature=0.0,
                max_tokens=420,
            )
        except Exception:
            return []

        normalized = [
            self._normalize_keyword(item, max_chars=12)
            for item in list((payload or {}).get("keywords", []) or [])
        ]
        normalized = [item for item in normalized if item and self._looks_like_display_keyword(item)]
        return dedupe(normalized, limit=DISPLAY_KEYWORD_LIMIT)

    def _finalize_keyword_candidates(self, summary: dict, raw_candidates: list[str]) -> list[str]:
        candidates = [
            self._normalize_keyword(item, max_chars=12)
            for item in list(raw_candidates or [])
        ]
        candidates = [item for item in candidates if item and self._looks_like_display_keyword(item)]
        candidates = dedupe(candidates, limit=24)
        if not candidates:
            return []

        evidence_text = "\n".join(self._keyword_evidence_sources(summary))[:5000]
        candidate_text = "\n".join(f"- {item}" for item in candidates)
        prompt = build_keyword_selection_prompt(evidence_text, candidate_text)
        try:
            payload = self.system.model.generate(
                prompt,
                return_json=True,
                schema=KEYWORD_SELECTION_SCHEMA,
                temperature=0.0,
                max_tokens=320,
            )
        except Exception:
            return dedupe(candidates, limit=DISPLAY_KEYWORD_LIMIT)

        normalized = [
            self._normalize_keyword(item, max_chars=12)
            for item in list((payload or {}).get("keywords", []) or [])
        ]
        normalized = [item for item in normalized if item and self._looks_like_display_keyword(item)]
        return dedupe(normalized or candidates, limit=DISPLAY_KEYWORD_LIMIT)

    def refine_display_keywords(self, summary: dict, raw_text: str) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        llm_keywords = self._llm_keyword_candidates(summary_dict)
        direct_keywords = self._candidate_keyword_pool(summary_dict)
        candidates = [item for item in [*llm_keywords, *direct_keywords] if item]
        candidates = dedupe(
            [
                self._normalize_keyword(item, max_chars=12)
                for item in candidates
                if self._normalize_keyword(item, max_chars=12)
            ],
            limit=24,
        )
        if not candidates:
            return []

        source_evidence = self.system._prepare_summary_source(raw_text, max_chars=2400)
        candidate_text = "\n".join(f"- {item}" for item in candidates)
        prompt = build_keyword_selection_prompt(source_evidence, candidate_text)
        try:
            payload = self.system.model.generate(
                prompt,
                return_json=True,
                schema=KEYWORD_SELECTION_SCHEMA,
                temperature=0.0,
                max_tokens=320,
            )
            normalized = [
                self._normalize_keyword(item, max_chars=12)
                for item in list((payload or {}).get("keywords", []) or [])
            ]
            normalized = [item for item in normalized if item and self._looks_like_display_keyword(item)]
            return dedupe(normalized or candidates, limit=DISPLAY_KEYWORD_LIMIT)
        except Exception:
            return self._finalize_keyword_candidates(summary_dict, candidates)

    def _extract_background_key_experiences(self, profile: dict, source_text: str) -> list[str]:
        profile_lines = [
            f"{key}: {self.system._normalize_fact(value, max_chars=80)}"
            for key, value in dict(profile or {}).items()
            if self.system._normalize_fact(value, max_chars=80)
        ]

        segments: list[str] = []
        raw_sentences = re.split(r"(?<=[。！？；\n])", str(source_text or ""))
        for sentence in raw_sentences:
            clean_sentence = self.system._normalize_fact(sentence, max_chars=160)
            if not clean_sentence:
                continue
            if len(clean_sentence) < 12:
                continue
            if re.match(r"^(并|但|而|后来|于是|然后|不过|只是|也|还|又)", clean_sentence):
                continue
            segments.append(clean_sentence)

        candidates: list[str] = []
        for sentence in segments:
            if not sentence or len(sentence) < 12:
                continue
            if self.system._is_meta_commentary(sentence):
                continue
            candidates.append(sentence)

        window_candidates: list[str] = []
        for index, sentence in enumerate(candidates):
            window_candidates.append(sentence)
            if index + 1 < len(candidates):
                merged = f"{sentence} {candidates[index + 1]}".strip()
                merged = self.system._normalize_fact(merged, max_chars=220)
                if merged and not self.system._is_meta_commentary(merged):
                    window_candidates.append(merged)
        candidates = dedupe(window_candidates, limit=24)

        if candidates:
            prompt = build_key_experience_selection_prompt(profile_lines, candidates)
            schema = KEY_EXPERIENCE_INDEX_SCHEMA
        else:
            prompt = build_background_extraction_prompt(profile_lines, source_text)
            schema = BACKGROUND_EXPERIENCE_SCHEMA

        try:
            payload = self.system.model.generate(
                prompt,
                return_json=True,
                schema=schema,
                temperature=0.0,
                max_tokens=360,
            )
        except Exception:
            return []

        if candidates:
            picked_indices = [
                int(item)
                for item in list((payload or {}).get("indices", []) or [])
                if isinstance(item, int) or (isinstance(item, str) and str(item).isdigit())
            ]
            experiences = [candidates[index - 1] for index in picked_indices if 1 <= index <= len(candidates)]
            if not experiences:
                fallback_prompt = build_background_extraction_prompt(profile_lines, source_text)
                try:
                    fallback_payload = self.system.model.generate(
                        fallback_prompt,
                        return_json=True,
                        schema=BACKGROUND_EXPERIENCE_SCHEMA,
                        temperature=0.0,
                        max_tokens=360,
                    )
                    raw_fallback = [
                        self.system._normalize_fact(item, max_chars=160)
                        for item in list((fallback_payload or {}).get("key_experiences", []) or [])
                    ]
                    experiences = [item for item in raw_fallback if item in candidates]
                except Exception:
                    experiences = []
        else:
            experiences = [
                self.system._normalize_fact(item, max_chars=160)
                for item in list((payload or {}).get("key_experiences", []) or [])
            ]

        experiences = dedupe([item for item in experiences if item], limit=5)
        experiences = self._prune_overlapping_experiences(experiences)
        if not experiences or not profile_lines:
            return experiences

        verification_prompt = f"""
你要从候选条目里确认哪些内容真的属于“背景关键经历”。
要求：
1. 只保留成长起点、启蒙、资格取得、拜师修行、重大试炼、重要约定、出发动机、关键转折。
2. 删除纯身份档案、称号、出生地、外貌、装备、一般偏好、泛泛性格概括。
3. 如果一条只是“是谁、来自哪里、叫什么、长什么样”，它不是关键经历。
4. 宁缺毋滥。
5. 如果一条把“身份档案”和“真正经历”硬拼在一起，也不要选；只保留更纯粹、更像单一经历单元的条目。
6. 只输出 JSON。

身份档案：
{chr(10).join(f"- {line}" for line in profile_lines[:8])}

候选条目：
{chr(10).join(f"{index}. {item}" for index, item in enumerate(experiences, start=1))}
""".strip()
        try:
            verification = self.system.model.generate(
                verification_prompt,
                return_json=True,
                schema=KEY_EXPERIENCE_INDEX_SCHEMA,
                temperature=0.0,
                max_tokens=220,
            )
            verified_indices = [
                int(item)
                for item in list((verification or {}).get("indices", []) or [])
                if isinstance(item, int) or (isinstance(item, str) and str(item).isdigit())
            ]
            verified = [experiences[index - 1] for index in verified_indices if 1 <= index <= len(experiences)]
            return self._prune_overlapping_experiences(dedupe(verified or experiences, limit=5))
        except Exception:
            return experiences

    def _prune_overlapping_experiences(self, experiences: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in sorted(
            [self.system._normalize_fact(text, max_chars=180) for text in list(experiences or []) if text],
            key=len,
        ):
            if any(item == existing or item in existing or existing in item for existing in cleaned):
                if any(existing in item and item != existing for existing in cleaned):
                    continue
            cleaned.append(item)
        return dedupe(cleaned, limit=5)

    def summarize_with_llm(self, raw_text: str, source_label: str) -> dict | None:
        source_text = self.system._prepare_summary_source(raw_text)
        if len(source_text) > 1800:
            source_text = source_text[:1800]

        summary_model = self._summary_model_name()
        summary_timeout = self._summary_timeout()
        summary_max_tokens = self._summary_max_tokens()
        prompt = build_persona_summary_prompt(
            persona_name=self.system.persona_name,
            source_label=source_label,
            source_text=source_text,
            reference_text="",
        )
        self.system.last_summary_debug = {
            "sourceLabel": source_label,
            "rawTextChars": len(str(raw_text or "")),
            "preparedSourceChars": len(source_text),
            "promptChars": len(prompt),
            "summaryModel": summary_model,
            "summaryTimeout": summary_timeout,
            "summaryMaxTokens": summary_max_tokens,
            "summaryStatus": "requesting",
        }
        try:
            summary_llm = MistralLLM(summary_model)
            data = summary_llm.generate(
                prompt,
                return_json=True,
                schema=PERSONA_SUMMARY_SCHEMA,
                temperature=0.1,
                max_tokens=summary_max_tokens,
                timeout=summary_timeout,
                max_tries=3,
                request_label="persona_summary",
            )
        except Exception as exc:
            self.system.last_summary_debug.update(
                {
                    "summaryStatus": "failed",
                    "summaryError": f"{type(exc).__name__}: {exc}",
                }
            )
            return None

        normalized = self.system._normalize_summary(data)
        normalized["character_voice_card"] = self._refine_voice_card_against_source(normalized, source_text)
        normalized["character_name"] = str(data.get("character_name") or self.system.persona_name or "").strip()
        normalized["source_label"] = str(data.get("source_label") or source_label or "").strip()
        background = normalized.setdefault("base_template", {}).setdefault(
            "00_BACKGROUND",
            {"profile": {}, "key_experiences": [], "confidence": "medium"},
        )
        profile = background.setdefault("profile", {})
        if self.system.persona_name and not profile.get("full_name"):
            profile["full_name"] = self.system.persona_name
        background["key_experiences"] = self._extract_background_key_experiences(profile, raw_text)
        self.system.last_summary_debug["summaryStatus"] = "success"
        return normalized

    def selected_keyword_candidates(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        llm_keywords = self._llm_keyword_candidates(summary_dict)
        direct_keywords = self._candidate_keyword_pool(summary_dict)
        return self._finalize_keyword_candidates(summary_dict, [*llm_keywords, *direct_keywords])

    def get_display_keywords(self, limit: int = DISPLAY_KEYWORD_LIMIT) -> list[str]:
        return dedupe(self.system.display_keywords, limit=limit)

    def rebuild_index_from_entries(self):
        self.system.index, self.system.index_dim = self.rag.rebuild_index(self.system.entries)

    def append_entries_without_embeddings(self, pending_entries: list[dict], source_fingerprint: str):
        self.system.entries.extend(pending_entries)
        if source_fingerprint:
            self.system.source_records[source_fingerprint] = {
                "source_label": pending_entries[0].get("source_label", ""),
                "updated_at": datetime.now().isoformat(),
            }
        self.system._dedupe_storage()

    def _wrap_progress_callback(self, progress_callback=None, base: int = 0, span: int = 100, default_total: int = 1):
        def emit(current, total, stage, detail):
            ratio = current / float(total or default_total or 1)
            overall = base + int(ratio * span)
            self.system._emit_progress(progress_callback, min(overall, base + span), 100, stage, detail)

        return emit

    def _commit_pending_entries(
        self,
        pending_entries: list[dict],
        source_label: str,
        source_fingerprint: str = "",
        progress_callback=None,
        progress_base: int = 0,
        progress_span: int = 100,
        success_message: str | None = None,
        fallback_message: str | None = None,
    ) -> int:
        if not pending_entries:
            return 0

        embed_progress = self._wrap_progress_callback(
            progress_callback=progress_callback,
            base=progress_base,
            span=progress_span,
            default_total=len(pending_entries),
        )
        try:
            self.rag.embed_chunks(pending_entries, progress_callback=embed_progress)
        except Exception:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            if fallback_message:
                self.system._emit_progress(progress_callback, 100, 100, "done", fallback_message)
            return len(pending_entries)

        self.system.entries.extend(pending_entries)
        if source_fingerprint:
            self.system.source_records[source_fingerprint] = {
                "source_label": source_label,
                "updated_at": datetime.now().isoformat(),
            }
        self.system._dedupe_storage()
        self.rebuild_index_from_entries()
        if success_message:
            self.system._emit_progress(progress_callback, 100, 100, "done", success_message)
        return len(pending_entries)

    def _chunk_markdown_source(self, raw_text: str, source_label: str, kind: str, priority: float) -> list[dict]:
        document_id = hashlib.sha1(f"{source_label}:{raw_text[:120]}".encode("utf-8")).hexdigest()[:12]
        markdown = self.rag.convert_text_to_markdown(raw_text, title=source_label)
        chunks = self.rag.build_chunks(
            markdown,
            document_id=document_id,
            source_label=source_label,
            source_type="persona",
            priority=priority,
            metadata={"kind": kind},
        )
        return [RAGChunk(**{**dict(chunk), "kind": kind, "source_label": source_label, "source_type": "persona"}).model_dump() for chunk in chunks]

    def _build_section_markdown(self, heading_path: list[str], paragraphs: list[str]) -> str:
        clean_paragraphs = [str(item or "").strip() for item in paragraphs if str(item or "").strip()]
        if not clean_paragraphs:
            return ""
        heading_block = "\n".join(
            f"{'#' * (index + 1)} {title}"
            for index, title in enumerate(list(heading_path or [])[:4])
            if str(title).strip()
        )
        body = "\n\n".join(clean_paragraphs)
        return "\n\n".join(part for part in (heading_block, body) if part).strip()

    def _segment_story_section(self, section: dict) -> StorySectionModel:
        payload = segment_story_section(self.system.model, section, timeout=self._story_segmentation_timeout())
        return StorySectionModel(**payload)

    def _build_story_chunk(
        self,
        source_label: str,
        document_id: str,
        section: dict,
        unit: StoryUnitModel,
        sequence: int,
    ) -> dict | None:
        paragraphs = list(section.get("paragraphs", []) or [])
        selected = [
            str(paragraphs[index - 1] or "").strip()
            for index in list(unit.paragraph_ids or [])
            if 1 <= index <= len(paragraphs) and str(paragraphs[index - 1] or "").strip()
        ]
        if not selected:
            return None

        heading_path = list(section.get("heading_path", []) or [])
        content = self._build_section_markdown(heading_path, selected)
        if not content.strip():
            return None

        title = str(unit.title or section.get("title") or f"故事片段{sequence}").strip() or f"故事片段{sequence}"
        chunk_id = hashlib.sha1(f"{document_id}:story:{sequence}:{title}:{content[:160]}".encode("utf-8")).hexdigest()[:16]
        return RAGChunk(
            chunk_id=chunk_id,
            document_id=f"{document_id}:story:{sequence}",
            source_label=source_label,
            source_type="persona",
            content=content,
            markdown_path=heading_path[:4],
            keywords=extract_keywords(f"{title}\n{content}", limit=10),
            token_count=estimate_tokens(content),
            priority=1.0,
            kind="story_chunk",
            title=title,
            metadata={
                "kind": "story_chunk",
                "title": title,
                "paragraph_ids": list(unit.paragraph_ids or []),
                "heading_path": heading_path[:4],
            },
        ).model_dump()

    def _build_structured_entries(self, raw_text: str, source_label: str) -> list[dict]:
        markdown = self.rag.convert_text_to_markdown(raw_text, title=source_label)
        sections = self.rag.document_processor.extract_sections(markdown)
        if not sections:
            return []

        document_id = hashlib.sha1(f"{source_label}:{raw_text[:120]}".encode("utf-8")).hexdigest()[:12]
        structured_entries: list[dict] = []
        story_sequence = 0

        for index, section in enumerate(sections, start=1):
            segmentation = self._segment_story_section(section)
            consumed_ids: set[int] = set()

            for unit in segmentation.story_units:
                story_sequence += 1
                story_entry = self._build_story_chunk(source_label, document_id, section, unit, story_sequence)
                if story_entry:
                    structured_entries.append(story_entry)
                    consumed_ids.update(unit.paragraph_ids)

            remaining_paragraphs = [
                paragraph
                for para_index, paragraph in enumerate(list(section.get("paragraphs", []) or []), start=1)
                if para_index not in consumed_ids and str(paragraph or "").strip()
            ]
            section_markdown = self._build_section_markdown(list(section.get("heading_path", []) or []), remaining_paragraphs)
            if not section_markdown.strip():
                continue
            source_entries = self.rag.build_chunks(
                section_markdown,
                document_id=f"{document_id}:section:{index}",
                source_label=source_label,
                source_type="persona",
                priority=1.0,
                metadata={"kind": "source_chunk", "section_type": segmentation.section_type or "non_story"},
            )
            structured_entries.extend(source_entries)

        return [RAGChunk(**chunk).model_dump() for chunk in structured_entries]

    def commit_entries_only(
        self,
        raw_text: str,
        source_label: str,
        kind: str = "source_chunk",
        priority: float = 1.0,
        progress_callback=None,
    ) -> int:
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            return 0

        source_fingerprint = self.system._source_fingerprint(raw_text)
        if source_fingerprint and source_fingerprint in self.system.source_records:
            existing = self.system.source_records[source_fingerprint]
            self.system._emit_progress(progress_callback, 100, 100, "done", f"检测到重复资料，已跳过：{existing.get('source_label', source_label)}")
            return 0

        pending_entries = self._chunk_markdown_source(raw_text, source_label, kind, priority)
        if not pending_entries:
            return 0
        return self._commit_pending_entries(
            pending_entries=pending_entries,
            source_label=source_label,
            source_fingerprint=source_fingerprint,
            progress_callback=progress_callback,
        )

    def commit_summary_and_entries(self, raw_text: str, source_label: str, summary: dict | None, progress_callback=None) -> int:
        raw_text = str(raw_text or "").strip()
        summary_dict = self.system._summary_to_dict(summary) if summary else {}
        source_fingerprint = self.system._source_fingerprint(raw_text)
        if source_fingerprint and source_fingerprint in self.system.source_records:
            existing = self.system.source_records[source_fingerprint]
            self.system._emit_progress(progress_callback, 100, 100, "done", f"检测到重复资料，已跳过：{existing.get('source_label', source_label)}")
            return 0

        self.system._emit_progress(progress_callback, 10, 100, "summary", "正在整理角色基础模板")
        if summary_dict:
            summary_dict["display_keywords"] = self.refine_display_keywords(summary_dict, raw_text)
            self.system._merge_summary_data(summary_dict)

        self.system._emit_progress(progress_callback, 22, 100, "structuring", "正在识别故事与资料结构")
        pending_entries = self._build_structured_entries(raw_text, source_label)
        if not pending_entries:
            if source_fingerprint:
                self.system.source_records[source_fingerprint] = {
                    "source_label": source_label,
                    "updated_at": datetime.now().isoformat(),
                }
            self.system._emit_progress(progress_callback, 100, 100, "done", "没有新增内容，已完成")
            return 0

        total_entries = len(pending_entries)
        story_count = sum(1 for entry in pending_entries if str(entry.get("kind") or "") == "story_chunk")
        self.system._emit_progress(
            progress_callback,
            40,
            100,
            "embedding",
            f"正在为 {total_entries} 个检索块生成向量，其中故事块 {story_count} 个",
        )
        return self._commit_pending_entries(
            pending_entries=pending_entries,
            source_label=source_label,
            source_fingerprint=source_fingerprint,
            progress_callback=progress_callback,
            progress_base=40,
            progress_span=45,
            success_message=f"学习完成，共新增 {len(pending_entries)} 个检索块，其中故事块 {story_count} 个",
            fallback_message=f"学习完成，共新增 {len(pending_entries)} 个检索块，其中故事块 {story_count} 个（纯文本检索模式）",
        )

    def load_text(self, raw_text: str, source_label: str = "manual_input", progress_callback=None) -> int:
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            return 0
        summary = self.summarize_with_llm(raw_text, source_label)
        return self.commit_summary_and_entries(raw_text, source_label, summary, progress_callback=progress_callback)

    def load_file(self, filepath: str, progress_callback=None) -> int:
        path = os.fspath(filepath)
        markdown = self.rag.convert_path_to_markdown(path)
        return self.load_text(markdown, source_label=os.path.basename(path), progress_callback=progress_callback)

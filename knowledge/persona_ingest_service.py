from __future__ import annotations

import hashlib
import os
from datetime import datetime

from pydantic import BaseModel, Field

from knowledge.persona_shared import DISPLAY_KEYWORD_LIMIT, PERSONA_SUMMARY_SCHEMA, dedupe
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


STORY_SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "section_type": {"type": "string"},
        "story_units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "paragraph_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["title", "paragraph_ids"],
            },
        },
    },
    "required": ["section_type", "story_units"],
}

KEYWORD_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["keywords"],
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

    def _keyword_text_sources(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        texts: list[str] = []

        voice_card = str(summary_dict.get("character_voice_card", "") or "").strip()
        if voice_card:
            texts.append(voice_card)

        for dim_data in (summary_dict.get("base_template", {}) or {}).values():
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

    def _filter_keywords_against_sources(self, keywords: list[str], summary: dict) -> list[str]:
        source_text = "\n".join(self._keyword_text_sources(summary))
        filtered: list[str] = []
        for token in keywords:
            clean = self._normalize_keyword(token, max_chars=16)
            if not clean:
                continue
            if clean not in source_text:
                continue
            filtered.append(clean)
        return dedupe(filtered, limit=DISPLAY_KEYWORD_LIMIT)

    def _llm_keyword_candidates(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        source_text = "\n".join(self._keyword_text_sources(summary_dict))[:5000]
        if not source_text.strip():
            return []
        prompt = f"""
你要从下面这份“角色分析师已提炼好的结构化信息”中，挑出适合展示和选择的关键词候选。

要求：
1. 关键词必须来自这份分析里已经明确出现或可以直接提炼出的内容，不能新增外部信息。
2. 关键词要优先覆盖：说话方式、态度、气质、关系距离感、口头习惯、明显喜恶、核心性格、叙事方式。
3. 不要输出过泛的词，例如“角色”“设定”“故事”“经历”“性格”“喜欢”“讨厌”。
4. 优先输出 12 到 24 个短关键词或短短语。
5. 尽量让这些词能帮助使用者理解“这个角色会怎么说、怎么表现自己”，而不只是泛泛事实标签。
6. 只输出 JSON。

分析信息：
{source_text}
""".strip()
        try:
            payload = self.system.model.generate(
                prompt,
                return_json=True,
                schema=KEYWORD_CANDIDATE_SCHEMA,
                temperature=0.1,
                max_tokens=400,
            )
        except Exception:
            return []
        raw_keywords = list((payload or {}).get("keywords", []) or [])
        normalized = [self._normalize_keyword(item, max_chars=16) for item in raw_keywords]
        normalized = [item for item in normalized if item]
        return dedupe(normalized, limit=DISPLAY_KEYWORD_LIMIT)

    def refine_display_keywords(self, summary: dict, raw_text: str) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        llm_keywords = self._llm_keyword_candidates(summary_dict)
        direct_keywords = [self._normalize_keyword(item, max_chars=16) for item in list(summary_dict.get("display_keywords", []) or [])]
        combined = [item for item in [*llm_keywords, *direct_keywords] if item]
        return dedupe(combined, limit=DISPLAY_KEYWORD_LIMIT)

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
        self.system.last_summary_debug["summaryStatus"] = "success"
        normalized["character_name"] = str(data.get("character_name") or self.system.persona_name or "").strip()
        normalized["source_label"] = str(data.get("source_label") or source_label or "").strip()
        background = normalized.setdefault("base_template", {}).setdefault(
            "00_BACKGROUND",
            {"profile": {}, "key_experiences": [], "confidence": "medium"},
        )
        profile = background.setdefault("profile", {})
        if self.system.persona_name and not profile.get("full_name"):
            profile["full_name"] = self.system.persona_name
        return normalized

    def selected_keyword_candidates(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        llm_keywords = self._llm_keyword_candidates(summary_dict)
        direct_keywords = [self._normalize_keyword(item, max_chars=16) for item in list(summary_dict.get("display_keywords", []) or [])]
        keyword_sources = [item for item in [*llm_keywords, *direct_keywords] if item]
        return dedupe(keyword_sources, limit=DISPLAY_KEYWORD_LIMIT)

    def commit_entries_only(self, raw_text: str, source_label: str, kind: str = "source_chunk", priority: float = 1.0, progress_callback=None) -> int:
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

        def embedding_progress(current, total, stage, detail):
            self.system._emit_progress(progress_callback, current, total, stage, detail)

        try:
            self.rag.embed_chunks(pending_entries, progress_callback=embedding_progress)
        except MistralEmbeddingCapacityError:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            return len(pending_entries)
        except Exception:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            return len(pending_entries)

        self.system.entries.extend(pending_entries)
        if source_fingerprint:
            self.system.source_records[source_fingerprint] = {"source_label": source_label, "updated_at": datetime.now().isoformat()}
        self.system._dedupe_storage()
        self.rebuild_index_from_entries()
        return len(pending_entries)

    def commit_structured_entries_only(self, raw_text: str, source_label: str, progress_callback=None) -> int:
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            return 0
        pending_entries = self._build_structured_entries(raw_text, source_label)
        if not pending_entries:
            return 0

        def embedding_progress(current, total, stage, detail):
            self.system._emit_progress(progress_callback, current, total, stage, detail)

        try:
            self.rag.embed_chunks(pending_entries, progress_callback=embedding_progress)
        except Exception:
            self.system.entries.extend(pending_entries)
            self.system._dedupe_storage()
            return len(pending_entries)

        self.system.entries.extend(pending_entries)
        self.system._dedupe_storage()
        self.rebuild_index_from_entries()
        return len(pending_entries)

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
        normalized: list[dict] = []
        for chunk in chunks:
            payload = {
                **dict(chunk),
                "kind": kind,
                "source_label": source_label,
                "source_type": "persona",
            }
            rag_chunk = RAGChunk(**payload)
            normalized.append(rag_chunk.model_dump())
        return normalized

    def _build_section_markdown(self, heading_path: list[str], paragraphs: list[str]) -> str:
        clean_paragraphs = [str(item or "").strip() for item in paragraphs if str(item or "").strip()]
        if not clean_paragraphs:
            return ""
        heading_block = "\n".join(
            f"{'#' * (index + 1)} {title}" for index, title in enumerate(list(heading_path or [])[:4]) if str(title).strip()
        )
        body = "\n\n".join(clean_paragraphs)
        return "\n\n".join(part for part in (heading_block, body) if part).strip()

    def _story_section_prompt(self, section: dict) -> str:
        heading_path = " > ".join(section.get("heading_path", []) or []) or "无标题"
        paragraphs = list(section.get("paragraphs", []) or [])
        numbered = "\n\n".join(f"[{idx}] {paragraph}" for idx, paragraph in enumerate(paragraphs, start=1))
        return f"""
你是角色资料结构分析器。你的任务是判断下面这个资料片段里，哪些段落属于可以单独叙述的“完整故事/事件/经历”。

判断标准：
1. “故事/事件/经历”指具体发生过的一段过程，通常包含情境、经过、结果、变化中的至少一部分。
2. 纯基础档案、外貌介绍、设定说明、抽象性格总结、喜好列表、作品外评价，不算故事。
3. 如果一个片段里包含多个不同事件，要拆成多个 story_unit。
4. story_unit 必须使用输入里的原段落编号，不要改写内容，不要创造新段落。
5. paragraph_ids 尽量连续，且每个 story_unit 都应尽量能独立成篇。
6. 如果这部分没有完整故事，就返回空数组。

输出 JSON：
- section_type: `story_section` / `mixed_section` / `non_story`
- story_units: 数组，每项包含：
  - title: 这个故事单元的简短标题
  - paragraph_ids: 属于这个故事单元的段落编号数组

章节路径：
{heading_path}

段落列表：
{numbered}
""".strip()

    def _segment_story_section(self, section: dict) -> StorySectionModel:
        paragraphs = list(section.get("paragraphs", []) or [])
        if not paragraphs:
            return StorySectionModel()
        prompt = self._story_section_prompt(section)
        try:
            payload = self.system.model.generate(
                prompt,
                return_json=True,
                schema=STORY_SECTION_SCHEMA,
                temperature=0.0,
                max_tokens=500,
                timeout=self._story_segmentation_timeout(),
            )
            parsed = StorySectionModel(**(payload or {}))
        except Exception:
            return StorySectionModel()

        normalized_units: list[StoryUnitModel] = []
        max_index = len(paragraphs)
        seen: set[tuple[int, ...]] = set()
        for unit in parsed.story_units:
            ids = []
            for raw_id in list(unit.paragraph_ids or []):
                try:
                    value = int(raw_id)
                except Exception:
                    continue
                if 1 <= value <= max_index and value not in ids:
                    ids.append(value)
            ids.sort()
            if not ids:
                continue
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            title = str(unit.title or "").strip() or (section.get("title") or "故事片段")
            normalized_units.append(StoryUnitModel(title=title, paragraph_ids=ids))
        return StorySectionModel(section_type=str(parsed.section_type or "non_story").strip() or "non_story", story_units=normalized_units)

    def _build_story_chunk(self, source_label: str, base_document_id: str, section: dict, unit: StoryUnitModel, sequence: int) -> dict | None:
        paragraphs = list(section.get("paragraphs", []) or [])
        selected = [paragraphs[idx - 1].strip() for idx in unit.paragraph_ids if 1 <= idx <= len(paragraphs) and paragraphs[idx - 1].strip()]
        if not selected:
            return None
        heading_path = list(section.get("heading_path", []) or [])
        story_title = str(unit.title or "").strip() or (heading_path[-1] if heading_path else "故事片段")
        content = self._build_section_markdown([*heading_path, story_title] if story_title not in heading_path else heading_path, selected)
        if not content.strip():
            return None
        chunk_id = f"{base_document_id}:story:{sequence}"
        rag_chunk = RAGChunk(
            chunk_id=chunk_id,
            document_id=base_document_id,
            source_label=source_label,
            source_type="persona",
            content=content,
            markdown_path=[*heading_path[:4], story_title][:5],
            keywords=extract_keywords(f"{story_title}\n{content}", limit=12),
            token_count=estimate_tokens(content),
            priority=1.0,
            kind="story_chunk",
            title=story_title,
            metadata={
                "kind": "story_chunk",
                "title": story_title,
                "section_type": "story",
                "paragraph_ids": list(unit.paragraph_ids or []),
            },
        )
        return rag_chunk.model_dump()

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

        normalized: list[dict] = []
        for chunk in structured_entries:
            rag_chunk = RAGChunk(**chunk)
            normalized.append(rag_chunk.model_dump())
        return normalized

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
                self.system.source_records[source_fingerprint] = {"source_label": source_label, "updated_at": datetime.now().isoformat()}
            self.system._emit_progress(progress_callback, 100, 100, "done", "没有新增内容，已完成")
            return 0

        total_entries = len(pending_entries)
        story_count = sum(1 for entry in pending_entries if str(entry.get("kind") or "") == "story_chunk")
        self.system._emit_progress(progress_callback, 40, 100, "embedding", f"正在为 {total_entries} 个检索块生成向量，其中故事块 {story_count} 个")

        def embedding_progress(current, total, stage, detail):
            ratio = current / (total or total_entries or 1)
            overall = 40 + int(ratio * 45)
            self.system._emit_progress(progress_callback, min(overall, 88), 100, stage, detail)

        try:
            self.rag.embed_chunks(pending_entries, progress_callback=embedding_progress)
        except MistralEmbeddingCapacityError:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 个检索块（故事块 {story_count} 个，纯文本检索模式）")
            return len(pending_entries)
        except Exception:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 个检索块（故事块 {story_count} 个，纯文本检索模式）")
            return len(pending_entries)

        self.system.entries.extend(pending_entries)
        if source_fingerprint:
            self.system.source_records[source_fingerprint] = {"source_label": source_label, "updated_at": datetime.now().isoformat()}
        self.system._dedupe_storage()
        self.rebuild_index_from_entries()
        self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 个检索块，其中故事块 {story_count} 个")
        return len(pending_entries)

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

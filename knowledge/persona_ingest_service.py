from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime

from knowledge.persona_shared import DISPLAY_KEYWORD_LIMIT, PERSONA_SUMMARY_SCHEMA, dedupe
from llm import MistralEmbeddingCapacityError, MistralLLM, get_llm_settings
from persona_prompting import build_persona_summary_prompt
from rag.models import RAGChunk
from rag.tool import RAGTool


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

    def _extract_literal_keyword_candidates(self, texts: list[str]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for raw in texts:
            text = str(raw or "").strip()
            if not text:
                continue
            for part in re.split(r"[\n,，、/；;：:（）()\[\]【】“”\"'‘’]+", text):
                clean = self._normalize_keyword(part, max_chars=16)
                if not clean or len(clean) < 2 or clean in seen:
                    continue
                seen.add(clean)
                candidates.append(clean)
                if len(candidates) >= DISPLAY_KEYWORD_LIMIT:
                    return candidates
        return candidates

    def _extract_source_phrase_candidates(self, raw_text: str) -> list[str]:
        text = str(raw_text or "")
        candidates: list[str] = []
        seen: set[str] = set()

        patterns = [
            r"《([^》]{2,12})》",
            r"[“\"「『]([\u4e00-\u9fffA-Za-z]{2,12})[”\"」』]",
            r"(?:被称为|称为|称作|被称作)([\u4e00-\u9fffA-Za-z]{2,12})",
            r"(?:喜欢|喜爱|偏好|讨厌|厌恶|不喜欢|擅长|不擅长|过敏|害怕)([\u4e00-\u9fffA-Za-z]{1,12})",
            r"(?:自称|称号|身份|职业|种族|派系|立场|信条|原则)([\u4e00-\u9fffA-Za-z]{2,12})",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                value = match if isinstance(match, str) else "".join(match)
                clean = self._normalize_keyword(value, max_chars=16)
                if not clean or len(clean) < 2 or clean in seen:
                    continue
                seen.add(clean)
                candidates.append(clean)
                if len(candidates) >= DISPLAY_KEYWORD_LIMIT:
                    return candidates
        return candidates

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

    def heuristic_keyword_candidates(self, summary: dict) -> list[str]:
        return dedupe(self._extract_literal_keyword_candidates(self._keyword_text_sources(summary)), limit=DISPLAY_KEYWORD_LIMIT)

    def refine_display_keywords(self, summary: dict, raw_text: str) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        direct_keywords = self._filter_keywords_against_sources(list(summary_dict.get("display_keywords", []) or []), summary_dict)
        source_keywords = self._extract_source_phrase_candidates(raw_text)
        summary_keywords = self.heuristic_keyword_candidates(summary_dict)
        merged = dedupe(direct_keywords + source_keywords + summary_keywords, limit=DISPLAY_KEYWORD_LIMIT)
        return merged

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
        direct_keywords = list(summary_dict.get("display_keywords", []) or [])
        normalized = self._filter_keywords_against_sources(direct_keywords, summary_dict)
        if normalized:
            return normalized
        return self.heuristic_keyword_candidates(summary_dict)

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

        pending_entries: list[dict] = []
        pending_entries.extend(self._chunk_markdown_source(raw_text, source_label, "source_chunk", 1.0))

        if not pending_entries:
            if source_fingerprint:
                self.system.source_records[source_fingerprint] = {"source_label": source_label, "updated_at": datetime.now().isoformat()}
            self.system._emit_progress(progress_callback, 100, 100, "done", "没有新增内容，已完成")
            return 0

        total_entries = len(pending_entries)
        self.system._emit_progress(progress_callback, 35, 100, "embedding", f"正在为 {total_entries} 个 Markdown 检索块生成向量")

        def embedding_progress(current, total, stage, detail):
            ratio = current / (total or total_entries or 1)
            overall = 35 + int(ratio * 50)
            self.system._emit_progress(progress_callback, min(overall, 88), 100, stage, detail)

        try:
            self.rag.embed_chunks(pending_entries, progress_callback=embedding_progress)
        except MistralEmbeddingCapacityError:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 个检索块（纯文本检索模式）")
            return len(pending_entries)
        except Exception:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 个检索块（纯文本检索模式）")
            return len(pending_entries)

        self.system.entries.extend(pending_entries)
        if source_fingerprint:
            self.system.source_records[source_fingerprint] = {"source_label": source_label, "updated_at": datetime.now().isoformat()}
        self.system._dedupe_storage()
        self.rebuild_index_from_entries()
        self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 个检索块")
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

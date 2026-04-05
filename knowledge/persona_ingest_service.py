from __future__ import annotations

import os
from datetime import datetime

import faiss
import numpy as np

from knowledge.persona_shared import DISPLAY_KEYWORD_LIMIT, KEYWORD_CANDIDATE_LIMIT, PERSONA_SUMMARY_SCHEMA, PREFERRED_PERSONA_LABELS, dedupe
from llm import MistralEmbeddingCapacityError, mistral_embed_texts
from persona_models import PersonaKeywordOption
from persona_prompting import build_persona_summary_prompt


class PersonaIngestService:
    def __init__(self, system):
        self.system = system

    def _normalize_keyword(self, token: str, max_chars: int = 14) -> str:
        token = self.system._normalize_fact(token, max_chars=max_chars)
        token = token.strip(" \"'[]()（）【】「」『』《》、，。！？?；：")
        return token.strip()

    def _is_good_keyword(self, token: str) -> bool:
        token = self._normalize_keyword(token)
        if not token:
            return False
        if token in PREFERRED_PERSONA_LABELS:
            return True
        if len(token) < 2 or len(token) > 12:
            return False
        if token in {"角色", "设定", "资料", "文本", "描述", "表现", "使用", "方式", "风格", "感觉", "经历", "故事", "内容", "部分"}:
            return False
        if token.startswith(("如", "以", "来", "把", "说", "因", "对", "将", "被", "而", "并")):
            return False
        if any(marker in token for marker in ("的", "是", "在", "和", "中", "被", "把", "说", "如果", "但是", "因为", "所以")):
            return False
        return True

    def heuristic_keyword_candidates(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        combined_parts = [summary_dict.get("character_voice_card", "")]
        combined_parts.extend(summary_dict.get("display_keywords", []))
        for dim_data in summary_dict.get("base_template", {}).values():
            if isinstance(dim_data, dict):
                combined_parts.extend(dim_data.get("rules", []))
        combined_text = "\n".join(str(part) for part in combined_parts if part)
        candidates = [label for label in PREFERRED_PERSONA_LABELS if label in combined_text]
        return dedupe(candidates, limit=KEYWORD_CANDIDATE_LIMIT)

    def summarize_with_llm(self, raw_text: str, source_label: str) -> dict | None:
        source_text = self.system._prepare_summary_source(raw_text)
        prompt = build_persona_summary_prompt(
            persona_name=self.system.persona_name,
            source_label=source_label,
            source_text=source_text,
            reference_text="",
        )
        try:
            data = self.system.model.generate(
                prompt,
                return_json=True,
                schema=PERSONA_SUMMARY_SCHEMA,
                temperature=0.1,
                max_tokens=4000,
            )
        except Exception:
            return None

        normalized = self.system._normalize_summary(data)
        normalized["character_name"] = str(data.get("character_name") or self.system.persona_name or "").strip()
        normalized["source_label"] = str(data.get("source_label") or source_label or "").strip()
        background = normalized.setdefault("base_template", {}).setdefault(
            "00_BACKGROUND_PROFILE",
            {"profile": {}, "key_experiences": [], "confidence": "中"},
        )
        profile = background.setdefault("profile", {})
        if self.system.persona_name and not profile.get("full_name"):
            profile["full_name"] = self.system.persona_name
        return normalized

    def append_core_summary(self, summary: dict, source_label: str):
        return

    def selected_keyword_candidates(self, summary: dict) -> list[str]:
        summary_dict = self.system._summary_to_dict(summary)
        candidates = []
        candidates.extend(summary_dict.get("display_keywords", []))
        candidates.extend(self.heuristic_keyword_candidates(summary_dict))
        for dim_keywords in summary_dict.get("section_keywords", {}).values():
            candidates.extend(dim_keywords)
        for chunk in summary_dict.get("story_chunks", []):
            candidates.extend(chunk.get("keywords", []))

        normalized = []
        for token in candidates:
            normalized_token = self._normalize_keyword(token, max_chars=14)
            if self._is_good_keyword(normalized_token):
                normalized.append(normalized_token)

        scored = sorted(dedupe(normalized), key=lambda item: self.system._keyword_rank_score(item, summary_dict), reverse=True)
        return scored[:KEYWORD_CANDIDATE_LIMIT]

    def build_keyword_options(self, summary: dict) -> list:
        summary_dict = self.system._summary_to_dict(summary)
        candidates = self.selected_keyword_candidates(summary_dict)
        if not candidates:
            fallback = []
            for dim in ("personality", "moral_qualities", "relationship_style", "role_identity", "values", "worldview"):
                fallback.extend(summary_dict.get(dim, []))
            candidates = []
            for item in fallback:
                token = self._normalize_keyword(item, max_chars=14)
                if self._is_good_keyword(token):
                    candidates.append(token)
            candidates = dedupe(candidates, limit=KEYWORD_CANDIDATE_LIMIT)
        return [PersonaKeywordOption(source="combined", title="综合提炼的角色关键词", keywords=candidates[:KEYWORD_CANDIDATE_LIMIT])]

    def get_display_keywords(self, limit: int = DISPLAY_KEYWORD_LIMIT) -> list[str]:
        chosen = dedupe(self.system.selected_keywords, limit=limit)
        if chosen:
            return chosen
        return dedupe(self.system.display_keywords, limit=limit)

    def ensure_index(self, dim: int):
        if self.system.index is None or self.system.index_dim != dim:
            self.system.index = faiss.IndexFlatIP(dim)
            self.system.index_dim = dim

    def rebuild_index_from_entries(self):
        embeddings = [np.asarray(entry["embedding"], dtype=np.float32) for entry in self.system.entries if entry.get("embedding") is not None]
        if not embeddings:
            self.system.index = None
            self.system.index_dim = None
            return
        dim = len(embeddings[0])
        self.ensure_index(dim)
        self.system.index.reset()
        self.system.index.add(np.asarray(embeddings, dtype=np.float32))

    def append_entries_without_embeddings(self, pending_entries: list[dict], source_fingerprint: str):
        self.system.entries.extend(pending_entries)
        if source_fingerprint:
            self.system.source_records[source_fingerprint] = {
                "source_label": pending_entries[0].get("source_label", ""),
                "updated_at": datetime.now().isoformat(),
            }
        self.system._dedupe_storage()

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
            self.system._merge_summary_data(summary_dict)
            self.append_core_summary(summary_dict, source_label)

        pending_entries = []
        for chunk in self.system._chunk_text(raw_text):
            pending_entries.append(
                {"text": chunk, "source_label": source_label, "kind": "source_chunk", "priority": self.system._entry_priority(chunk, "source_chunk"), "keywords": self.system._fact_to_keywords(chunk), "embedding": None}
            )

        if summary_dict:
            base_template_text = self.system._base_template_text(summary_dict)
            if base_template_text:
                pending_entries.append(
                    {"text": base_template_text, "source_label": source_label, "kind": "base_template", "priority": self.system._entry_priority(base_template_text, "base_template"), "keywords": dedupe(summary_dict.get("display_keywords", []), limit=12), "embedding": None}
                )
            summary_text = self.system._summary_to_text(summary_dict)
            if summary_text:
                pending_entries.append(
                    {"text": summary_text, "source_label": source_label, "kind": "core_summary", "priority": self.system._entry_priority(summary_text, "core_summary"), "keywords": dedupe(summary_dict.get("display_keywords", []), limit=12), "embedding": None}
                )
            for chunk in summary_dict.get("story_chunks", []):
                story_text = self.system._story_chunk_to_text(chunk)
                if story_text:
                    pending_entries.append(
                        {"text": story_text, "source_label": source_label, "kind": "story_chunk", "priority": self.system._entry_priority(story_text, "story_chunk"), "keywords": dedupe(chunk.get("keywords", []) + chunk.get("trigger_topics", []), limit=10), "embedding": None}
                    )

        if not pending_entries:
            if source_fingerprint:
                self.system.source_records[source_fingerprint] = {"source_label": source_label, "updated_at": datetime.now().isoformat()}
            self.system._emit_progress(progress_callback, 100, 100, "done", "没有新增内容，已完成")
            return 0

        total_entries = len(pending_entries)
        self.system._emit_progress(progress_callback, 35, 100, "embedding", f"正在为 {total_entries} 条人设内容生成向量")

        def embedding_progress(current, total, stage, detail):
            ratio = current / (total or total_entries or 1)
            overall = 35 + int(ratio * 50)
            self.system._emit_progress(progress_callback, min(overall, 88), 100, stage, detail)

        try:
            embeddings = mistral_embed_texts([entry["text"] for entry in pending_entries], progress_callback=embedding_progress)
        except MistralEmbeddingCapacityError:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 条（纯文本检索模式）")
            return len(pending_entries)
        except Exception:
            self.append_entries_without_embeddings(pending_entries, source_fingerprint)
            self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 条（纯文本检索模式）")
            return len(pending_entries)

        clean_embeddings = []
        for entry, embedding in zip(pending_entries, embeddings):
            normalized = self.system.normalize_vector(embedding).tolist()
            entry["embedding"] = normalized
            clean_embeddings.append(np.asarray(normalized, dtype=np.float32))

        self.system.entries.extend(pending_entries)
        if clean_embeddings:
            dim = len(clean_embeddings[0])
            self.ensure_index(dim)
            self.system.index.add(np.asarray(clean_embeddings, dtype=np.float32))

        if source_fingerprint:
            self.system.source_records[source_fingerprint] = {"source_label": source_label, "updated_at": datetime.now().isoformat()}
        self.system._dedupe_storage()
        self.rebuild_index_from_entries()
        self.system._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 条")
        return len(pending_entries)

    def load_text(self, raw_text: str, source_label: str = "manual_input", progress_callback=None) -> int:
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            return 0
        summary = self.summarize_with_llm(raw_text, source_label)
        return self.commit_summary_and_entries(raw_text, source_label, summary, progress_callback=progress_callback)

    def load_file(self, filepath: str, progress_callback=None) -> int:
        path = os.fspath(filepath)
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
            try:
                with open(path, "r", encoding=encoding) as handle:
                    return self.load_text(handle.read(), source_label=os.path.basename(path), progress_callback=progress_callback)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法读取文件：{filepath}")

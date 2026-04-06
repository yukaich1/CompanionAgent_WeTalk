from __future__ import annotations

import hashlib
import json
import re

import faiss

from knowledge.persona_conflict_filter import PersonaConflictFilter
from knowledge.persona_context_service import PersonaContextService
from knowledge.persona_ingest_service import PersonaIngestService
from knowledge.persona_preview_service import PersonaPreviewService
from knowledge.persona_shared import (
    AUDIENCE_PERSPECTIVE_WORDS,
    DIMENSION_ORDER,
    DIMENSION_TITLES_ZH,
    DISPLAY_KEYWORD_LIMIT,
    DYNAMIC_KEYWORD_RE,
    META_EXCLUSION_WORDS,
    META_REACTION_WORDS,
    TRAIT_SPLIT_RE,
    dedupe,
    normalize_vector,
)
from knowledge.persona_state import (
    AbsoluteTaboo,
    AttitudeTowardUser,
    ChildChunk,
    CoreTrait,
    EvidenceVault,
    GrowthLogEntry,
    IdentityProfile,
    ImmutableCore,
    InnateBelief,
    ParentChunk,
    PersonaState,
    PersonaSystemStore,
    SlowChangeLayer,
    SpeechDNA,
)
from llm import FallbackMistralLLM
from persona_models import PersonaSummaryModel, PersonaStatusModel


class PersonaSystem:
    def __init__(self, persona_name: str = "Ireina"):
        self.persona_name = persona_name or "Ireina"
        self.base_template = self._empty_base_template()
        self.entries: list[dict] = []
        self.source_records: dict[str, dict] = {}
        self.pending_previews: dict[str, dict] = {}
        self.display_keywords: list[str] = []
        self.style_examples: list[dict] = []
        self.character_voice_card = ""
        self.last_summary_debug: dict = {}
        self.index = None
        self.index_dim = None
        self.model = FallbackMistralLLM()
        self.conflict_filter = PersonaConflictFilter()
        self.dimension_titles = DIMENSION_TITLES_ZH
        self.dynamic_keyword_re = DYNAMIC_KEYWORD_RE
        self.normalize_vector = normalize_vector
        self._setup_runtime_services()

    @property
    def chunk_count(self) -> int:
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_serialized_index"] = faiss.serialize_index(self.index) if self.index is not None else None
        state["index"] = None
        state["model"] = None
        state["context_service"] = None
        state["ingest_service"] = None
        state["preview_service"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        serialized = self.__dict__.pop("_serialized_index", None)
        try:
            self.index = faiss.deserialize_index(serialized) if serialized is not None else None
        except Exception:
            self.index = None
        self.model = FallbackMistralLLM()
        self.dimension_titles = DIMENSION_TITLES_ZH
        self.dynamic_keyword_re = DYNAMIC_KEYWORD_RE
        self.normalize_vector = normalize_vector
        self._setup_runtime_services()
        self._repair_state()

    def _setup_runtime_services(self):
        self.context_service = PersonaContextService(self)
        self.ingest_service = PersonaIngestService(self)
        self.preview_service = PersonaPreviewService(self)

    def _empty_base_template(self):
        template = {}
        for dim in DIMENSION_ORDER:
            if dim == "00_BACKGROUND":
                template[dim] = {"profile": {}, "key_experiences": [], "confidence": ""}
            elif dim in {"E_LIKES", "F_DISLIKES_AND_TABOOS"}:
                template[dim] = {"items": [], "confidence": ""}
            elif dim in {"B_CATCHPHRASES", "G_AVOID_PATTERNS"}:
                template[dim] = {"patterns": [], "confidence": ""}
            else:
                template[dim] = {"rules": [], "confidence": ""}
        return template

    def _repair_state(self):
        self.base_template = self.base_template if isinstance(getattr(self, "base_template", None), dict) else self._empty_base_template()
        for dim, payload in self._empty_base_template().items():
            self.base_template.setdefault(dim, payload)
        self.entries = list(getattr(self, "entries", []) or [])
        self.source_records = dict(getattr(self, "source_records", {}) or {})
        self.pending_previews = dict(getattr(self, "pending_previews", {}) or {})
        if not isinstance(getattr(self, "display_keywords", None), list):
            self.display_keywords = []
        self.style_examples = [
            item if isinstance(item, dict) else {
                "text": str(item),
                "scene": "",
                "emotion": "",
                "rules_applied": [],
                "source": "legacy",
                "affinity_level": "any",
            }
            for item in list(getattr(self, "style_examples", []) or [])
        ]
        self.character_voice_card = str(getattr(self, "character_voice_card", "") or "")
        if not hasattr(self, "index_dim"):
            self.index_dim = None
        if not hasattr(self, "conflict_filter") or self.conflict_filter is None:
            self.conflict_filter = PersonaConflictFilter()
        self._setup_runtime_services()
        self._dedupe_storage()
        if self.index is None and self.entries:
            self.ingest_service.rebuild_index_from_entries()

    def clear(self):
        self.base_template = self._empty_base_template()
        self.entries = []
        self.source_records = {}
        self.pending_previews = {}
        self.display_keywords = []
        self.style_examples = []
        self.character_voice_card = ""
        self.index = None
        self.index_dim = None

    def _summary_to_dict(self, summary):
        if summary is None:
            return {}
        if isinstance(summary, PersonaSummaryModel):
            return summary.dict()
        return dict(summary)

    def _emit_progress(self, callback, current, total, stage, detail):
        if callback:
            callback(current=current, total=total, stage=stage, detail=detail)

    def _canonicalize_source_text(self, text):
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _source_fingerprint(self, text):
        canonical = self._canonicalize_source_text(text)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest() if canonical else ""

    def _normalize_fact(self, text, max_chars: int = 80):
        text = re.sub(r"\s+", " ", str(text or ""))
        text = text.strip(" \t\r\n-:：；，。！？\"'()[]{}")
        return text[:max_chars].rstrip() if len(text) > max_chars else text

    def _normalize_display_keyword(self, token: str, max_chars: int = 12):
        token = self._normalize_fact(token, max_chars=max_chars)
        token = token.strip(" \"'[]()（）【】「」『』《》、，。！？；：")
        token = re.sub(r"^(具有|带有|偏向|呈现|表现出|显得|属于|是一种)", "", token)
        token = re.sub(r"(倾向|风格|气质|感觉|意味|效果|表现|使用|表达)$", "", token)
        token = re.sub(r"(一般来说|本身|方面)$", "", token)
        return token.strip()

    def _split_sentences(self, text):
        return [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", text or "") if part.strip()]

    def _is_meta_commentary(self, text):
        text = (text or "").strip()
        return (not text) or any(word in text for word in META_EXCLUSION_WORDS) or (
            any(word in text for word in AUDIENCE_PERSPECTIVE_WORDS) and any(word in text for word in META_REACTION_WORDS)
        )

    def _fact_to_keywords(self, value):
        text = self._normalize_fact(value, max_chars=24)
        pieces = [text] + [part.strip(" :：；，。！？、\"'()（）") for part in TRAIT_SPLIT_RE.split(text)]
        result = []
        noise = {"这个", "那个", "一种", "一个", "自己", "别人", "因为", "所以", "如果", "但是", "然后", "感觉", "有点", "其实", "真的", "非常", "比较", "不是", "可以"}
        for piece in pieces:
            for token in DYNAMIC_KEYWORD_RE.findall(piece):
                if token in noise:
                    continue
                result.append(token)
        return dedupe(result, limit=8)

    def _dedupe_storage(self):
        seen = set()
        new_entries = []
        for entry in self.entries:
            text = self._canonicalize_source_text(entry.get("content", "") or entry.get("text", ""))
            kind = entry.get("kind", entry.get("metadata", {}).get("kind", "source_chunk"))
            if kind != "source_chunk":
                continue
            if not text or (kind, text) in seen:
                continue
            seen.add((kind, text))
            entry["content"] = text
            entry.pop("text", None)
            entry["kind"] = kind
            entry.setdefault("priority", self._entry_priority(text, kind))
            entry.setdefault("keywords", self._fact_to_keywords(text))
            entry.setdefault("chunk_id", entry.get("chunk_id") or hashlib.sha1(f"{kind}:{text}".encode("utf-8")).hexdigest()[:12])
            entry.setdefault("document_id", entry.get("document_id") or entry["chunk_id"])
            entry["metadata"] = {
                **dict(entry.get("metadata", {}) or {}),
                "kind": kind,
                "title": str(entry.get("title", "") or entry.get("metadata", {}).get("title", "") or ""),
            }
            new_entries.append(entry)
        self.entries = new_entries
        self.source_records = {k: v for k, v in self.source_records.items() if k}

        example_seen = set()
        new_examples = []
        for item in self.style_examples:
            text = self._canonicalize_source_text(item.get("text", "") if isinstance(item, dict) else str(item))
            if not text or text in example_seen:
                continue
            example_seen.add(text)
            new_examples.append(item)
        self.style_examples = new_examples[:18]

    def _normalize_summary(self, summary: dict):
        normalized = {"base_template": {}}
        base_template_raw = summary.get("base_template", {})
        for dim in DIMENSION_ORDER:
            dim_data = base_template_raw.get(dim, {})
            if dim == "00_BACKGROUND":
                profile = {}
                for key, value in (dim_data.get("profile", {}) if isinstance(dim_data, dict) else {}).items():
                    clean = self._normalize_fact(value, max_chars=120)
                    if clean and not self._is_meta_commentary(clean):
                        profile[str(key)] = clean
                experiences = [self._normalize_fact(item, max_chars=160) for item in dim_data.get("key_experiences", []) if item and not self._is_meta_commentary(item)]
                normalized["base_template"][dim] = {"profile": profile, "key_experiences": dedupe(experiences, limit=8), "confidence": dim_data.get("confidence", "")}
            elif dim in {"E_LIKES", "F_DISLIKES_AND_TABOOS"}:
                items = []
                for item in dim_data.get("items", []):
                    if isinstance(item, dict):
                        label = self._normalize_fact(item.get("item", ""), max_chars=24)
                        behavior = self._normalize_fact(item.get("behavior", ""), max_chars=160)
                        level = self._normalize_fact(item.get("level", ""), max_chars=20)
                        if label and not self._is_meta_commentary(label):
                            payload = {"item": label, "behavior": behavior}
                            if level:
                                payload["level"] = level
                            items.append(payload)
                    else:
                        label = self._normalize_fact(item, max_chars=40)
                        if label and not self._is_meta_commentary(label):
                            items.append({"item": label, "behavior": ""})
                normalized["base_template"][dim] = {"items": items[:8], "confidence": dim_data.get("confidence", "")}
            elif dim == "B_CATCHPHRASES":
                patterns = []
                for pattern in dim_data.get("patterns", []):
                    if not isinstance(pattern, dict):
                        continue
                    text = self._normalize_fact(pattern.get("pattern", ""), max_chars=80)
                    usage = self._normalize_fact(pattern.get("usage", ""), max_chars=80)
                    tone = self._normalize_fact(pattern.get("tone", ""), max_chars=80)
                    if text and not self._is_meta_commentary(text):
                        patterns.append({"pattern": text, "usage": usage, "tone": tone})
                normalized["base_template"][dim] = {"patterns": patterns[:8], "confidence": dim_data.get("confidence", "")}
            elif dim == "G_AVOID_PATTERNS":
                patterns = []
                for pattern in dim_data.get("patterns", []):
                    if not isinstance(pattern, dict):
                        continue
                    text = self._normalize_fact(pattern.get("pattern", ""), max_chars=80)
                    reason = self._normalize_fact(pattern.get("reason", ""), max_chars=120)
                    alternative = self._normalize_fact(pattern.get("alternative", ""), max_chars=120)
                    if text and not self._is_meta_commentary(text):
                        patterns.append({"pattern": text, "reason": reason, "alternative": alternative})
                normalized["base_template"][dim] = {"patterns": patterns[:8], "confidence": dim_data.get("confidence", "")}
            else:
                rules = [self._normalize_fact(rule, max_chars=200) for rule in dim_data.get("rules", []) if rule and not self._is_meta_commentary(rule)]
                normalized["base_template"][dim] = {"rules": dedupe(rules, limit=5), "confidence": dim_data.get("confidence", "")}

        normalized["character_voice_card"] = self._normalize_fact(summary.get("character_voice_card", ""), max_chars=600)
        examples = []
        for example in summary.get("style_examples", []):
            if not isinstance(example, dict):
                continue
            text = self._normalize_fact(example.get("text", ""), max_chars=200)
            if not text or self._is_meta_commentary(text):
                continue
            examples.append({
                "text": text,
                "scene": self._normalize_fact(example.get("scene", ""), max_chars=40),
                "emotion": self._normalize_fact(example.get("emotion", ""), max_chars=30),
                "rules_applied": dedupe([self._normalize_fact(rule, max_chars=40) for rule in example.get("rules_applied", [])], limit=3),
                "source": example.get("source", ""),
                "affinity_level": example.get("affinity_level", "any"),
            })
        normalized["style_examples"] = examples[:18]
        normalized["display_keywords"] = dedupe(
            [
                kw
                for kw in (self._normalize_display_keyword(value, max_chars=16) for value in summary.get("display_keywords", []))
                if kw and not self._is_meta_commentary(kw)
            ],
            limit=DISPLAY_KEYWORD_LIMIT,
        )

        return normalized

    def _merge_summary_data(self, summary):
        summary = self._summary_to_dict(summary)
        base_template = summary.get("base_template", {})
        for dim in DIMENSION_ORDER:
            dim_data = base_template.get(dim, {})
            if dim == "00_BACKGROUND":
                existing_profile = self.base_template[dim].setdefault("profile", {})
                for key, value in dim_data.get("profile", {}).items():
                    if value and key not in existing_profile:
                        existing_profile[key] = value
                self.base_template[dim]["key_experiences"] = dedupe(self.base_template[dim].get("key_experiences", []) + dim_data.get("key_experiences", []), limit=8)
            elif dim in {"E_LIKES", "F_DISLIKES_AND_TABOOS"}:
                existing = self.base_template[dim].get("items", [])
                seen = {item.get("item") for item in existing if isinstance(item, dict)}
                for item in dim_data.get("items", []):
                    label = item.get("item") if isinstance(item, dict) else None
                    if label and label not in seen:
                        existing.append(item)
                        seen.add(label)
                self.base_template[dim]["items"] = existing[:8]
            elif dim == "B_CATCHPHRASES":
                existing = self.base_template[dim].get("patterns", [])
                seen = {pattern["pattern"] for pattern in existing if isinstance(pattern, dict) and pattern.get("pattern")}
                for pattern in dim_data.get("patterns", []):
                    if isinstance(pattern, dict) and pattern.get("pattern") and pattern["pattern"] not in seen:
                        existing.append(pattern)
                        seen.add(pattern["pattern"])
                self.base_template[dim]["patterns"] = existing[:8]
            elif dim == "G_AVOID_PATTERNS":
                existing = self.base_template[dim].get("patterns", [])
                seen = {pattern["pattern"] for pattern in existing if isinstance(pattern, dict) and pattern.get("pattern")}
                for pattern in dim_data.get("patterns", []):
                    if isinstance(pattern, dict) and pattern.get("pattern") and pattern["pattern"] not in seen:
                        existing.append(pattern)
                        seen.add(pattern["pattern"])
                self.base_template[dim]["patterns"] = existing[:8]
            else:
                self.base_template[dim]["rules"] = dedupe(self.base_template[dim].get("rules", []) + dim_data.get("rules", []), limit=5)
            if dim_data.get("confidence"):
                self.base_template[dim]["confidence"] = dim_data["confidence"]

        existing_texts = {example.get("text", "") for example in self.style_examples if isinstance(example, dict)}
        for example in summary.get("style_examples", []):
            if isinstance(example, dict) and example.get("text") and example["text"] not in existing_texts:
                self.style_examples.append(example)
                existing_texts.add(example["text"])
        self.style_examples = self.style_examples[:18]
        self.display_keywords = dedupe(self.display_keywords + summary.get("display_keywords", []), limit=DISPLAY_KEYWORD_LIMIT)
        new_card = summary.get("character_voice_card", "")
        if new_card:
            self.character_voice_card = new_card if not self.character_voice_card else (self._merge_voice_cards(self.character_voice_card, new_card) or new_card)

    def _merge_voice_cards(self, old_card: str, new_card: str) -> str:
        prompt = (
            "下面是同一个角色的两段说话方式描述，请合并成一段简洁、稳定、可直接用于扮演的角色声音底稿。"
            "要求保留最有辨识度的说话节奏、距离感和价值取向，去掉重复和分析性描述。\n\n"
            f"描述A：{old_card}\n\n描述B：{new_card}\n\n请直接输出合并后的结果。"
        )
        try:
            result = self.model.generate(prompt, temperature=0.1, max_tokens=300)
            return self._normalize_fact(str(result or ""), max_chars=600)
        except Exception:
            return ""

    def _base_template_text(self, summary: dict):
        payload = {
            "character_name": summary.get("character_name", self.persona_name),
            "source_label": summary.get("source_label", ""),
            "base_template": summary.get("base_template", {}),
            "character_voice_card": summary.get("character_voice_card", ""),
            "display_keywords": summary.get("display_keywords", []),
            "style_examples": summary.get("style_examples", []),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _entry_priority(self, text, kind) -> float:
        return 1.0

    def _looks_like_persona_chunk(self, text) -> int:
        if self._is_meta_commentary(text):
            return -5
        score = 2 * len(self._fact_to_keywords(text))
        if '"' in text or "“" in text or "”" in text:
            score += 2
        if len(text) < 16:
            score -= 1
        return score

    def _chunk_text(self, raw_text, target_chars: int = 420, hard_limit: int = 650):
        markdown = self.ingest_service.rag.convert_text_to_markdown(str(raw_text or ""), title=self.persona_name)
        chunks = self.ingest_service.rag.build_chunks(
            markdown,
            document_id="summary-preview",
            source_label=self.persona_name,
            source_type="persona",
            priority=1.0,
            metadata={"kind": "summary_preview"},
        )
        return dedupe([chunk.get("content", "") for chunk in chunks if chunk.get("content")], limit=max(1, hard_limit))

    def _prepare_summary_source(self, raw_text, max_chars: int = 1800):
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            return ""

        chunks = [str(chunk or "").strip() for chunk in self._chunk_text(raw_text) if str(chunk or "").strip()]
        ranked_chunks = sorted(chunks, key=self._looks_like_persona_chunk, reverse=True)

        selected: list[str] = []
        total = 0
        for chunk in ranked_chunks[:8]:
            compact = chunk[:320].strip()
            if not compact:
                continue
            projected = total + len(compact) + 2
            if selected and projected > max_chars:
                continue
            selected.append(compact)
            total = projected
            if len(selected) >= 5 or total >= max_chars:
                break

        if not selected:
            return raw_text[:max_chars]
        return "\n\n".join(selected)[:max_chars]

    def build_precise_query_context(self, query: str, top_k: int = 5, char_budget: int = 700) -> str:
        return self.context_service.build_precise_query_context(query, top_k=top_k, char_budget=char_budget)

    def _summarize_with_llm(self, raw_text: str, source_label: str):
        return self.ingest_service.summarize_with_llm(raw_text, source_label)

    def get_display_keywords(self, limit: int = DISPLAY_KEYWORD_LIMIT):
        return self.ingest_service.get_display_keywords(limit=limit)

    def _commit_summary_and_entries(self, raw_text: str, source_label: str, summary: dict | None, progress_callback=None):
        return self.ingest_service.commit_summary_and_entries(raw_text, source_label, summary, progress_callback=progress_callback)

    def load_text(self, raw_text: str, source_label: str = "manual_input", progress_callback=None):
        return self.ingest_service.load_text(raw_text, source_label=source_label, progress_callback=progress_callback)

    def load_file(self, filepath: str, progress_callback=None):
        return self.ingest_service.load_file(filepath, progress_callback=progress_callback)

    def get_status(self):
        return PersonaStatusModel(persona_name=self.persona_name, chunk_count=self.chunk_count, display_keywords=self.get_display_keywords())

    def preview_from_sources(self, persona_name: str = "", work_title: str = "", local_text: str = "", local_label: str = "manual_input", max_results: int = 5, local_snippets: list | None = None, enable_web_search: bool = True):
        return self.preview_service.preview_from_sources(
            persona_name=persona_name,
            work_title=work_title,
            local_text=local_text,
            local_label=local_label,
            max_results=max_results,
            local_snippets=local_snippets,
            enable_web_search=enable_web_search,
        )

    def confirm_preview(self, preview_id: str, selected_keywords: list[str] | None = None, progress_callback=None):
        return self.preview_service.confirm_preview(preview_id, selected_keywords=selected_keywords, progress_callback=progress_callback)

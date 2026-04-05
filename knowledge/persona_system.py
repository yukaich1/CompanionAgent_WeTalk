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
    COMPACT_TEMPLATE_DIMENSIONS,
    DIMENSION_ORDER,
    DIMENSION_TITLES_ZH,
    DISPLAY_KEYWORD_LIMIT,
    DYNAMIC_KEYWORD_RE,
    KEYWORD_STOPWORDS,
    META_EXCLUSION_WORDS,
    META_REACTION_WORDS,
    PREFERRED_PERSONA_LABELS,
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
from persona_models import PersonaPreviewModel, PersonaStatusModel, PersonaSummaryModel


class PersonaSystem:
    def __init__(self, persona_name: str = "Ireina"):
        self.persona_name = persona_name or "Ireina"
        self.base_template = self._empty_base_template()
        self.entries = []
        self.source_records = {}
        self.pending_previews = {}
        self.display_keywords = []
        self.selected_keywords = []
        self.style_examples = []
        self.natural_reference_triggers = []
        self.character_voice_card = ""
        self.story_chunks = []
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
            if dim == "00_BACKGROUND_PROFILE":
                template[dim] = {"profile": {}, "key_experiences": [], "confidence": ""}
            elif dim in {"17_LIKES_AND_PREFERENCES", "18_DISLIKES_AND_TABOOS"}:
                template[dim] = {"items": [], "confidence": ""}
            elif dim == "19_AVOID_PATTERNS":
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
        for field in ("display_keywords", "selected_keywords", "natural_reference_triggers", "story_chunks"):
            if not isinstance(getattr(self, field, None), list):
                setattr(self, field, [])
        self.style_examples = [item if isinstance(item, dict) else {"text": str(item), "scene": "", "emotion": "", "rules_applied": [], "source": "legacy", "affinity_level": "any"} for item in list(getattr(self, "style_examples", []) or [])]
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
        self.selected_keywords = []
        self.style_examples = []
        self.natural_reference_triggers = []
        self.character_voice_card = ""
        self.story_chunks = []
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
        text = text.strip(" \t\r\n-:：；，。！？?\"'()[]{}")
        return text[:max_chars].rstrip() if len(text) > max_chars else text

    def _normalize_display_keyword(self, token: str, max_chars: int = 12):
        token = self._normalize_fact(token, max_chars=max_chars)
        token = token.strip(" \"'[]()（）【】「」『』《》、，。！？?；：")
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

    def _is_display_keyword_candidate(self, token):
        token = self._normalize_display_keyword(token, max_chars=12)
        if token in PREFERRED_PERSONA_LABELS:
            return True
        if len(token) < 2 or len(token) > 10 or token in KEYWORD_STOPWORDS:
            return False
        if token.startswith(("如", "以", "来", "把", "说", "因", "对", "将", "被", "而", "并")):
            return False
        if any(marker in token for marker in ("的", "是", "在", "和", "中", "被", "把", "说", "如果", "但是", "因为", "所以")):
            return False
        return not bool(re.search(r"[A-Za-z0-9,\"'()\[\]{}]", token))

    def _keyword_rank_score(self, token, summary=None):
        token = self._normalize_display_keyword(token, max_chars=12)
        if not token:
            return 0.0
        summary_dict = self._summary_to_dict(summary) if summary is not None else {}
        score = 0.0
        matched_dims = 0
        for dim, weight in [("01_PERSONALITY_CORE", 3.0), ("03_TONE_LAYER", 2.8), ("12_WORLDVIEW_ASSUMPTION", 2.6), ("13_VALUE_EXPRESSION", 2.4), ("08_EMOTION_EXPRESSION_PATH", 2.2), ("15_RELATIONSHIP_DYNAMICS", 2.0), ("02_SPEECH_SURFACE", 1.8)]:
            rules = summary_dict.get("base_template", {}).get(dim, {}).get("rules", [])
            if any(token in rule for rule in rules):
                score += weight
                matched_dims += 1
        if token in summary_dict.get("display_keywords", []):
            score += 3.2
        if any(token in item.get("keywords", []) for item in summary_dict.get("story_chunks", [])):
            score += 1.0
        if token in summary_dict.get("character_voice_card", ""):
            score += 1.0
        if matched_dims >= 2:
            score += 0.8
        if 2 <= len(token) <= 4:
            score += 0.4
        return score

    def _fact_to_keywords(self, value):
        text = self._normalize_fact(value, max_chars=24)
        pieces = [text] + [part.strip(" :：；，。！？?、\"'()（）") for part in TRAIT_SPLIT_RE.split(text)]
        result = []
        for piece in pieces:
            for token in DYNAMIC_KEYWORD_RE.findall(piece):
                if token in {"这个", "那个", "一种", "一个", "自己", "别人", "因为", "所以", "如果", "但是", "然后", "感觉", "有点", "其实", "真的", "非常", "比较", "不是", "可以"}:
                    continue
                result.append(token)
        return dedupe(result, limit=8)

    def _dedupe_storage(self):
        seen = set()
        new_entries = []
        for entry in self.entries:
            text = self._canonicalize_source_text(entry.get("text", ""))
            kind = entry.get("kind", "chunk")
            if not text or (kind, text) in seen:
                continue
            seen.add((kind, text))
            entry["text"] = text
            entry.setdefault("kind", kind)
            entry.setdefault("priority", self._entry_priority(text, kind))
            new_entries.append(entry)
        self.entries = new_entries
        self.source_records = {k: v for k, v in self.source_records.items() if k}
        story_seen = set()
        cleaned_stories = []
        for item in self.story_chunks:
            title = self._normalize_fact(item.get("title", ""), max_chars=32)
            content = self._canonicalize_source_text(item.get("content", "") or item.get("summary", ""))
            if not content or (title, content) in story_seen:
                continue
            story_seen.add((title, content))
            item["content"] = content
            item["keywords"] = dedupe([self._normalize_fact(keyword, max_chars=16) for keyword in item.get("keywords", [])], limit=6)
            item["trigger_topics"] = dedupe([self._normalize_fact(topic, max_chars=20) for topic in item.get("trigger_topics", [])], limit=4)
            item.setdefault("emotional_weight", "medium")
            item.setdefault("character_impact", "")
            item.setdefault("source_confidence", item.get("source_hint", "推断"))
            cleaned_stories.append(item)
        self.story_chunks = cleaned_stories
        ex_seen = set()
        new_examples = []
        for item in self.style_examples:
            text = self._canonicalize_source_text(item.get("text", "") if isinstance(item, dict) else str(item))
            if not text or text in ex_seen:
                continue
            ex_seen.add(text)
            new_examples.append(item)
        self.style_examples = new_examples[:18]

    def _normalize_summary(self, summary: dict):
        normalized = {"base_template": {}}
        base_template_raw = summary.get("base_template", {})
        for dim in DIMENSION_ORDER:
            dim_data = base_template_raw.get(dim, {})
            if dim == "00_BACKGROUND_PROFILE":
                profile = {}
                for key, value in (dim_data.get("profile", {}) if isinstance(dim_data, dict) else {}).items():
                    clean = self._normalize_fact(value, max_chars=120)
                    if clean and not self._is_meta_commentary(clean):
                        profile[str(key)] = clean
                experiences = [self._normalize_fact(item, max_chars=160) for item in dim_data.get("key_experiences", []) if item and not self._is_meta_commentary(item)]
                normalized["base_template"][dim] = {"profile": profile, "key_experiences": dedupe(experiences, limit=8), "confidence": dim_data.get("confidence", "")}
            elif dim in {"17_LIKES_AND_PREFERENCES", "18_DISLIKES_AND_TABOOS"}:
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
            elif dim == "19_AVOID_PATTERNS":
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
            examples.append({"text": text, "scene": self._normalize_fact(example.get("scene", ""), max_chars=40), "emotion": self._normalize_fact(example.get("emotion", ""), max_chars=30), "rules_applied": dedupe([self._normalize_fact(rule, max_chars=40) for rule in example.get("rules_applied", [])], limit=3), "source": example.get("source", ""), "affinity_level": example.get("affinity_level", "any")})
        normalized["style_examples"] = examples[:18]
        normalized["natural_reference_triggers"] = dedupe([self._normalize_fact(value, max_chars=30) for value in summary.get("natural_reference_triggers", []) if value], limit=10)
        normalized["display_keywords"] = dedupe([kw for kw in (self._normalize_display_keyword(value, max_chars=12) for value in summary.get("display_keywords", [])) if kw and self._is_display_keyword_candidate(kw) and not self._is_meta_commentary(kw)], limit=DISPLAY_KEYWORD_LIMIT)
        stories = []
        for item in summary.get("story_chunks", []) or []:
            content = self._normalize_fact(item.get("content", "") or item.get("summary", ""), max_chars=300)
            if not content or self._is_meta_commentary(content):
                continue
            keywords = dedupe([value for value in (self._normalize_fact(raw, max_chars=16) for raw in item.get("keywords", [])) if value and not self._is_meta_commentary(value)], limit=6) or self._fact_to_keywords(content)
            stories.append({"story_id": item.get("story_id", hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]), "title": self._normalize_fact(item.get("title", ""), max_chars=20), "content": content, "keywords": keywords, "emotional_weight": item.get("emotional_weight", "medium"), "character_impact": self._normalize_fact(item.get("character_impact", ""), max_chars=100), "trigger_topics": dedupe([self._normalize_fact(value, max_chars=20) for value in item.get("trigger_topics", []) if value], limit=4), "source_confidence": item.get("source_confidence", "")})
        normalized["story_chunks"] = stories[:12]
        return normalized

    def _merge_summary_data(self, summary):
        summary = self._summary_to_dict(summary)
        base_template = summary.get("base_template", {})
        for dim in DIMENSION_ORDER:
            dim_data = base_template.get(dim, {})
            if dim == "00_BACKGROUND_PROFILE":
                existing_profile = self.base_template[dim].setdefault("profile", {})
                for key, value in dim_data.get("profile", {}).items():
                    if value and key not in existing_profile:
                        existing_profile[key] = value
                self.base_template[dim]["key_experiences"] = dedupe(self.base_template[dim].get("key_experiences", []) + dim_data.get("key_experiences", []), limit=8)
            elif dim in {"17_LIKES_AND_PREFERENCES", "18_DISLIKES_AND_TABOOS"}:
                existing = self.base_template[dim].get("items", [])
                seen = {item.get("item") for item in existing if isinstance(item, dict)}
                for item in dim_data.get("items", []):
                    label = item.get("item") if isinstance(item, dict) else None
                    if label and label not in seen:
                        existing.append(item)
                        seen.add(label)
                self.base_template[dim]["items"] = existing[:8]
            elif dim == "19_AVOID_PATTERNS":
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
        self.natural_reference_triggers = dedupe(self.natural_reference_triggers + summary.get("natural_reference_triggers", []), limit=12)
        self.display_keywords = dedupe(self.display_keywords + summary.get("display_keywords", []), limit=DISPLAY_KEYWORD_LIMIT)
        new_card = summary.get("character_voice_card", "")
        if new_card:
            self.character_voice_card = new_card if not self.character_voice_card else (self._merge_voice_cards(self.character_voice_card, new_card) or new_card)
        existing_story_keys = {(chunk.get("title", ""), chunk.get("content", "")) for chunk in self.story_chunks}
        for chunk in summary.get("story_chunks", []):
            key = (chunk.get("title", ""), chunk.get("content", ""))
            if key not in existing_story_keys:
                self.story_chunks.append(chunk)
                existing_story_keys.add(key)

    def _merge_voice_cards(self, old_card: str, new_card: str) -> str:
        prompt = (
            "下面是同一个角色的两段说话方式描述，请将它们合并成一段简洁、稳定、可直接用于扮演的角色声音底稿，控制在100到150字。\n"
            "要求保留最有辨识度的说话腔调、距离感和价值取向，去掉重复和解释性分析，只保留角色本人会自然流露出来的说法。\n\n"
            f"描述A：{old_card}\n\n描述B：{new_card}\n\n请直接输出合并后的结果，不要添加前缀。"
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
            "natural_reference_triggers": summary.get("natural_reference_triggers", []),
            "story_chunks": summary.get("story_chunks", []),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _story_chunk_to_text(self, chunk: dict):
        content = chunk.get("content") or chunk.get("summary", "")
        lines = [f"【故事标题】{chunk.get('title', '角色经历')}", f"【故事内容】{content}"]
        if chunk.get("keywords"):
            lines.append("【检索关键词】" + "、".join(chunk["keywords"][:6]))
        if chunk.get("trigger_topics"):
            lines.append("【自然触发话题】" + "、".join(chunk["trigger_topics"][:4]))
        if chunk.get("character_impact"):
            lines.append(f"【对角色的影响】{chunk['character_impact']}")
        if chunk.get("emotional_weight"):
            lines.append(f"【情感权重】{chunk['emotional_weight']}")
        return "\n".join(lines).strip()

    def _summary_to_text(self, summary: dict):
        lines = [f"{self.persona_name} core persona canon:"]
        if summary.get("character_voice_card"):
            lines += ["[Character Voice Card]", summary["character_voice_card"]]
        background = summary.get("base_template", {}).get("00_BACKGROUND_PROFILE", {})
        if background.get("profile"):
            lines.append("[Background Profile]")
            lines.extend(f"- {key}: {value}" for key, value in background.get("profile", {}).items())
        if background.get("key_experiences"):
            lines.append("[Key Experiences]")
            lines.extend(f"- {item}" for item in background.get("key_experiences", [])[:5])
        for dim in COMPACT_TEMPLATE_DIMENSIONS:
            title = DIMENSION_TITLES_ZH.get(dim, dim)
            dim_data = summary.get("base_template", {}).get(dim, {})
            if dim in {"17_LIKES_AND_PREFERENCES", "18_DISLIKES_AND_TABOOS"}:
                items = dim_data.get("items", [])
                if items:
                    lines.append(f"[{title}]")
                    for item in items[:4]:
                        label = item.get("item", "") if isinstance(item, dict) else str(item)
                        behavior = item.get("behavior", "") if isinstance(item, dict) else ""
                        lines.append(f"- {label}" + (f": {behavior}" if behavior else ""))
            elif dim == "19_AVOID_PATTERNS":
                patterns = dim_data.get("patterns", [])
                if patterns:
                    lines.append(f"[{title}]")
                    lines.extend(f"- {pattern.get('pattern', '')}" for pattern in patterns[:4])
            else:
                rules = dim_data.get("rules", [])
                if rules:
                    lines.append(f"[{title}]")
                    lines.extend(f"- {rule}" for rule in rules[:2])
        if summary.get("style_examples"):
            lines.append("[Style Examples]")
            lines.extend(f"- {(example.get('text', '') if isinstance(example, dict) else str(example))}" for example in summary["style_examples"][:6])
        if summary.get("display_keywords"):
            lines += ["[Display Keywords]"] + [f"- {keyword}" for keyword in summary["display_keywords"]]
        if summary.get("story_chunks"):
            lines.append("[Story Units]")
            lines.extend(f"- {chunk.get('title', '故事')}: {chunk.get('content') or chunk.get('summary', '')}" for chunk in summary["story_chunks"][:5])
        return "\n".join(lines).strip()

    def _entry_priority(self, text, kind) -> float:
        return 2.6 if kind == "base_template" else 2.3 if kind == "story_chunk" else 2.0 if kind == "core_summary" else 1.0

    def _looks_like_persona_chunk(self, text) -> int:
        if self._is_meta_commentary(text):
            return -5
        score = 2 * len(self.context_service.detect_relevant_dimensions(text))
        if "“" in text or "”" in text or '"' in text:
            score += 2
        if len(text) < 16:
            score -= 1
        return score

    def _chunk_text(self, raw_text, target_chars: int = 420, hard_limit: int = 650):
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", raw_text or "") if part.strip()]
        if not paragraphs:
            paragraphs = [str(raw_text or "").strip()] if str(raw_text or "").strip() else []
        chunks, current = [], ""
        for paragraph in paragraphs:
            if len(paragraph) <= hard_limit and len(current) + len(paragraph) + 2 <= target_chars:
                current = (current + "\n\n" + paragraph).strip()
                continue
            if current:
                chunks.append(current)
                current = ""
            if len(paragraph) <= hard_limit:
                current = paragraph
                continue
            sentence_chunk = ""
            for sentence in self._split_sentences(paragraph):
                if len(sentence_chunk) + len(sentence) + 1 <= target_chars:
                    sentence_chunk = (sentence_chunk + sentence).strip()
                else:
                    if sentence_chunk:
                        chunks.append(sentence_chunk)
                    sentence_chunk = sentence
            if sentence_chunk:
                current = sentence_chunk
        if current:
            chunks.append(current)
        return dedupe([chunk for chunk in chunks if chunk and len(chunk) >= 6])

    def _prepare_summary_source(self, raw_text, max_chars: int = 16000):
        chunks = sorted(self._chunk_text(raw_text), key=self._looks_like_persona_chunk, reverse=True)
        selected, total = [], 0
        for chunk in chunks:
            projected = total + len(chunk) + 2
            if selected and projected > max_chars:
                continue
            selected.append(chunk)
            total = projected
            if total >= max_chars:
                break
        return ("\n\n".join(selected) if selected else str(raw_text or "")[:max_chars])[:max_chars]

    def build_context(self, query: str) -> str:
        return self.context_service.build_context(query)

    def build_precise_query_context(self, query: str, top_k: int = 5, char_budget: int = 700) -> str:
        return self.context_service.build_precise_query_context(query, top_k=top_k, char_budget=char_budget)

    def build_story_context(self, query: str, top_k: int = 1, char_budget: int = 900) -> str:
        return self.context_service.build_story_context(query, top_k=top_k, char_budget=char_budget)

    def _summarize_with_llm(self, raw_text: str, source_label: str):
        return self.ingest_service.summarize_with_llm(raw_text, source_label)

    def _append_core_summary(self, summary: dict, source_label: str):
        self.ingest_service.append_core_summary(summary, source_label)

    def _build_keyword_options(self, summary: dict):
        return self.ingest_service.build_keyword_options(summary)

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

    def confirm_preview(self, preview_id: str, selected_keywords: list | None = None, progress_callback=None):
        return self.preview_service.confirm_preview(preview_id, selected_keywords=selected_keywords, progress_callback=progress_callback)

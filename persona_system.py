import os
import pickle
import re
import uuid
import hashlib
from datetime import datetime

import faiss
import numpy as np

from const import PERSONA_CONTEXT_CHAR_BUDGET, PERSONA_RETRIEVAL_THRESHOLD, PERSONA_SAVE_PATH
from llm import FallbackMistralLLM, MistralEmbeddingCapacityError, mistral_embed_texts
from persona_models import PersonaPreviewModel, PersonaSourceSnippet, PersonaStatusModel, PersonaSummaryModel
from persona_prompting import build_persona_summary_prompt
from tools import DEFAULT_TOOL_REGISTRY


SECTION_ORDER = [
    "voice_style",
    "catchphrases",
    "addressing_habits",
    "sentence_endings",
    "personality",
    "values",
    "worldview",
    "likes",
    "dislikes",
    "appearance",
    "core_setup",
    "life_experiences",
]

SECTION_TITLES = {
    "voice_style": "Voice and Expression",
    "catchphrases": "Catchphrases",
    "addressing_habits": "Addressing Habits",
    "sentence_endings": "Sentence Endings",
    "personality": "Personality and Temperament",
    "values": "Values",
    "worldview": "Worldview",
    "likes": "Likes",
    "dislikes": "Dislikes",
    "appearance": "Appearance",
    "core_setup": "Core Setup",
    "life_experiences": "Natural Experience Hooks",
}

SECTION_KEYWORDS = {
    "voice_style": ["说话", "语气", "口吻", "表达", "措辞", "自称", "称呼", "台词"],
    "catchphrases": ["口头禅", "常说", "惯用语", "经典台词", "标志性说法", "总会说"],
    "addressing_habits": ["称呼", "自称", "叫法", "怎么称呼", "如何称呼别人", "怎么叫自己"],
    "sentence_endings": ["句末", "尾音", "结尾习惯", "语尾", "句尾", "常用结尾"],
    "personality": ["性格", "特质", "特点", "气质", "作风", "脾气"],
    "values": ["价值观", "原则", "信念", "重视", "底线", "选择"],
    "worldview": ["世界观", "看待", "态度", "想法", "判断", "人生"],
    "likes": ["喜欢", "偏好", "钟爱", "在意", "感兴趣", "偏爱"],
    "dislikes": ["讨厌", "反感", "厌恶", "排斥", "不能接受", "禁忌"],
    "appearance": ["外貌", "外观", "形象", "发色", "衣着", "瞳色"],
    "core_setup": ["身份", "设定", "背景", "来历", "职业", "身份定位"],
    "life_experiences": ["经历", "故事", "曾经", "遇到", "旅途", "过去"],
}

META_EXCLUSION_WORDS = [
    "受众",
    "粉丝",
    "观众",
    "读者",
    "作者",
    "制作组",
    "评价",
    "人气",
    "圈粉",
    "热度",
    "口碑",
    "讨论度",
    "设定外",
    "演出效果",
    "创作意图",
]

NOVEL_PERSONA_HINTS = ["说道", "说着", "语气", "眼神", "神情", "笑了笑", "沉默", "低声", "淡淡地"]
VOICE_HINTS = ["她说", "他说", "语气", "口吻", "自称", "称呼", "淡淡地说", "漫不经心地说"]
TRAIT_MARKERS = ["关键词", "人设", "标签", "特质", "性格", "口吻", "价值观", "世界观", "口头禅", "自称", "称呼"]
AUDIENCE_PERSPECTIVE_WORDS = ["受众", "粉丝", "观众", "读者", "网友", "路人", "玩家", "大家", "别人"]
META_REACTION_WORDS = ["喜欢", "偏爱", "喜爱", "萌点", "圈粉", "吸引", "人气", "受欢迎", "评价", "印象", "讨论"]
TRAIT_SPLIT_RE = re.compile(r"[、/|·,，；;：:\n]+")
DYNAMIC_KEYWORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,12}")
DISPLAY_KEYWORD_LIMIT = 5

PERSONA_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        **{section: {"type": "array", "items": {"type": "string"}} for section in SECTION_ORDER},
        "style_examples": {"type": "array", "items": {"type": "string"}},
        "natural_reference_triggers": {"type": "array", "items": {"type": "string"}},
        "display_keywords": {"type": "array", "items": {"type": "string"}},
        "section_keywords": {
            "type": "object",
            "properties": {section: {"type": "array", "items": {"type": "string"}} for section in SECTION_ORDER},
            "required": SECTION_ORDER,
            "additionalProperties": False,
        },
        "meta_exclusion_words": {"type": "array", "items": {"type": "string"}},
        "novel_persona_hints": {"type": "array", "items": {"type": "string"}},
        "voice_hints": {"type": "array", "items": {"type": "string"}},
        "trait_markers": {"type": "array", "items": {"type": "string"}},
    },
    "required": SECTION_ORDER
    + [
        "style_examples",
        "natural_reference_triggers",
        "display_keywords",
        "section_keywords",
        "meta_exclusion_words",
        "novel_persona_hints",
        "voice_hints",
        "trait_markers",
    ],
    "additionalProperties": False,
}

def _normalize_vector(vector):
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    return array if norm == 0 else array / norm


def _dedupe(items, limit=None):
    seen = set()
    result = []
    for item in items:
        item = (item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if limit and len(result) >= limit:
            break
    return result


class PersonaSystem:
    def __init__(self, persona_name="Ireina", save_path=PERSONA_SAVE_PATH):
        self.persona_name = persona_name
        self.save_path = save_path
        self.profile = {section: [] for section in SECTION_ORDER}
        self.core_summaries = []
        self.entries = []
        self.source_records = {}
        self.pending_previews = {}
        self.dynamic_keywords = {section: [] for section in SECTION_ORDER}
        self.dynamic_meta_exclusions = []
        self.dynamic_novel_hints = []
        self.dynamic_voice_hints = []
        self.dynamic_trait_markers = []
        self.display_keywords = []
        self.index = None
        self.index_dim = None
        self.model = FallbackMistralLLM()
        self._load_from_disk()
        self.persona_name = persona_name or self.persona_name

    @property
    def chunk_count(self):
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_serialized_index"] = faiss.serialize_index(self.index) if self.index is not None else None
        state["index"] = None
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        serialized = self.__dict__.pop("_serialized_index", None)
        try:
            self.index = faiss.deserialize_index(serialized) if serialized is not None else None
        except Exception:
            self.index = None
        self.model = FallbackMistralLLM()
        self._repair_state()

    def _repair_state(self):
        self.profile = self.profile if isinstance(self.profile, dict) else {}
        for section in SECTION_ORDER:
            self.profile.setdefault(section, [])
        self.core_summaries = self.core_summaries if isinstance(self.core_summaries, list) else []
        self.entries = self.entries if isinstance(self.entries, list) else []
        self.source_records = self.source_records if isinstance(getattr(self, "source_records", None), dict) else {}
        self.pending_previews = self.pending_previews if isinstance(getattr(self, "pending_previews", None), dict) else {}
        self.dynamic_keywords = self.dynamic_keywords if isinstance(self.dynamic_keywords, dict) else {}
        for section in SECTION_ORDER:
            self.dynamic_keywords.setdefault(section, [])
        for field in ("dynamic_meta_exclusions", "dynamic_novel_hints", "dynamic_voice_hints", "dynamic_trait_markers", "display_keywords"):
            if not isinstance(getattr(self, field, None), list):
                setattr(self, field, [])
        if not hasattr(self, "index_dim"):
            self.index_dim = None
        self._dedupe_storage()
        if self.index is None and self.entries:
            self._rebuild_index_from_entries()

    def _load_from_disk(self):
        if not self.save_path or not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, "rb") as file:
                data = pickle.load(file)
        except Exception:
            return
        if isinstance(data, PersonaSystem):
            self.__setstate__(data.__getstate__())
        elif isinstance(data, dict):
            self.__setstate__(data)

    def _save_to_disk(self):
        if not self.save_path:
            return
        with open(self.save_path, "wb") as file:
            pickle.dump(self, file)

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
        canonical = re.sub(r"\s+", " ", str(text or "")).strip()
        return canonical

    def _source_fingerprint(self, text):
        canonical = self._canonicalize_source_text(text)
        if not canonical:
            return ""
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

    def _dedupe_storage(self):
        unique_entries = []
        seen_entry_keys = set()
        for entry in self.entries:
            text = self._canonicalize_source_text(entry.get("text", ""))
            kind = entry.get("kind", "chunk")
            source_label = entry.get("source_label", "")
            if not text:
                continue
            key = (kind, text)
            if key in seen_entry_keys:
                continue
            seen_entry_keys.add(key)
            entry["text"] = text
            entry["source_label"] = source_label
            entry.setdefault("kind", kind)
            entry.setdefault("priority", self._entry_priority(text, kind))
            unique_entries.append(entry)
        self.entries = unique_entries

        unique_summaries = []
        seen_summary_texts = set()
        for item in self.core_summaries:
            text = self._canonicalize_source_text(item.get("text", ""))
            if not text or text in seen_summary_texts:
                continue
            seen_summary_texts.add(text)
            item["text"] = text
            unique_summaries.append(item)
        self.core_summaries = unique_summaries

        clean_records = {}
        for fingerprint, payload in self.source_records.items():
            if not fingerprint or fingerprint in clean_records:
                continue
            clean_records[fingerprint] = payload
        self.source_records = clean_records

    def clear(self):
        self.profile = {section: [] for section in SECTION_ORDER}
        self.core_summaries = []
        self.entries = []
        self.source_records = {}
        self.dynamic_keywords = {section: [] for section in SECTION_ORDER}
        self.dynamic_meta_exclusions = []
        self.dynamic_novel_hints = []
        self.dynamic_voice_hints = []
        self.dynamic_trait_markers = []
        self.display_keywords = []
        self.index = None
        self.index_dim = None
        self._save_to_disk()

    def _all_section_keywords(self, section):
        return _dedupe(SECTION_KEYWORDS.get(section, []) + self.dynamic_keywords.get(section, []))

    def _all_meta_exclusions(self):
        return _dedupe(META_EXCLUSION_WORDS + self.dynamic_meta_exclusions)

    def _normalize_fact(self, text, max_chars=80):
        text = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n-:：;；，,。")
        return text[:max_chars].rstrip() if len(text) > max_chars else text

    def _split_sentences(self, text):
        return [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", text or "") if part.strip()]

    def _is_meta_commentary(self, text):
        text = (text or "").strip()
        if not text:
            return True
        if any(word in text for word in self._all_meta_exclusions()):
            return True
        if any(word in text for word in AUDIENCE_PERSPECTIVE_WORDS) and any(word in text for word in META_REACTION_WORDS):
            return True
        if ("控" in text or "党" in text or "厨" in text or "推" in text) and any(word in text for word in AUDIENCE_PERSPECTIVE_WORDS):
            return True
        return False

    def _is_display_keyword_candidate(self, token):
        token = self._normalize_fact(token, max_chars=12)
        if len(token) < 2 or len(token) > 10:
            return False
        if token in {
            "适合", "不合", "离去", "分类", "国家", "感觉", "感受到", "看到", "发现",
            "进行", "故事", "经历", "设定", "资料", "文本", "描述", "观众", "粉丝", "受众", "读者", "作者", "评价",
        }:
            return False
        if any(mark in token for mark in ("的", "了", "是", "在", "和", "与", "及", "被", "把", "让", "就", "很", "又", "并")):
            return False
        if re.search(r"[0-9A-Za-z]{4,}", token):
            return False
        if re.search(r"[，。！？；：“”\"'（）()]", token):
            return False
        return True

    def _fact_to_keywords(self, value):
        text = self._normalize_fact(value, max_chars=24)
        pieces = [text]
        pieces.extend(part.strip(" :：;；，,、。！？“”\"'（）()") for part in TRAIT_SPLIT_RE.split(text))
        result = []
        for piece in pieces:
            for token in DYNAMIC_KEYWORD_RE.findall(piece):
                if token in {"这个", "那个", "一种", "一些", "自己", "别人", "因为", "所以", "如果", "但是", "然后", "感觉", "有点", "其实", "真的", "非常", "比较", "不是", "可以"}:
                    continue
                result.append(token)
        return _dedupe(result, limit=8)

    def _add_fact(self, profile, section, fact):
        fact = self._normalize_fact(fact, max_chars=80)
        if not fact or self._is_meta_commentary(fact):
            return
        bucket = profile.setdefault(section, [])
        if fact not in bucket:
            bucket.append(fact)

    def _extract_profile_facts(self, raw_text):
        profile = {section: [] for section in SECTION_ORDER}
        for line in re.split(r"\n+", raw_text or ""):
            stripped = line.strip()
            if len(stripped) < 2 or self._is_meta_commentary(stripped):
                continue
            if any(marker in stripped for marker in TRAIT_MARKERS) or "“" in stripped or "”" in stripped:
                parts = [part.strip(" :：;；，,、") for part in TRAIT_SPLIT_RE.split(stripped) if part.strip(" :：;；，,、")]
                for part in parts:
                    if self._is_meta_commentary(part):
                        continue
                    matched = False
                    for section in SECTION_ORDER:
                        if any(keyword in part for keyword in self._all_section_keywords(section)):
                            self._add_fact(profile, section, part)
                            matched = True
                    if not matched and self._is_display_keyword_candidate(part):
                        self._add_fact(profile, "personality", part)
        for sentence in self._split_sentences(raw_text):
            if self._is_meta_commentary(sentence):
                continue
            for section in SECTION_ORDER:
                if any(keyword in sentence for keyword in self._all_section_keywords(section)):
                    self._add_fact(profile, section, sentence)
            for quote in re.findall(r"[“\"]([^”\"]{3,60})[”\"]", sentence):
                self._add_fact(profile, "voice_style", quote)
                if len(quote) <= 24:
                    self._add_fact(profile, "catchphrases", quote)
                if any(marker in quote for marker in ("我", "本小姐", "在下", "咱", "你", "阁下", "先生", "小姐", "大人", "前辈")):
                    self._add_fact(profile, "addressing_habits", quote)
                if quote.endswith(("呢", "哦", "啊", "呀", "吧", "喔", "啦", "嘛")):
                    self._add_fact(profile, "sentence_endings", quote[-2:] if len(quote) >= 2 else quote)
        return profile

    def _merge_profile(self, profile):
        for section in SECTION_ORDER:
            self.profile[section] = _dedupe(self.profile.get(section, []) + profile.get(section, []), limit=18)

    def _merge_dynamic_list(self, existing, incoming, limit):
        clean = [self._normalize_fact(value, max_chars=24) for value in incoming or []]
        clean = [value for value in clean if value and not self._is_meta_commentary(value)]
        return _dedupe((existing or []) + clean, limit=limit)

    def _normalize_summary(self, summary):
        normalized = {}
        for section in SECTION_ORDER:
            values = [self._normalize_fact(value) for value in summary.get(section, [])]
            normalized[section] = _dedupe([value for value in values if value and not self._is_meta_commentary(value)], limit=10)
        normalized["style_examples"] = _dedupe([self._normalize_fact(value, max_chars=72) for value in summary.get("style_examples", []) if value], limit=6)
        normalized["natural_reference_triggers"] = _dedupe([self._normalize_fact(value, max_chars=24) for value in summary.get("natural_reference_triggers", []) if value], limit=8)
        normalized["section_keywords"] = {
            section: _dedupe(
                [
                    keyword
                    for keyword in (self._normalize_fact(value, max_chars=12) for value in (summary.get("section_keywords", {}) or {}).get(section, []))
                    if keyword and not self._is_meta_commentary(keyword) and self._is_display_keyword_candidate(keyword)
                ],
                limit=8,
            )
            for section in SECTION_ORDER
        }
        normalized["meta_exclusion_words"] = self._merge_dynamic_list([], summary.get("meta_exclusion_words", []), limit=12)
        normalized["novel_persona_hints"] = self._merge_dynamic_list([], summary.get("novel_persona_hints", []), limit=10)
        normalized["voice_hints"] = self._merge_dynamic_list([], summary.get("voice_hints", []), limit=10)
        normalized["trait_markers"] = self._merge_dynamic_list([], summary.get("trait_markers", []), limit=10)
        normalized["display_keywords"] = _dedupe(
            [
                keyword
                for keyword in (self._normalize_fact(value, max_chars=12) for value in summary.get("display_keywords", []))
                if keyword and not self._is_meta_commentary(keyword) and self._is_display_keyword_candidate(keyword)
            ],
            limit=6,
        )
        return normalized

    def _summary_to_text(self, summary):
        lines = [f"{self.persona_name} core persona canon:"]
        for section in SECTION_ORDER:
            values = summary.get(section, [])
            if values:
                lines.append(f"[{SECTION_TITLES[section]}]")
                lines.extend(f"- {value}" for value in values)
        if summary.get("style_examples"):
            lines.append("[Style Examples]")
            lines.extend(f"- {value}" for value in summary["style_examples"])
        if summary.get("natural_reference_triggers"):
            lines.append("[Natural Reference Triggers]")
            lines.extend(f"- {value}" for value in summary["natural_reference_triggers"])
        if summary.get("display_keywords"):
            lines.append("[Display Keywords]")
            lines.extend(f"- {value}" for value in summary["display_keywords"])
        return "\n".join(lines).strip()

    def _looks_like_persona_chunk(self, text):
        if self._is_meta_commentary(text):
            return -5
        score = 0
        for section in SECTION_ORDER:
            score += sum(2 for keyword in self._all_section_keywords(section) if keyword in text)
        score += sum(1 for hint in NOVEL_PERSONA_HINTS + self.dynamic_novel_hints if hint in text)
        score += sum(2 for hint in VOICE_HINTS + self.dynamic_voice_hints if hint in text)
        if "“" in text or "”" in text or "\"" in text:
            score += 2
        if len(text) < 16:
            score -= 1
        return score

    def _chunk_text(self, raw_text, target_chars=420, hard_limit=650):
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", raw_text or "") if paragraph.strip()]
        if not paragraphs:
            paragraphs = [raw_text.strip()] if raw_text and raw_text.strip() else []
        chunks = []
        current = ""
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
        return _dedupe([chunk for chunk in chunks if chunk and len(chunk) >= 6])

    def _prepare_summary_source(self, raw_text, max_chars=16000):
        chunks = sorted(self._chunk_text(raw_text), key=self._looks_like_persona_chunk, reverse=True)
        selected = []
        total = 0
        for chunk in chunks:
            projected = total + len(chunk) + 2
            if selected and projected > max_chars:
                continue
            selected.append(chunk)
            total = projected
            if total >= max_chars:
                break
        return ("\n\n".join(selected) if selected else raw_text[:max_chars])[:max_chars]

    def _build_reference_text(self):
        skill_result = DEFAULT_TOOL_REGISTRY.run("web_search", persona_name=self.persona_name, query="", max_results=2, timeout=8)
        snippets = skill_result.get("snippets", [])
        lines = []
        for snippet in snippets:
            lines.append(f"[{snippet['source']} | {snippet['title']}]")
            lines.append(snippet["text"])
        return "\n\n".join(lines)[:2400]

    def build_story_context(self, query, top_k=4, char_budget=900):
        if not query:
            return ""
        story_candidates = []
        seen = set()
        search_pool = self._keyword_search_entries(query, top_k=top_k * 3) + self._search_entries(query, top_k=top_k * 3)
        for entry in search_pool:
            text = entry.get("text", "")
            if self._is_meta_commentary(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            kind = entry.get("kind", "")
            source_label = entry.get("source_label", "source")
            score = 0
            if kind == "core_summary":
                score += 2
            if any(keyword in text for keyword in ("经历", "故事", "过去", "旅行", "曾经", "背景", "设定")):
                score += 2
            if any(value in text for value in self.profile.get("life_experiences", [])[:6]):
                score += 1
            story_candidates.append((score, f"[Local | {source_label}] {text}"))
        story_candidates.sort(key=lambda item: item[0], reverse=True)
        local_lines = [line for _, line in story_candidates[:top_k]]
        context = "\n".join(local_lines).strip()
        if len(context) > char_budget:
            context = context[: char_budget - 3].rstrip() + "..."
        return context

    def _summarize_with_llm(self, raw_text, source_label):
        prompt = build_persona_summary_prompt(
            persona_name=self.persona_name,
            source_label=source_label,
            source_text=self._prepare_summary_source(raw_text),
            reference_text=self._build_reference_text() or "None",
        )
        try:
            summary = self.model.generate(prompt, return_json=True, schema=PERSONA_SUMMARY_SCHEMA, temperature=0.1, max_tokens=2200)
        except Exception:
            return None
        if not isinstance(summary, dict):
            return None
        normalized = self._normalize_summary(summary)
        return PersonaSummaryModel(**normalized)

    def _merge_summary_data(self, summary):
        summary = self._summary_to_dict(summary)
        for section in SECTION_ORDER:
            self.profile[section] = _dedupe(self.profile.get(section, []) + summary.get(section, []), limit=18)
            self.dynamic_keywords[section] = _dedupe(
                self.dynamic_keywords.get(section, []) + summary.get("section_keywords", {}).get(section, []),
                limit=18,
            )
        self.dynamic_meta_exclusions = self._merge_dynamic_list(self.dynamic_meta_exclusions, summary.get("meta_exclusion_words", []), limit=24)
        self.dynamic_novel_hints = self._merge_dynamic_list(self.dynamic_novel_hints, summary.get("novel_persona_hints", []), limit=18)
        self.dynamic_voice_hints = self._merge_dynamic_list(self.dynamic_voice_hints, summary.get("voice_hints", []), limit=18)
        self.dynamic_trait_markers = self._merge_dynamic_list(self.dynamic_trait_markers, summary.get("trait_markers", []), limit=18)
        self.display_keywords = _dedupe(self.display_keywords + summary.get("display_keywords", []), limit=8)

    def _register_dynamic_keywords(self, extracted_profile, summary):
        summary = self._summary_to_dict(summary)
        for section in SECTION_ORDER:
            values = list(extracted_profile.get(section, []))
            if summary:
                values.extend(summary.get(section, []))
                values.extend(summary.get("section_keywords", {}).get(section, []))
            tokens = []
            for value in values:
                tokens.extend(self._fact_to_keywords(value))
            self.dynamic_keywords[section] = _dedupe(self.dynamic_keywords.get(section, []) + tokens, limit=18)

    def _append_core_summary(self, summary, source_label):
        summary = self._summary_to_dict(summary)
        text = self._summary_to_text(summary)
        self.core_summaries = [item for item in self.core_summaries if item.get("source_label") != source_label]
        self.core_summaries.append({"source_label": source_label, "summary": summary, "text": text})

    def _entry_priority(self, text, kind):
        return 2.0 if kind == "core_summary" else 1.0 + max(0, self._looks_like_persona_chunk(text)) * 0.1

    def _ensure_index(self, dim):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
            self.index_dim = dim
        elif self.index_dim != dim:
            self._rebuild_index_from_entries()

    def _rebuild_index_from_entries(self):
        vectors = [entry.get("embedding") for entry in self.entries if entry.get("embedding") is not None]
        if not vectors:
            self.index = None
            self.index_dim = None
            return
        normalized = [_normalize_vector(vector) for vector in vectors]
        self.index = faiss.IndexFlatIP(len(normalized[0]))
        self.index.add(np.vstack(normalized).astype(np.float32))
        self.index_dim = len(normalized[0])
        for entry, vector in zip(self.entries, normalized):
            entry["embedding"] = vector

    def _append_entries_without_embeddings(self, pending_entries, source_fingerprint):
        for entry in pending_entries:
            self.entries.append(
                {
                    "text": self._canonicalize_source_text(entry["text"]),
                    "source_label": entry["source_label"],
                    "kind": entry["kind"],
                    "embedding": None,
                    "priority": self._entry_priority(entry["text"], entry["kind"]),
                    "source_fingerprint": source_fingerprint,
                }
            )

    def _commit_summary_and_entries(self, raw_text, source_label, summary, progress_callback=None):
        source_fingerprint = self._source_fingerprint(raw_text)
        if source_fingerprint and source_fingerprint in self.source_records:
            existing_label = self.source_records[source_fingerprint].get("source_label", source_label)
            self._emit_progress(progress_callback, 100, 100, "done", f"检测到重复资料，已跳过：{existing_label}")
            self._save_to_disk()
            return 0

        extracted_profile = self._extract_profile_facts(raw_text)
        self._merge_profile(extracted_profile)
        if summary:
            self._merge_summary_data(summary)
            self._append_core_summary(summary, source_label)
        self._register_dynamic_keywords(extracted_profile, summary)

        existing_texts = {self._canonicalize_source_text(entry["text"]) for entry in self.entries}
        chunks = []
        for chunk in self._chunk_text(raw_text):
            canonical_chunk = self._canonicalize_source_text(chunk)
            if self._looks_like_persona_chunk(canonical_chunk) < 0 or canonical_chunk in existing_texts:
                continue
            chunks.append(canonical_chunk)
        pending_entries = [{"text": chunk, "source_label": source_label, "kind": "chunk"} for chunk in chunks]
        if summary:
            summary_text = self._canonicalize_source_text(self._summary_to_text(self._summary_to_dict(summary)))
            if summary_text not in existing_texts:
                pending_entries.append({"text": summary_text, "source_label": f"{source_label}#core_summary", "kind": "core_summary"})

        if not pending_entries:
            if source_fingerprint:
                self.source_records[source_fingerprint] = {
                    "source_label": source_label,
                    "updated_at": datetime.now().isoformat(),
                }
            self._save_to_disk()
            self._emit_progress(progress_callback, 100, 100, "done", "没有新增内容，已完成")
            return 0

        total_entries = len(pending_entries)

        def embedding_progress(current, total, stage, detail):
            safe_total = total or total_entries or 1
            ratio = current / safe_total
            overall = 35 + int(ratio * 50)
            self._emit_progress(progress_callback, min(overall, 88), 100, stage, f"{detail}（{current}/{safe_total}）")

        self._emit_progress(progress_callback, 35, 100, "embedding", f"正在为 {total_entries} 条人设内容生成向量")
        try:
            embeddings = mistral_embed_texts([entry["text"] for entry in pending_entries], progress_callback=embedding_progress)
        except MistralEmbeddingCapacityError:
            self._emit_progress(progress_callback, 90, 100, "saving", "嵌入服务繁忙，已降级为文本入库，稍后可继续补全向量")
            self._append_entries_without_embeddings(pending_entries, source_fingerprint)
            if source_fingerprint:
                self.source_records[source_fingerprint] = {
                    "source_label": source_label,
                    "updated_at": datetime.now().isoformat(),
                }
            self._dedupe_storage()
            self._save_to_disk()
            self._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 条（当前为文本检索模式）")
            print(f"Loaded {len(pending_entries)} persona entries from: {source_label} (without embeddings)")
            return len(pending_entries)
        normalized = [_normalize_vector(vector) for vector in embeddings]
        self._ensure_index(len(normalized[0]))
        self.index.add(np.vstack(normalized).astype(np.float32))

        self._emit_progress(progress_callback, 92, 100, "saving", "正在写入人设库")
        for entry, embedding in zip(pending_entries, normalized):
            self.entries.append(
                {
                    "text": self._canonicalize_source_text(entry["text"]),
                    "source_label": entry["source_label"],
                    "kind": entry["kind"],
                    "embedding": embedding,
                    "priority": self._entry_priority(entry["text"], entry["kind"]),
                    "source_fingerprint": source_fingerprint,
                }
            )
        if source_fingerprint:
            self.source_records[source_fingerprint] = {
                "source_label": source_label,
                "updated_at": datetime.now().isoformat(),
            }
        self._dedupe_storage()
        self._save_to_disk()
        self._emit_progress(progress_callback, 100, 100, "done", f"学习完成，共新增 {len(pending_entries)} 条")
        print(f"Loaded {len(pending_entries)} persona entries from: {source_label}")
        return len(pending_entries)

    def load_text(self, raw_text, source_label="manual_input", progress_callback=None):
        self._emit_progress(progress_callback, 0, 100, "parsing", f"正在解析 {source_label}")
        self._emit_progress(progress_callback, 12, 100, "summary", "正在提取核心人设")
        summary = self._summarize_with_llm(raw_text, source_label)
        self._emit_progress(progress_callback, 28, 100, "chunking", "正在切分学习文本")
        return self._commit_summary_and_entries(
            raw_text,
            source_label,
            summary,
            progress_callback=progress_callback,
        )

    def load_file(self, filepath, progress_callback=None):
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        suffix = os.path.splitext(filepath)[1].lower()
        if suffix and suffix not in (".txt", ".md", ".log", ".json", ".csv"):
            raise ValueError("Only text-like files are supported: .txt, .md, .log, .json, .csv")
        self._emit_progress(progress_callback, 0, 100, "reading", f"正在读取 {os.path.basename(filepath)}")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()
        return self.load_text(text, source_label=os.path.basename(filepath), progress_callback=progress_callback)

    def get_status(self):
        return PersonaStatusModel(
            persona_name=self.persona_name,
            chunk_count=self.chunk_count,
            display_keywords=self.get_display_keywords(),
        )

    def _collect_web_snippets(self, persona_name, work_title="", max_results=5):
        query_candidates = [
            " ".join(part for part in [work_title, "角色设定", "性格", "口癖", "价值观", "世界观", "故事", "经历"] if part),
            " ".join(part for part in [work_title, "角色设定", "故事"] if part),
            work_title,
            "",
        ]
        snippets = []
        seen_keys = set()
        for query in query_candidates:
            try:
                search_result = DEFAULT_TOOL_REGISTRY.run(
                    "web_search",
                    persona_name=persona_name,
                    query=query,
                    max_results=max_results,
                    timeout=10,
                )
            except Exception:
                continue
            for item in search_result.get("snippets", []):
                text = (item.get("text", "") or "").strip()
                if not text:
                    continue
                key = (item.get("source", "web"), item.get("title", ""), text[:120])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                snippets.append(
                    PersonaSourceSnippet(
                        source=item.get("source", "web"),
                        title=item.get("title", ""),
                        text=text,
                    )
                )
            if len(snippets) >= max_results:
                break
        return snippets[:max_results]

    def preview_from_sources(self, persona_name="", work_title="", local_text="", local_label="manual_input", max_results=5):
        persona_name = (persona_name or self.persona_name or "").strip()
        work_title = (work_title or "").strip()
        local_text = (local_text or "").strip()
        local_label = (local_label or "manual_input").strip() or "manual_input"
        if not persona_name and not local_text:
            raise ValueError("角色名或本地资料至少需要提供一个。")

        snippets = []
        if local_text:
            snippets.append(
                PersonaSourceSnippet(
                    source="local",
                    title=local_label,
                    text=local_text,
                )
            )

        if persona_name:
            snippets.extend(self._collect_web_snippets(persona_name, work_title=work_title, max_results=max_results))

        if not snippets:
            raise ValueError("没有找到可用于预览的人设资料，请补充作品名或上传本地资料。")

        mode = "hybrid" if local_text and len(snippets) > 1 else ("local_only" if local_text else "cold_start")
        source_label = mode
        if persona_name:
            source_label += f":{persona_name}"
        if work_title:
            source_label += f":{work_title}"
        source_text = "\n\n".join(
            f"[{snippet.source} | {snippet.title}]\n{snippet.text}" for snippet in snippets
        )
        summary = self._summarize_with_llm(source_text, source_label) or PersonaSummaryModel()
        preview = PersonaPreviewModel(
            preview_id=str(uuid.uuid4()),
            persona_name=persona_name or self.persona_name,
            work_title=work_title,
            source_label=source_label,
            source_text=source_text,
            snippets=snippets,
            summary=summary,
            created_at=datetime.now().isoformat(timespec="seconds"),
            mode=mode,
            committed=False,
        )
        self.pending_previews[preview.preview_id] = preview.dict()
        self._save_to_disk()
        return preview

    def preview_cold_start(self, persona_name, work_title="", max_results=5):
        return self.preview_from_sources(
            persona_name=persona_name,
            work_title=work_title,
            local_text="",
            local_label="manual_input",
            max_results=max_results,
        )

    def confirm_preview(self, preview_id, progress_callback=None):
        payload = self.pending_previews.get(preview_id)
        if not payload:
            raise KeyError(f"Unknown preview_id: {preview_id}")
        preview = PersonaPreviewModel(**payload)
        count = self._commit_summary_and_entries(
            preview.source_text,
            preview.source_label,
            preview.summary,
            progress_callback=progress_callback,
        )
        self.persona_name = preview.persona_name or self.persona_name
        committed_preview = preview.copy(update={"committed": True})
        self.pending_previews[preview.preview_id] = committed_preview.dict()
        self._save_to_disk()
        return {"count": count, "preview": committed_preview}

    def _search_entries(self, query, top_k=6):
        if not query or self.index is None or not self.entries:
            return []
        try:
            query_vector = _normalize_vector(mistral_embed_texts(query))
        except MistralEmbeddingCapacityError:
            return []
        scores, indices = self.index.search(np.asarray([query_vector], dtype=np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.entries) or score < PERSONA_RETRIEVAL_THRESHOLD:
                continue
            entry = self.entries[idx]
            results.append((float(score) * float(entry.get("priority", 1.0)), entry))
        results.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in results]

    def _extract_query_terms(self, query):
        text = self._normalize_fact(query, max_chars=120)
        terms = []
        for token in DYNAMIC_KEYWORD_RE.findall(text):
            if len(token) < 2:
                continue
            if token in {"角色", "设定", "故事", "经历", "背景", "一下", "什么", "怎么", "这个", "那个", "有关", "关于"}:
                continue
            terms.append(token)
        return _dedupe(terms, limit=10)

    def _keyword_search_entries(self, query, top_k=6):
        if not query or not self.entries:
            return []
        terms = self._extract_query_terms(query)
        if not terms:
            return []
        scored = []
        for entry in self.entries:
            text = entry.get("text", "")
            if not text or self._is_meta_commentary(text):
                continue
            hits = sum(1 for term in terms if term in text)
            if not hits:
                continue
            score = float(hits)
            kind = entry.get("kind", "")
            if kind == "core_summary":
                score += 1.5
            if any(keyword in text for keyword in ("经历", "故事", "过去", "旅行", "曾经", "背景", "设定")):
                score += 0.5
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def get_display_keywords(self, limit=DISPLAY_KEYWORD_LIMIT):
        candidates = []
        candidates.extend(self.display_keywords)
        for summary_item in reversed(self.core_summaries[-3:]):
            candidates.extend(summary_item.get("summary", {}).get("display_keywords", []))
        for section in ("personality", "voice_style", "values", "worldview"):
            candidates.extend(self.dynamic_keywords.get(section, []))
            candidates.extend(self.profile.get(section, []))
        filtered = []
        for candidate in candidates:
            candidate = self._normalize_fact(candidate, max_chars=12)
            if candidate and self._is_display_keyword_candidate(candidate) and not self._is_meta_commentary(candidate):
                filtered.append(candidate)
        return _dedupe(filtered, limit=limit)

    def build_context(self, query):
        sections = []
        canon_lines = []
        for section in SECTION_ORDER:
            values = self.profile.get(section, [])
            if values:
                canon_lines.append(f"[{SECTION_TITLES[section]}]")
                canon_lines.extend(f"- {value}" for value in values[:6])
        top_keywords = self.get_display_keywords()
        if top_keywords:
            canon_lines.append("[Top Persona Keywords]")
            canon_lines.extend(f"- {value}" for value in top_keywords)
        if canon_lines:
            sections.append("Canon Persona Facts:\n" + "\n".join(canon_lines))

        if self.core_summaries:
            summary_lines = []
            for item in self.core_summaries[-2:]:
                summary_lines.append(f"[Source: {item.get('source_label', 'unknown')}]")
                summary_lines.append(item.get("text", ""))
            sections.append("Core Persona Summaries:\n" + "\n\n".join(summary_lines))

        retrieved_entries = []
        seen = set()
        for entry in self._keyword_search_entries(query, top_k=4) + self._search_entries(query, top_k=4):
            text = entry.get("text", "")
            if text in seen:
                continue
            seen.add(text)
            retrieved_entries.append(entry)
        if retrieved_entries:
            retrieval_lines = [f"[{entry.get('source_label', 'source')}] {entry['text']}" for entry in retrieved_entries[:5]]
            sections.append("Retrieved Persona Evidence:\n" + "\n".join(retrieval_lines))

        context = "\n\n".join(section for section in sections if section).strip()
        if len(context) > PERSONA_CONTEXT_CHAR_BUDGET:
            context = context[: PERSONA_CONTEXT_CHAR_BUDGET - 3].rstrip() + "..."
        return context

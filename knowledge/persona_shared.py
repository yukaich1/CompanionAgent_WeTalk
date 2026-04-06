from __future__ import annotations

import re

import numpy as np


DIMENSION_ORDER = [
    "00_BACKGROUND",
    "A_SPEECH_STYLE",
    "B_CATCHPHRASES",
    "C_ADDRESS_AND_PAUSE",
    "D_EMOTION_PATH",
    "E_LIKES",
    "F_DISLIKES_AND_TABOOS",
    "G_AVOID_PATTERNS",
    "H_PERSONALITY",
    "I_VALUES_AND_WORLDVIEW",
    "J_RELATIONSHIP",
    "K_NARRATIVE",
    "L_HUMOR",
]

DIMENSION_TITLES_ZH = {
    "00_BACKGROUND": "基础身份信息",
    "A_SPEECH_STYLE": "说话方式",
    "B_CATCHPHRASES": "口头禅与标志句式",
    "C_ADDRESS_AND_PAUSE": "称呼与停顿习惯",
    "D_EMOTION_PATH": "情绪表达路径",
    "E_LIKES": "喜好与偏好",
    "F_DISLIKES_AND_TABOOS": "厌恶与禁忌",
    "G_AVOID_PATTERNS": "避免出现的表达",
    "H_PERSONALITY": "性格核心",
    "I_VALUES_AND_WORLDVIEW": "价值观与世界观",
    "J_RELATIONSHIP": "关系处理方式",
    "K_NARRATIVE": "叙事与回忆方式",
    "L_HUMOR": "幽默与玩笑方式",
}

COMPACT_TEMPLATE_DIMENSIONS = [
    "A_SPEECH_STYLE",
    "B_CATCHPHRASES",
    "C_ADDRESS_AND_PAUSE",
    "D_EMOTION_PATH",
    "E_LIKES",
    "F_DISLIKES_AND_TABOOS",
    "G_AVOID_PATTERNS",
    "H_PERSONALITY",
    "I_VALUES_AND_WORLDVIEW",
    "J_RELATIONSHIP",
    "K_NARRATIVE",
    "L_HUMOR",
]

KEYWORD_STOPWORDS = {
    "角色",
    "设定",
    "资料",
    "文本",
    "内容",
    "部分",
    "方面",
    "故事",
    "经历",
    "背景",
    "身份",
    "外貌",
    "说话方式",
    "口头禅",
    "称呼",
    "语气",
    "风格",
    "世界观",
    "价值观",
    "性格",
    "喜欢",
    "讨厌",
    "禁忌",
    "人物",
    "这个",
    "那个",
    "自己",
    "别人",
}

META_EXCLUSION_WORDS = [
    "粉丝",
    "观众",
    "读者",
    "作者",
    "制作组",
    "评价",
    "人气",
    "热度",
    "讨论度",
    "创作意图",
]

AUDIENCE_PERSPECTIVE_WORDS = ["观众", "粉丝", "读者", "网友", "路人", "玩家", "大家", "别人"]
META_REACTION_WORDS = ["喜欢", "偏爱", "喜爱", "萌点", "圈粉", "吸引", "人气", "受欢迎", "评价", "印象", "讨论"]

TRAIT_SPLIT_RE = re.compile(r"[、，；。！？?\n/]+")
DYNAMIC_KEYWORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,16}")
DISPLAY_KEYWORD_LIMIT = 32

_RULE_DIMENSIONS = {
    dim: {
        "type": "object",
        "properties": {
            "rules": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string"},
        },
        "required": ["rules", "confidence"],
        "additionalProperties": False,
    }
    for dim in DIMENSION_ORDER
    if dim not in {"00_BACKGROUND", "B_CATCHPHRASES", "E_LIKES", "F_DISLIKES_AND_TABOOS", "G_AVOID_PATTERNS"}
}

_RULE_DIMENSIONS["00_BACKGROUND"] = {
    "type": "object",
    "properties": {
        "profile": {"type": "object"},
        "key_experiences": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string"},
    },
    "required": ["profile", "key_experiences", "confidence"],
    "additionalProperties": False,
}

_RULE_DIMENSIONS["B_CATCHPHRASES"] = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "usage": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": ["pattern", "usage", "tone"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string"},
    },
    "required": ["patterns", "confidence"],
    "additionalProperties": False,
}

_RULE_DIMENSIONS["E_LIKES"] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "behavior": {"type": "string"},
                },
                "required": ["item", "behavior"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string"},
    },
    "required": ["items", "confidence"],
    "additionalProperties": False,
}

_RULE_DIMENSIONS["F_DISLIKES_AND_TABOOS"] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "level": {"type": "string"},
                    "behavior": {"type": "string"},
                },
                "required": ["item", "level", "behavior"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string"},
    },
    "required": ["items", "confidence"],
    "additionalProperties": False,
}

_RULE_DIMENSIONS["G_AVOID_PATTERNS"] = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "reason": {"type": "string"},
                    "alternative": {"type": "string"},
                },
                "required": ["pattern", "reason", "alternative"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string"},
    },
    "required": ["patterns", "confidence"],
    "additionalProperties": False,
}

PERSONA_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "character_name": {"type": "string"},
        "source_label": {"type": "string"},
        "base_template": {
            "type": "object",
            "properties": _RULE_DIMENSIONS,
            "required": DIMENSION_ORDER,
            "additionalProperties": False,
        },
        "character_voice_card": {"type": "string"},
        "display_keywords": {"type": "array", "items": {"type": "string"}},
        "natural_reference_triggers": {"type": "array", "items": {"type": "string"}},
        "story_chunks": {"type": "array"},
        "style_examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "scene": {"type": "string"},
                    "emotion": {"type": "string"},
                    "rules_applied": {"type": "array", "items": {"type": "string"}},
                    "source": {"type": "string"},
                    "affinity_level": {"type": "string"},
                },
                "required": ["text", "scene", "emotion", "rules_applied", "source", "affinity_level"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "character_name",
        "source_label",
        "base_template",
        "character_voice_card",
        "display_keywords",
        "style_examples",
    ],
    "additionalProperties": False,
}


def normalize_vector(vector) -> list[float]:
    array = np.asarray(vector or [], dtype=np.float32)
    if array.size == 0:
        return []
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array.tolist()
    return (array / norm).tolist()


def dedupe(values, limit: int | None = None):
    seen = set()
    result = []
    for value in values or []:
        key = value if isinstance(value, str) else repr(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if limit is not None and len(result) >= limit:
            break
    return result

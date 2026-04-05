from __future__ import annotations

import re

import numpy as np


DIMENSION_ORDER = [
    "00_BACKGROUND_PROFILE",
    "01_PERSONALITY_CORE",
    "02_SPEECH_SURFACE",
    "03_TONE_LAYER",
    "04_VOCABULARY_PREFERENCE",
    "05_CATCHPHRASES_AND_PATTERNS",
    "06_ADDRESSING_SYSTEM",
    "07_PUNCTUATION_AND_PAUSE",
    "08_EMOTION_EXPRESSION_PATH",
    "09_HUMOR_MECHANISM",
    "10_INFORMATION_DELIVERY",
    "11_TOPIC_CONTROL",
    "12_WORLDVIEW_ASSUMPTION",
    "13_VALUE_EXPRESSION",
    "14_SELF_PERCEPTION_STABILITY",
    "15_RELATIONSHIP_DYNAMICS",
    "16_NARRATIVE_STYLE",
    "17_LIKES_AND_PREFERENCES",
    "18_DISLIKES_AND_TABOOS",
    "19_AVOID_PATTERNS",
]

DIMENSION_TITLES_ZH = {
    "00_BACKGROUND_PROFILE": "背景档案",
    "01_PERSONALITY_CORE": "性格核心特质",
    "02_SPEECH_SURFACE": "说话表层结构",
    "03_TONE_LAYER": "语气层次",
    "04_VOCABULARY_PREFERENCE": "词汇偏好",
    "05_CATCHPHRASES_AND_PATTERNS": "口头禅与标志性句式",
    "06_ADDRESSING_SYSTEM": "称呼体系",
    "07_PUNCTUATION_AND_PAUSE": "标点与停顿习惯",
    "08_EMOTION_EXPRESSION_PATH": "情绪表达路径",
    "09_HUMOR_MECHANISM": "幽默机制",
    "10_INFORMATION_DELIVERY": "信息传递方式",
    "11_TOPIC_CONTROL": "话题控制方式",
    "12_WORLDVIEW_ASSUMPTION": "世界观隐含假设",
    "13_VALUE_EXPRESSION": "价值观的语言投影",
    "14_SELF_PERCEPTION_STABILITY": "自我认知稳定性",
    "15_RELATIONSHIP_DYNAMICS": "关系动态中的语言变化",
    "16_NARRATIVE_STYLE": "叙事风格",
    "17_LIKES_AND_PREFERENCES": "喜好与偏好",
    "18_DISLIKES_AND_TABOOS": "厌恶与禁忌",
    "19_AVOID_PATTERNS": "反例模式",
}

HIGH_PRIORITY_DIMENSIONS = {
    "02_SPEECH_SURFACE",
    "03_TONE_LAYER",
    "04_VOCABULARY_PREFERENCE",
    "05_CATCHPHRASES_AND_PATTERNS",
    "06_ADDRESSING_SYSTEM",
    "07_PUNCTUATION_AND_PAUSE",
    "08_EMOTION_EXPRESSION_PATH",
    "09_HUMOR_MECHANISM",
    "13_VALUE_EXPRESSION",
    "17_LIKES_AND_PREFERENCES",
    "18_DISLIKES_AND_TABOOS",
    "19_AVOID_PATTERNS",
}

COMPACT_TEMPLATE_DIMENSIONS = [
    "01_PERSONALITY_CORE",
    "02_SPEECH_SURFACE",
    "03_TONE_LAYER",
    "05_CATCHPHRASES_AND_PATTERNS",
    "06_ADDRESSING_SYSTEM",
    "08_EMOTION_EXPRESSION_PATH",
    "09_HUMOR_MECHANISM",
    "13_VALUE_EXPRESSION",
    "15_RELATIONSHIP_DYNAMICS",
    "17_LIKES_AND_PREFERENCES",
    "18_DISLIKES_AND_TABOOS",
    "19_AVOID_PATTERNS",
]

DIMENSION_QUERY_KEYWORDS = {
    "00_BACKGROUND_PROFILE": ["背景", "出身", "身份", "年龄", "外貌", "档案", "成长"],
    "01_PERSONALITY_CORE": ["性格", "个性", "特质", "脾气", "气质"],
    "02_SPEECH_SURFACE": ["说话", "表达", "句式", "措辞", "聊天方式"],
    "03_TONE_LAYER": ["语气", "口吻", "礼貌", "毒舌", "腹黑"],
    "04_VOCABULARY_PREFERENCE": ["用词", "词汇", "书面", "口语", "敬语"],
    "05_CATCHPHRASES_AND_PATTERNS": ["口头禅", "常说", "惯用句", "经典台词"],
    "06_ADDRESSING_SYSTEM": ["称呼", "自称", "叫法", "怎么称呼"],
    "07_PUNCTUATION_AND_PAUSE": ["停顿", "省略号", "句尾", "标点", "沉默"],
    "08_EMOTION_EXPRESSION_PATH": ["情绪", "心情", "感受", "高兴", "难过", "生气", "关心"],
    "09_HUMOR_MECHANISM": ["幽默", "玩笑", "调侃", "嘲讽", "挖苦"],
    "10_INFORMATION_DELIVERY": ["解释", "说明", "怎么回答", "表达看法"],
    "11_TOPIC_CONTROL": ["转移话题", "回避", "不想谈", "结束话题"],
    "12_WORLDVIEW_ASSUMPTION": ["世界观", "看法", "人生观", "怎么看世界"],
    "13_VALUE_EXPRESSION": ["价值观", "原则", "底线", "在乎什么", "信念"],
    "14_SELF_PERCEPTION_STABILITY": ["自我认知", "自信", "怎么看自己", "被夸", "被批评"],
    "15_RELATIONSHIP_DYNAMICS": ["关系", "相处", "亲密", "边界", "距离感"],
    "16_NARRATIVE_STYLE": ["叙事", "讲故事", "描述方式", "经历", "过去"],
    "17_LIKES_AND_PREFERENCES": ["喜欢", "偏好", "爱好", "偏爱", "最喜欢"],
    "18_DISLIKES_AND_TABOOS": ["讨厌", "厌恶", "禁忌", "敏感", "底线", "不喜欢"],
    "19_AVOID_PATTERNS": ["不会说", "违和", "出戏", "不像她", "绝不会这么说"],
}

STORY_QUERY_KEYWORDS = [
    "故事",
    "经历",
    "过去",
    "以前",
    "旅行见闻",
    "往事",
    "旅途",
    "曾经",
    "讲讲",
    "回忆",
]

KEYWORD_STOPWORDS = {
    "角色",
    "设定",
    "资料",
    "文本",
    "描述",
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
    "句尾",
    "语气",
    "口吻",
    "表达",
    "风格",
    "价值观",
    "世界观",
    "喜欢",
    "讨厌",
    "偏好",
    "禁忌",
    "性格",
    "品质",
    "关系",
    "相处",
    "人物",
    "这个",
    "那个",
    "自己",
    "别人",
}

PREFERRED_PERSONA_LABELS = [
    "自恋",
    "自信",
    "腹黑",
    "腹黑而礼貌",
    "毒舌",
    "傲娇",
    "冷淡",
    "冷静",
    "温柔",
    "强势",
    "克制",
    "现实主义",
    "理想主义",
    "利己",
    "利他",
    "贪财",
    "聪慧",
    "敏感",
    "疏离",
    "神秘",
    "恶趣味",
    "魔女",
    "旅人",
    "旁观者",
    "高傲",
    "礼貌系",
    "独立",
    "孤独",
    "悲观",
    "乐观",
    "冷酷",
    "直率",
    "成熟",
]

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

NOVEL_PERSONA_HINTS = ["说道", "说着", "语气", "眼神", "神情", "低声", "淡淡地", "沉默"]
VOICE_HINTS = ["她说", "他说", "语气", "口吻", "自称", "称呼", "淡淡地说", "漫不经心地说"]
AUDIENCE_PERSPECTIVE_WORDS = ["观众", "粉丝", "读者", "网友", "路人", "玩家", "大家", "别人"]
META_REACTION_WORDS = ["喜欢", "偏爱", "喜爱", "萌点", "圈粉", "吸引", "人气", "受欢迎", "评价", "印象", "讨论"]

TRAIT_SPLIT_RE = re.compile(r"[、，；。！？!?\n/]+")
DYNAMIC_KEYWORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,16}")
DISPLAY_KEYWORD_LIMIT = 8
KEYWORD_CANDIDATE_LIMIT = 24

_DIMENSION_SCHEMA = {
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
    if dim not in {"00_BACKGROUND_PROFILE", "17_LIKES_AND_PREFERENCES", "18_DISLIKES_AND_TABOOS", "19_AVOID_PATTERNS"}
}

_DIMENSION_SCHEMA["00_BACKGROUND_PROFILE"] = {
    "type": "object",
    "properties": {
        "profile": {"type": "object"},
        "key_experiences": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string"},
    },
    "required": ["profile", "key_experiences", "confidence"],
    "additionalProperties": False,
}

_DIMENSION_SCHEMA["17_LIKES_AND_PREFERENCES"] = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": ["object", "string"]}},
        "confidence": {"type": "string"},
    },
    "required": ["items", "confidence"],
    "additionalProperties": False,
}

_DIMENSION_SCHEMA["18_DISLIKES_AND_TABOOS"] = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": ["object", "string"]}},
        "confidence": {"type": "string"},
    },
    "required": ["items", "confidence"],
    "additionalProperties": False,
}

_DIMENSION_SCHEMA["19_AVOID_PATTERNS"] = {
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
            "properties": _DIMENSION_SCHEMA,
            "required": DIMENSION_ORDER,
            "additionalProperties": False,
        },
        "character_voice_card": {"type": "string"},
        "display_keywords": {"type": "array", "items": {"type": "string"}},
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
        "natural_reference_triggers": {"type": "array", "items": {"type": "string"}},
        "story_chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "story_id": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "emotional_weight": {"type": "string"},
                    "character_impact": {"type": "string"},
                    "trigger_topics": {"type": "array", "items": {"type": "string"}},
                    "source_confidence": {"type": "string"},
                },
                "required": [
                    "story_id",
                    "title",
                    "content",
                    "keywords",
                    "emotional_weight",
                    "character_impact",
                    "trigger_topics",
                    "source_confidence",
                ],
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
        "natural_reference_triggers",
        "story_chunks",
    ],
}


def normalize_vector(vector):
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    return array if norm == 0 else array / norm


def dedupe(items, limit=None):
    seen = set()
    result = []
    for item in items:
        if item is None:
            continue
        key = item if isinstance(item, (str, int, float, tuple)) else repr(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if limit and len(result) >= limit:
            break
    return result

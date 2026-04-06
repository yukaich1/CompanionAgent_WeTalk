from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from llm import FallbackMistralLLM


INTENT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "needs_tool": {"type": "boolean"},
        "tool_name": {"type": ["string", "null"]},
        "tool_params": {"type": ["object", "null"]},
        "extracted_topic": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["intent", "needs_tool", "tool_name", "tool_params", "extracted_topic", "reasoning"],
}


class IntentExtractionResult(BaseModel):
    intent: str = "casual_chat"
    needs_tool: bool = False
    tool_name: str | None = None
    tool_params: dict[str, Any] | None = Field(default_factory=dict)
    extracted_topic: str = ""
    reasoning: str = ""


PROMPT_INTENT_EXTRACTION = """
你是对话路由分析器。请根据用户输入和最近几轮对话，判断当前主要意图。
意图类型：
- `character_related`: 角色设定、自我介绍、经历、性格、口头禅、喜好、禁忌
- `weather_query`: 天气、气温、降雨、风力
- `web_search_query`: 与角色无关的外部知识、现实信息、最新事实
- `casual_chat`: 普通闲聊
- `emotional_interaction`: 安慰、倾诉、情绪回应
- `value_judgment`: 需要角色表达态度或立场
- `help_request`: 明确要求查资料或执行工具任务

最近对话：
{recent_conversation}

角色名：{character_name}
用户输入：{user_input}

只输出 JSON。
""".strip()


class IntentExtractor:
    def __init__(self, model=None):
        self.model = model or FallbackMistralLLM()

    def extract(self, user_input: str, recent_conversation: str = "", character_name: str = "") -> IntentExtractionResult:
        direct = self._rule_first_result(user_input, recent_conversation, character_name)
        if direct is not None:
            return direct
        prompt = PROMPT_INTENT_EXTRACTION.format(
            recent_conversation=recent_conversation or "无",
            character_name=character_name or "当前角色",
            user_input=user_input or "",
        )
        try:
            data = self.model.generate(
                prompt,
                return_json=True,
                schema=INTENT_EXTRACTION_SCHEMA,
                temperature=0.1,
                max_tokens=260,
            )
            return self._normalize_result(data or {}, user_input, recent_conversation, character_name)
        except Exception:
            return self._heuristic_fallback(user_input, recent_conversation, character_name)

    def _rule_first_result(self, user_input: str, recent_conversation: str, character_name: str) -> IntentExtractionResult | None:
        text = str(user_input or "").strip()
        lowered = text.lower()
        if not text:
            return IntentExtractionResult(intent="casual_chat", extracted_topic="")

        if self._looks_like_character_related(text, character_name):
            return IntentExtractionResult(intent="character_related", extracted_topic=self._fallback_topic(text), reasoning="strong persona rule")

        if any(token in text or token in lowered for token in ("天气", "气温", "温度", "下雨", "下雪", "weather", "forecast")):
            location = self._extract_weather_location(text, recent_conversation)
            return IntentExtractionResult(
                intent="weather_query",
                needs_tool=True,
                tool_name="weather",
                tool_params={
                    "location": location,
                    "location_confidence": "high" if location else "low",
                    "location_source": "current_input" if location and location in text else "context",
                },
                extracted_topic=self._fallback_topic(text),
                reasoning="strong weather rule",
            )

        if any(token in text for token in ("新闻", "最新", "最近", "搜索", "查一下", "帮我查", "百科", "资料", "价格", "汇率")):
            return IntentExtractionResult(
                intent="web_search_query",
                needs_tool=True,
                tool_name="web_search",
                tool_params={"search_query": self._fallback_topic(text), "search_type": self._infer_search_type(text), "language": "zh"},
                extracted_topic=self._fallback_topic(text),
                reasoning="strong external rule",
            )
        return None

    def _normalize_result(self, data: dict[str, Any], user_input: str, recent_conversation: str, character_name: str) -> IntentExtractionResult:
        result = IntentExtractionResult(**data)
        if not result.extracted_topic:
            result.extracted_topic = self._fallback_topic(user_input)

        if result.intent not in {"weather_query", "web_search_query"} and self._looks_like_character_related(user_input, character_name):
            result.intent = "character_related"
            result.needs_tool = False
            result.tool_name = None
            result.tool_params = {}
            return result

        if result.intent == "weather_query":
            location = self._normalize_location((result.tool_params or {}).get("location"))
            if not location:
                location = self._extract_weather_location(user_input, recent_conversation)
            result.needs_tool = True
            result.tool_name = "weather"
            result.tool_params = {
                "location": location,
                "location_confidence": "high" if location else "low",
                "location_source": "current_input" if location and location in str(user_input or "") else "context",
            }
        elif result.intent == "web_search_query":
            result.needs_tool = True
            result.tool_name = "web_search"
            params = dict(result.tool_params or {})
            params["search_query"] = str(params.get("search_query") or self._fallback_topic(user_input)).strip()
            params["search_type"] = str(params.get("search_type") or self._infer_search_type(user_input))
            params["language"] = str(params.get("language") or "zh")
            result.tool_params = params
        elif result.intent == "character_related":
            result.needs_tool = False
            result.tool_name = None
            result.tool_params = {}

        if not isinstance(result.tool_params, dict):
            result.tool_params = {}
        return result

    def _heuristic_fallback(self, user_input: str, recent_conversation: str, character_name: str) -> IntentExtractionResult:
        text = str(user_input or "").strip()
        lowered = text.lower()
        weather_tokens = ("天气", "气温", "温度", "下雨", "下雪", "weather", "forecast")
        emotional_tokens = ("难过", "伤心", "委屈", "抱抱", "安慰", "生气")
        knowledge_tokens = ("是什么", "为什么", "介绍", "解释", "科普", "最新", "新闻", "价格", "汇率")

        if self._looks_like_character_related(text, character_name):
            return IntentExtractionResult(intent="character_related", extracted_topic=self._fallback_topic(text), reasoning="角色相关问题")
        if any(token in text or token in lowered for token in weather_tokens):
            location = self._extract_weather_location(text, recent_conversation)
            return IntentExtractionResult(
                intent="weather_query",
                needs_tool=True,
                tool_name="weather",
                tool_params={
                    "location": location,
                    "location_confidence": "high" if location else "low",
                    "location_source": "current_input" if location and location in text else "context",
                },
                extracted_topic=self._fallback_topic(text),
                reasoning="天气问题",
            )
        if any(token in text for token in emotional_tokens):
            return IntentExtractionResult(intent="emotional_interaction", extracted_topic=self._fallback_topic(text), reasoning="情绪互动")
        if any(token in text for token in knowledge_tokens) or re.search(r"[A-Za-z]{2,}[A-Za-z0-9:_-]*", text):
            return IntentExtractionResult(
                intent="web_search_query",
                needs_tool=True,
                tool_name="web_search",
                tool_params={"search_query": self._fallback_topic(text), "search_type": self._infer_search_type(text), "language": "zh"},
                extracted_topic=self._fallback_topic(text),
                reasoning="外部知识问题",
            )
        return IntentExtractionResult(intent="casual_chat", extracted_topic=self._fallback_topic(text), reasoning="普通闲聊")

    def _looks_like_character_related(self, text: str, character_name: str) -> bool:
        value = str(text or "")
        strong_patterns = (
            "你是谁",
            "自我介绍",
            "你的过去",
            "你以前",
            "你的性格",
            "你的经历",
            "你的故事",
            "你的设定",
            "口头禅",
            "说话方式",
        )
        if any(token in value for token in strong_patterns):
            return True
        if re.search(r"你.*(喜不喜欢|讨不讨厌|偏好|害怕|在意)", value):
            return True
        if character_name and character_name in value and any(token in value for token in ("是谁", "介绍", "设定", "故事", "经历")):
            return True
        return False

    def _infer_search_type(self, text: str) -> str:
        value = str(text or "")
        if any(token in value for token in ("最新", "最近", "今天", "现在", "新闻", "票房", "比分", "汇率", "价格")):
            return "realtime"
        return "knowledge"

    def _normalize_location(self, value: Any) -> str | None:
        location = str(value or "").strip()
        if not location or location.lower() == "null":
            return None
        cleaned = re.sub(r"[\s，。！？]+", "", location)
        return cleaned or None

    def _extract_weather_location(self, text: str, recent_conversation: str) -> str | None:
        text = str(text or "").strip()
        sanitized = self._sanitize_weather_location(text)
        if sanitized:
            return sanitized
        candidates = ["东京", "北京市", "北京", "上海", "广州", "深圳", "杭州", "南京", "京都", "纽约", "伦敦", "巴黎", "首尔"]
        for candidate in candidates:
            if candidate in text:
                return candidate
        if any(token in text for token in ("那里", "那边")):
            for candidate in candidates:
                if candidate in recent_conversation:
                    return candidate
        return None

    def _sanitize_weather_location(self, text: str) -> str | None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        cleaned = re.sub(r"^(请问|帮我|我想知道|我想问|今天|现在|目前)\s*", "", cleaned)
        cleaned = re.sub(r"(的)?(天气|气温|温度|会不会下雨|会不会下雪).*$", "", cleaned)
        cleaned = re.sub(r"[\s，。！？]+", "", cleaned)
        return cleaned.strip() or None

    def _fallback_topic(self, text: str) -> str:
        cleaned = re.sub(r"^(请问|可以|能不能|麻烦|帮我|我想知道|我想问)\s*", "", str(text or "").strip())
        return cleaned.strip("，。！？；? ") or "当前话题"

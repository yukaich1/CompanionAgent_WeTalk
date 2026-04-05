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
    "additionalProperties": True,
}


class IntentExtractionResult(BaseModel):
    intent: str = "casual_chat"
    needs_tool: bool = False
    tool_name: str | None = None
    tool_params: dict[str, Any] | None = Field(default_factory=dict)
    extracted_topic: str = ""
    reasoning: str = ""


PROMPT_INTENT_EXTRACTION = """
你是一个对话意图分析器。
你的任务是分析用户最新的输入，判断需要执行什么操作。

重要规则：
1. 必须结合最近 3 轮对话理解省略、承接和指代。
2. 如果问题是角色本身相关（经历、感受、喜好、看法、价值观、设定、自我介绍），优先判为 character_related，不要默认触发工具。
3. 如果问题明显需要现实世界实时或外部知识，才触发工具。
4. 天气问题需要尽量提取标准地点名；无法确定地点时，location 设为 null，不要猜。
5. web_search 的 search_query 必须是优化后的搜索词，不要直接照抄原始用户输入。

最近对话（最近 3 轮）：
{recent_conversation}

当前角色：
{character_name}

用户最新输入：
{user_input}

请只输出 JSON：
{{
  "intent": "<意图类型>",
  "needs_tool": true 或 false,
  "tool_name": "<工具名或null>",
  "tool_params": {{<工具参数对象或null>}},
  "extracted_topic": "<核心话题>",
  "reasoning": "<一句话判断依据>"
}}

intent 只能是以下之一：
- character_related
- weather_query
- web_search_query
- casual_chat
- emotional_interaction
- value_judgment
- help_request

天气查询时 tool_params 格式：
{{
  "location": "<标准地点名或null>",
  "location_confidence": "high 或 low",
  "location_source": "current_input 或 context"
}}

联网搜索时 tool_params 格式：
{{
  "search_query": "<优化后的搜索词>",
  "search_type": "realtime 或 knowledge",
  "language": "zh 或 ja 或 en"
}}
""".strip()


class IntentExtractor:
    def __init__(self, model=None):
        self.model = model or FallbackMistralLLM()

    def extract(self, user_input: str, recent_conversation: str = "", character_name: str = "") -> IntentExtractionResult:
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
                max_tokens=280,
            )
            return self._normalize_result(data or {}, user_input, recent_conversation)
        except Exception:
            return self._heuristic_fallback(user_input, recent_conversation)

    def _normalize_result(self, data: dict[str, Any], user_input: str, recent_conversation: str) -> IntentExtractionResult:
        result = IntentExtractionResult(**data)
        if not result.extracted_topic:
            result.extracted_topic = self._fallback_topic(user_input)
        if result.intent == "weather_query":
            result.needs_tool = True
            result.tool_name = "weather"
        elif result.intent == "web_search_query":
            result.needs_tool = True
            result.tool_name = "web_search"
        elif result.intent == "character_related":
            result.needs_tool = False
            result.tool_name = None
        if not isinstance(result.tool_params, dict):
            result.tool_params = {}
        return result

    def _heuristic_fallback(self, user_input: str, recent_conversation: str) -> IntentExtractionResult:
        text = str(user_input or "").strip()
        lowered = text.lower()
        weather_tokens = ("天气", "气温", "温度", "下雨", "下雪", "晴", "阴", "冷不冷", "热不热", "weather", "forecast")
        persona_tokens = ("经历", "故事", "设定", "性格", "口头禅", "喜欢", "讨厌", "价值观", "世界观", "你是谁", "自我介绍")
        emotional_tokens = ("难过", "伤心", "委屈", "抱抱", "安慰", "生气", "讨厌", "喜欢你")

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
                extracted_topic=f"{location or '天气'}天气",
                reasoning="用户在询问天气信息。",
            )

        if any(token in text for token in emotional_tokens):
            return IntentExtractionResult(
                intent="emotional_interaction",
                extracted_topic=self._fallback_topic(text),
                reasoning="用户在进行情感互动。",
            )

        if any(token in text for token in persona_tokens):
            return IntentExtractionResult(
                intent="character_related",
                extracted_topic=self._fallback_topic(text),
                reasoning="用户在询问角色本身的设定或经历。",
            )

        if re.search(r"[A-Za-z]{2,}[A-Za-z0-9:_-]*", text) or any(
            token in text for token in ("评价", "介绍", "资料", "信息", "新闻", "最近", "最新")
        ):
            return IntentExtractionResult(
                intent="web_search_query",
                needs_tool=True,
                tool_name="web_search",
                tool_params={
                    "search_query": self._fallback_topic(text),
                    "search_type": "knowledge",
                    "language": "zh",
                },
                extracted_topic=self._fallback_topic(text),
                reasoning="用户在询问需要外部确认的现实信息。",
            )

        return IntentExtractionResult(
            intent="casual_chat",
            extracted_topic=self._fallback_topic(text),
            reasoning="用户在进行日常闲聊。",
        )

    def _extract_weather_location(self, text: str, recent_conversation: str) -> str | None:
        candidates = [
            "东京",
            "东京都",
            "北京",
            "上海",
            "广州",
            "深圳",
            "杭州",
            "南京",
            "大阪",
            "京都",
            "名古屋",
            "福冈",
            "横滨",
            "纽约",
            "伦敦",
            "巴黎",
            "首尔",
        ]
        for candidate in candidates:
            if candidate in text:
                return candidate
        if any(token in text for token in ("那边", "那里的", "的呢", "呢")):
            for candidate in candidates:
                if candidate in recent_conversation:
                    return candidate
        return None

    def _fallback_topic(self, text: str) -> str:
        cleaned = re.sub(r"^(请问|可以|能不能|麻烦|帮我|我想知道|我想问)\s*", "", str(text or "").strip())
        return cleaned.strip("，。！？?；： ") or "当前话题"

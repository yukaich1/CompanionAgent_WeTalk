from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from knowledge.knowledge_source import SearchMode


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    reason: str


@dataclass
class ToolExecutionReport:
    calls: list[ToolCall]
    results: list[dict[str, Any]]
    context: str


GENERAL_SEARCH_KEYWORDS = (
    "谁是",
    "是谁",
    "如何评价",
    "怎么评价",
    "怎么看",
    "介绍一下",
    "什么是",
    "资料",
    "信息",
    "新闻",
    "战绩",
    "职业选手",
    "选手",
    "战队",
    "比赛",
    "冠军",
    "排名",
    "作品",
    "游戏",
    "动漫",
    "电影",
    "小说",
    "公司",
    "品牌",
    "where",
    "what is",
    "who is",
    "news",
    "profile",
    "summary",
    "review",
    "evaluate",
    "latest",
)
PERSONA_SEARCH_KEYWORDS = (
    "故事",
    "经历",
    "设定",
    "背景",
    "关系",
    "原作",
    "剧情",
    "性格",
    "口头禅",
    "口癖",
    "说话方式",
    "语气",
    "称呼",
    "喜欢",
    "讨厌",
    "厌恶",
    "价值观",
    "世界观",
    "身份",
    "外貌",
    "外观",
    "发色",
    "喜好",
    "习惯",
    "经典台词",
    "自我介绍",
    "你是谁",
)
WEATHER_KEYWORDS = ("天气", "气温", "温度", "下雨", "晴天", "阴天", "weather", "forecast")


class ToolRuntime:
    def __init__(self, registry):
        self.registry = registry

    def _should_use_weather(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(token in text for token in WEATHER_KEYWORDS) or any(token in lowered for token in WEATHER_KEYWORDS)

    def _is_persona_search(self, text: str) -> bool:
        text = text or ""
        return any(keyword in text for keyword in PERSONA_SEARCH_KEYWORDS)

    def _looks_like_external_fact_query(self, text: str) -> bool:
        lowered = (text or "").lower()
        if any(keyword in text for keyword in GENERAL_SEARCH_KEYWORDS):
            return True
        if any(
            keyword in lowered
            for keyword in ("who is", "what is", "latest", "news", "player", "team", "match", "review", "evaluate", "tell me about", "introduce")
        ):
            return True
        if any(keyword in text for keyword in ("评价", "介绍", "资料", "信息", "新闻", "战绩", "排名", "是什么", "是谁")):
            return True
        if bool(re.search(r"[A-Za-z]{2,}[A-Za-z0-9_-]*", text)):
            return True
        return text.endswith(("?", "？")) and len(text) >= 4

    def infer_search_mode(self, user_input: str) -> SearchMode:
        if self._should_use_weather(user_input):
            return SearchMode.REALITY_SEARCH
        if self._is_persona_search(user_input):
            return SearchMode.PERSONA_SEARCH
        if self._looks_like_external_fact_query(user_input):
            return SearchMode.REALITY_SEARCH
        return SearchMode.NONE

    def plan(
        self,
        user_input: str,
        persona_name: str = "",
        recent_context: str = "",
        search_mode: SearchMode | None = None,
        persona_query: str | None = None,
        reality_query: str | None = None,
    ) -> list[ToolCall]:
        text = user_input if isinstance(user_input, str) else str(user_input or "")
        recent_context = recent_context if isinstance(recent_context, str) else str(recent_context or "")
        mode = search_mode or self.infer_search_mode(text)
        calls: list[ToolCall] = []

        if self._should_use_weather(text) and mode in {SearchMode.REALITY_SEARCH, SearchMode.BOTH}:
            weather_query = "\n".join(part for part in (recent_context.strip(), reality_query or text) if part).strip()
            calls.append(
                ToolCall(
                    name="weather",
                    arguments={"query": weather_query},
                    reason="The user is asking for current weather information.",
                )
            )

        if mode in {SearchMode.PERSONA_SEARCH, SearchMode.BOTH}:
            query = persona_query or text
            calls.append(
                ToolCall(
                    name="web_search",
                    arguments={"persona_name": persona_name, "query": query, "max_results": 4, "timeout": 8},
                    reason="The user is asking for persona-grounded factual information.",
                )
            )

        if mode in {SearchMode.REALITY_SEARCH, SearchMode.BOTH} and not self._should_use_weather(text):
            query = reality_query or text
            calls.append(
                ToolCall(
                    name="web_search",
                    arguments={"persona_name": "", "query": query, "max_results": 4, "timeout": 8},
                    reason="The user is asking for grounded external information.",
                )
            )
        return calls

    def execute(
        self,
        user_input: str,
        persona_name: str = "",
        recent_context: str = "",
        search_mode: SearchMode | None = None,
        persona_query: str | None = None,
        reality_query: str | None = None,
    ) -> ToolExecutionReport:
        calls = self.plan(
            user_input,
            persona_name=persona_name,
            recent_context=recent_context,
            search_mode=search_mode,
            persona_query=persona_query,
            reality_query=reality_query,
        )
        results: list[dict[str, Any]] = []
        context_blocks: list[str] = []

        for call in calls:
            try:
                result = self.registry.run(call.name, **call.arguments)
                ok = result.get("ok", True) if isinstance(result, dict) else True
            except Exception as exc:
                result = {"ok": False, "summary": str(exc)}
                ok = False

            results.append({"tool": call.name, "reason": call.reason, "result": result, "ok": ok})

            if call.name == "weather":
                summary = result.get("summary", "") if isinstance(result, dict) else str(result)
                context_blocks.append(
                    "\n".join(
                        [
                            "Realtime Tool Result: weather",
                            f"Reason: {call.reason}",
                            f"Result: {summary or 'No weather result.'}",
                        ]
                    )
                )
                continue

            snippets = result.get("snippets", []) if isinstance(result, dict) else []
            if snippets:
                lines = [f"[{item.get('source', 'web')} | {item.get('title', '')}] {item.get('text', '')}" for item in snippets[:4]]
                heading = "Persona Web Result" if call.arguments.get("persona_name") else "Reference Tool Result: web_search"
                context_blocks.append("\n".join([heading, f"Reason: {call.reason}", *lines]))

        return ToolExecutionReport(
            calls=calls,
            results=results,
            context="\n\n".join(block for block in context_blocks if block).strip() or "None",
        )

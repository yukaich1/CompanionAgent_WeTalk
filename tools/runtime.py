from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    reason: str


@dataclass
class ToolExecutionReport:
    calls: List[ToolCall]
    results: List[Dict[str, Any]]
    context: str


class ToolRuntime:
    def __init__(self, registry):
        self.registry = registry

    def _should_use_weather(self, text):
        lower = (text or "").lower()
        return any(token in text for token in ("天气", "气温", "温度", "下雨", "晴天", "阴天")) or "weather" in lower

    def _should_use_web_search(self, text):
        return any(keyword in (text or "") for keyword in ("故事", "经历", "设定", "背景", "关系", "原作", "剧情"))

    def plan(self, user_input, persona_name="", recent_context=""):
        text = user_input if isinstance(user_input, str) else str(user_input or "")
        recent_context = recent_context if isinstance(recent_context, str) else str(recent_context or "")
        weather_query = "\n".join(part for part in (recent_context.strip(), text.strip()) if part).strip() or text
        calls = []
        if self._should_use_weather(text):
            calls.append(ToolCall(name="weather", arguments={"query": weather_query}, reason="The user is asking for real-time weather information."))
        if self._should_use_web_search(text):
            calls.append(
                ToolCall(
                    name="web_search",
                    arguments={"persona_name": persona_name, "query": text, "max_results": 2, "timeout": 8},
                    reason="The user is asking for factual character or story information that should be grounded in references.",
                )
            )
        return calls

    def execute(self, user_input, persona_name="", recent_context=""):
        calls = self.plan(user_input, persona_name=persona_name, recent_context=recent_context)
        results = []
        context_blocks = []
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
                            "Use this result directly if the user is asking about weather or current conditions.",
                        ]
                    )
                )
            elif call.name == "web_search":
                snippets = result.get("snippets", []) if isinstance(result, dict) else []
                if snippets:
                    lines = [f"[{item.get('source', 'web')} | {item.get('title', '')}] {item.get('text', '')}" for item in snippets[:2]]
                    context_blocks.append(
                        "\n".join(
                            [
                                "Reference Tool Result: web_search",
                                f"Reason: {call.reason}",
                                *lines,
                                "Treat these snippets as supporting references. Prefer supported facts and avoid invention.",
                            ]
                        )
                    )

        return ToolExecutionReport(
            calls=calls,
            results=results,
            context="\n\n".join(block for block in context_blocks if block).strip() or "None",
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.intent_extractor import IntentExtractionResult


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    reason: str


@dataclass
class ToolExecutionReport:
    calls: list[ToolCall] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    context: str = "None"
    follow_up_message: str | None = None


class ToolRouter:
    """接收结构化意图并执行工具，统一处理缺参、失败和结果整理。"""

    def __init__(self, registry):
        self.registry = registry

    def execute_intent(self, intent_result: IntentExtractionResult, persona_name: str = "") -> ToolExecutionReport:
        if not intent_result.needs_tool or not intent_result.tool_name:
            return ToolExecutionReport()

        if intent_result.tool_name == "weather":
            return self._execute_weather(intent_result)
        if intent_result.tool_name == "web_search":
            return self._execute_web_search(intent_result, persona_name)
        return ToolExecutionReport()

    def _execute_weather(self, intent_result: IntentExtractionResult) -> ToolExecutionReport:
        params = intent_result.tool_params or {}
        location = str(params.get("location") or "").strip()
        confidence = str(params.get("location_confidence") or "low").strip().lower()
        if not location:
            return ToolExecutionReport(
                follow_up_message="你想问哪里的天气？刚才地点还不够明确，我没法替你查。",
            )

        call = ToolCall(
            name="weather",
            arguments={"query": location},
            reason="用户在询问实时天气信息。",
        )
        result = self._run_tool(call, fallback={"ok": False, "location": location, "summary": "天气查询失败"})
        if not result.get("ok"):
            return self._failed_report(call, result)

        summary = str(result.get("summary", "") or "").strip()
        if not summary:
            return self._failed_report(call, result)

        context_lines = [
            "[外部信息 - 天气]",
            f"地点：{result.get('location', location)}",
            f"摘要：{summary}",
        ]
        if confidence == "low":
            context_lines.append("备注：地点来自上下文推断，回答时请保持一点谨慎。")
        return ToolExecutionReport(
            calls=[call],
            results=[{"tool": call.name, "reason": call.reason, "result": result, "ok": True}],
            context="\n".join(context_lines),
        )

    def _execute_web_search(self, intent_result: IntentExtractionResult, persona_name: str) -> ToolExecutionReport:
        params = intent_result.tool_params or {}
        query = str(params.get("search_query") or intent_result.extracted_topic or "").strip()
        if not query:
            return ToolExecutionReport()

        call = ToolCall(
            name="web_search",
            arguments={
                "persona_name": "",
                "query": query,
                "max_results": 4,
                "timeout": 8,
                "source_mode": "general",
            },
            reason="当前问题需要外部信息支撑回答。",
        )
        result = self._run_tool(call, fallback={"snippets": [], "summary": "", "ok": False})
        snippets = result.get("snippets", []) if isinstance(result, dict) else []
        summary = str(result.get("summary", "") or "").strip() if isinstance(result, dict) else ""

        context_lines: list[str] = []
        if summary:
            context_lines.extend(["[外部信息 - 搜索]", f"主题：{intent_result.extracted_topic or query}", f"摘要：{summary}"])
        for item in snippets[:4]:
            title = str(item.get("title", "") or "").strip()
            text = str(item.get("text", "") or "").strip()
            source = str(item.get("source", "web") or "web").strip()
            if text:
                if not context_lines:
                    context_lines.extend(["[外部信息 - 搜索]", f"主题：{intent_result.extracted_topic or query}"])
                line = f"[{source}"
                if title:
                    line += f" | {title}"
                line += f"] {text}"
                context_lines.append(line)

        if not context_lines:
            return self._failed_report(call, result)

        return ToolExecutionReport(
            calls=[call],
            results=[{"tool": call.name, "reason": call.reason, "result": result, "ok": True}],
            context="\n".join(context_lines),
        )

    def _run_tool(self, call: ToolCall, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.registry.run(call.name, **call.arguments)
        except Exception as exc:
            result = dict(fallback)
            result["summary"] = str(exc)
            result["ok"] = False
        return result if isinstance(result, dict) else dict(fallback)

    def _failed_report(self, call: ToolCall, result: dict[str, Any]) -> ToolExecutionReport:
        return ToolExecutionReport(
            calls=[call],
            results=[{"tool": call.name, "reason": call.reason, "result": result, "ok": False}],
            context="None",
        )

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class ToolSafetyDecision:
    decision: str = "allow"
    reason: str = ""


class ToolSafetyReviewer:
    HARD_PATTERNS = (
        r"(?i)\b(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|secret|password|cookie|session[_ -]?id)\b",
        r"(?i)\b(\.env|id_rsa|known_hosts|passwd|credential)\b",
        r"(?i)\b(read|dump|show|reveal|export|leak).{0,40}\b(secret|token|password|cookie|credential)\b",
        r"(?i)\b(system prompt|hidden prompt|developer prompt)\b",
    )
    SOFT_PATTERNS = (
        r"(?i)\b(ignore previous|ignore above|bypass|jailbreak|prompt injection)\b",
        r"(?i)\b(local file|filesystem|registry|powershell|cmd\.exe|bash)\b",
        r"[\u200b-\u200f\u202a-\u202e\u2060\u3000]",
    )

    def review(self, *, tool_name: str, query: str) -> ToolSafetyDecision:
        text = str(query or "").strip()
        if not text:
            return ToolSafetyDecision()
        if len(text) > 420:
            return ToolSafetyDecision(
                decision="soft_reject",
                reason="工具查询过长，容易混入无关或不安全内容，建议先缩短问题。",
            )
        for pattern in self.HARD_PATTERNS:
            if re.search(pattern, text):
                return ToolSafetyDecision(
                    decision="hard_reject",
                    reason=f"{tool_name} 请求中出现敏感信息或隐含泄露指令，已阻止执行。",
                )
        for pattern in self.SOFT_PATTERNS:
            if re.search(pattern, text):
                return ToolSafetyDecision(
                    decision="soft_reject",
                    reason=f"{tool_name} 请求中包含可疑操作描述，先请用户明确和收缩查询范围。",
                )
        return ToolSafetyDecision(decision="allow", reason="query_ok")

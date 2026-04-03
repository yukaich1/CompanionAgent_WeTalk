from __future__ import annotations

from pydantic import BaseModel, Field

from knowledge.knowledge_source import RouteDecision, SearchMode


class QueryRewriteResult(BaseModel):
    persona_query: str | None = None
    reality_query: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryRewriter:
    def rewrite(self, user_input: str, route_decision: RouteDecision, persona_name: str) -> QueryRewriteResult:
        core = self._extract_core(user_input)
        if route_decision.web_search_mode == SearchMode.NONE:
            return QueryRewriteResult()
        if route_decision.web_search_mode == SearchMode.PERSONA_SEARCH:
            return QueryRewriteResult(
                persona_query=f"{persona_name} {core} 角色设定 性格 说话方式 经历 资料"
            )
        if route_decision.web_search_mode == SearchMode.REALITY_SEARCH:
            return QueryRewriteResult(reality_query=core)
        return QueryRewriteResult(
            persona_query=f"{persona_name} {core} 角色设定 性格 说话方式 经历 资料",
            reality_query=core,
        )

    def _extract_core(self, text: str) -> str:
        cleaned = " ".join(
            (text or "")
            .replace("你的", "")
            .replace("你", "")
            .replace("请问", "")
            .replace("可以", "")
            .replace("能不能", "")
            .split()
        )
        return cleaned.strip(" ？?。！!") or (text or "").strip()

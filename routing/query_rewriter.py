from __future__ import annotations

import re

from pydantic import BaseModel, Field

from knowledge.knowledge_source import RouteDecision, SearchMode
from tools.intent_extractor import IntentExtractionResult


class QueryRewriteResult(BaseModel):
    persona_query: str | None = None
    reality_query: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryRewriter:
    """基于结构化意图结果生成检索和工具查询词。"""

    PERSONA_SUFFIX = "角色设定 性格 说话方式 经历 背景 资料"

    def rewrite(
        self,
        user_input: str,
        route_decision: RouteDecision,
        persona_name: str,
        intent_result: IntentExtractionResult | None = None,
    ) -> QueryRewriteResult:
        intent = intent_result or IntentExtractionResult(extracted_topic=self._extract_core(user_input))
        topic = self._extract_core(intent.extracted_topic or user_input)
        metadata = {"topic": topic}

        if route_decision.web_search_mode == SearchMode.NONE:
            return QueryRewriteResult(metadata=metadata)

        if route_decision.web_search_mode == SearchMode.PERSONA_SEARCH:
            return QueryRewriteResult(
                persona_query=self._build_persona_query(persona_name, topic),
                metadata=metadata,
            )

        if route_decision.web_search_mode == SearchMode.REALITY_SEARCH:
            return QueryRewriteResult(
                reality_query=self._build_reality_query(intent, topic),
                metadata=metadata,
            )

        return QueryRewriteResult(
            persona_query=self._build_persona_query(persona_name, topic),
            reality_query=self._build_reality_query(intent, topic),
            metadata=metadata,
        )

    def _build_persona_query(self, persona_name: str, topic: str) -> str:
        return " ".join(part for part in (persona_name.strip(), topic, self.PERSONA_SUFFIX) if part).strip()

    def _build_reality_query(self, intent: IntentExtractionResult, topic: str) -> str:
        params = intent.tool_params or {}
        if intent.tool_name == "weather":
            location = str(params.get("location") or "").strip()
            return f"{location} 天气".strip() if location else f"{topic} 天气".strip()

        search_query = str(params.get("search_query") or "").strip()
        return search_query or topic

    def _extract_core(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^(请问|能不能|可以|你能|请你|麻烦|帮我|我想知道|我想问)\s*", "", cleaned)
        cleaned = cleaned.replace("你的", "").replace("你呢", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" ，。！？?；：")
        return cleaned or (text or "").strip()

from __future__ import annotations

from knowledge.knowledge_source import RouteDecision, RouteType, SearchMode
from tools.intent_extractor import IntentExtractionResult, IntentExtractor


class QueryRouter:
    """基于轻量意图提取结果决定本轮主路径。"""

    def __init__(self, extractor: IntentExtractor | None = None):
        self.extractor = extractor or IntentExtractor()
        self.last_intent_result = IntentExtractionResult()

    def _ensure_runtime_fields(self) -> None:
        if not hasattr(self, "extractor") or self.extractor is None:
            self.extractor = IntentExtractor()
        if not hasattr(self, "last_intent_result") or self.last_intent_result is None:
            self.last_intent_result = IntentExtractionResult()

    def route(
        self,
        user_input: str,
        persona_recall,
        is_public: bool,
        recent_conversation: str = "",
        character_name: str = "",
    ) -> RouteDecision:
        self._ensure_runtime_fields()
        intent = self.extractor.extract(
            user_input=user_input,
            recent_conversation=recent_conversation,
            character_name=character_name,
        )
        self.last_intent_result = intent
        coverage = float(getattr(persona_recall, "coverage_score", 0.0) or 0.0)

        if intent.intent == "weather_query":
            return RouteDecision(type=RouteType.E4, web_search_mode=SearchMode.REALITY_SEARCH, info_domain="REALITY_FACTUAL")

        if intent.intent == "web_search_query":
            return RouteDecision(type=RouteType.E4, web_search_mode=SearchMode.REALITY_SEARCH, info_domain="REALITY_FACTUAL")

        if intent.intent == "emotional_interaction":
            return RouteDecision(type=RouteType.E5, web_search_mode=SearchMode.NONE, info_domain="RELATIONAL")

        if intent.intent == "casual_chat":
            return RouteDecision(type=RouteType.E1, web_search_mode=SearchMode.NONE, info_domain="CHARACTER_INTERNAL")

        if intent.intent == "help_request" and intent.needs_tool:
            return RouteDecision(type=RouteType.E4, web_search_mode=SearchMode.REALITY_SEARCH, info_domain="REALITY_FACTUAL")

        if intent.intent == "value_judgment":
            if intent.needs_tool:
                return RouteDecision(type=RouteType.E3, web_search_mode=SearchMode.BOTH, info_domain="MIXED")
            return RouteDecision(type=RouteType.E1, web_search_mode=SearchMode.NONE, info_domain="CHARACTER_INTERNAL")

        if coverage >= 0.7:
            return RouteDecision(type=RouteType.E1, web_search_mode=SearchMode.NONE, info_domain="CHARACTER_INTERNAL")

        if is_public:
            return RouteDecision(
                type=RouteType.E2,
                web_search_mode=SearchMode.PERSONA_SEARCH if intent.needs_tool else SearchMode.NONE,
                search_hint=[intent.extracted_topic] if intent.extracted_topic else [],
                info_domain="CHARACTER_INTERNAL",
            )

        return RouteDecision(
            type=RouteType.E2B,
            web_search_mode=SearchMode.NONE,
            search_hint=[intent.extracted_topic] if intent.extracted_topic else [],
            fallback="conservative",
            info_domain="CHARACTER_INTERNAL",
        )

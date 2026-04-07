from __future__ import annotations

from knowledge.knowledge_source import RouteDecision, RouteType, SearchMode
from tools.intent_extractor import IntentExtractionResult, IntentExtractor


class QueryRouter:
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
        intent_result: IntentExtractionResult | None = None,
    ) -> RouteDecision:
        self._ensure_runtime_fields()
        intent = intent_result or self.extractor.extract(
            user_input=user_input,
            recent_conversation=recent_conversation,
            character_name=character_name,
        )
        self.last_intent_result = intent

        search_hint = []
        if intent.extracted_topic:
            search_hint.append(intent.extracted_topic)
        search_hint.extend(list(intent.extracted_keywords or []))
        search_hint = [item for item in search_hint if str(item or "").strip()][:6]

        if intent.response_mode == "external" or intent.needs_tool:
            return RouteDecision(
                type=RouteType.E4,
                web_search_mode=SearchMode.REALITY_SEARCH,
                info_domain="REALITY_FACTUAL",
                search_hint=search_hint,
            )

        if intent.response_mode == "emotional":
            return RouteDecision(
                type=RouteType.E5,
                web_search_mode=SearchMode.NONE,
                info_domain="RELATIONAL",
                search_hint=search_hint,
            )

        if intent.response_mode in {"self_intro", "story", "persona_fact", "value"}:
            return RouteDecision(
                type=RouteType.E2,
                web_search_mode=SearchMode.NONE,
                info_domain="CHARACTER_INTERNAL",
                search_hint=search_hint,
            )

        return RouteDecision(
            type=RouteType.E1,
            web_search_mode=SearchMode.NONE,
            info_domain="CHARACTER_INTERNAL",
            search_hint=search_hint,
        )

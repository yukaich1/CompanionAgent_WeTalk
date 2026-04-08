from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llm import FallbackMistralLLM


INTENT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "response_mode": {"type": "string"},
        "recall_mode": {"type": "string"},
        "persona_focus": {"type": "string"},
        "needs_tool": {"type": "boolean"},
        "tool_name": {"type": ["string", "null"]},
        "tool_params": {"type": ["object", "null"]},
        "extracted_topic": {"type": "string"},
        "extracted_keywords": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "intent",
        "response_mode",
        "recall_mode",
        "persona_focus",
        "needs_tool",
        "tool_name",
        "tool_params",
        "extracted_topic",
        "extracted_keywords",
        "reasoning",
    ],
}


class IntentExtractionResult(BaseModel):
    intent: str = "casual_chat"
    response_mode: str = "casual"
    recall_mode: str = "none"
    persona_focus: str = "general"
    needs_tool: bool = False
    tool_name: str | None = None
    tool_params: dict[str, Any] | None = Field(default_factory=dict)
    extracted_topic: str = ""
    extracted_keywords: list[str] = Field(default_factory=list)
    reasoning: str = ""


PROMPT_INTENT_EXTRACTION = """
You are a dialogue routing analyzer for a role-playing chat system.

Your job:
1. Understand the user's actual intent from the current message and recent conversation.
2. Decide which response path should be used.
3. Decide whether persona retrieval is needed, whether a tool is needed, and what topic/keywords should drive retrieval.

Allowed values:

intent:
- character_related
- weather_query
- web_search_query
- casual_chat
- emotional_interaction
- value_judgment
- help_request

response_mode:
- self_intro
- story
- persona_fact
- external
- emotional
- value
- casual

recall_mode:
- none
- identity
- persona
- story

persona_focus:
- general
- likes
- dislikes
- catchphrase
- personality
- self_intro

Decision rules:
- Use character_related only when the user is asking about the current role character themself.
- Use story only when the user is genuinely asking for the character's past experience, a concrete event, or a story retelling.
- If the user is just chatting, reacting, venting, greeting, or continuing a natural conversation, do not force story retrieval.
- Use weather_query only when the user is explicitly asking about weather.
- Use web_search_query only when the user is explicitly asking for real-world facts, explanations, introductions, or external information.
- If the user mentions a real-world topic while expressing emotion or casual conversation, do not automatically trigger search.
- extracted_topic should be a short phrase describing the true topic.
- extracted_keywords should contain 3 to 10 useful retrieval/search keywords, prioritizing entities, domain terms, constraints, and named concepts from the user.
- If weather_query is chosen, put the location in tool_params.location.
- If web_search_query or help_request is chosen, put a concise search string in tool_params.search_query.
- Output JSON only.

Recent conversation:
{recent_conversation}

Current character name:
{character_name}

User input:
{user_input}
""".strip()


class IntentExtractor:
    def __init__(self, model=None):
        self.model = model or FallbackMistralLLM()

    def extract(self, user_input: str, recent_conversation: str = "", character_name: str = "") -> IntentExtractionResult:
        prompt = PROMPT_INTENT_EXTRACTION.format(
            recent_conversation=recent_conversation or "None",
            character_name=character_name or "Current character",
            user_input=user_input or "",
        )
        try:
            data = self.model.generate(
                prompt,
                return_json=True,
                schema=INTENT_EXTRACTION_SCHEMA,
                temperature=0.1,
                max_tokens=320,
            )
            return self._normalize_result(data or {}, user_input)
        except Exception:
            return IntentExtractionResult(
                intent="casual_chat",
                response_mode="casual",
                recall_mode="none",
                persona_focus="general",
                needs_tool=False,
                tool_name=None,
                tool_params={},
                extracted_topic=str(user_input or "").strip(),
                extracted_keywords=[],
                reasoning="llm_failed_default_to_casual_chat",
            )

    def _normalize_result(self, data: dict[str, Any], user_input: str) -> IntentExtractionResult:
        result = IntentExtractionResult(**data)

        allowed_intents = {
            "character_related",
            "weather_query",
            "web_search_query",
            "casual_chat",
            "emotional_interaction",
            "value_judgment",
            "help_request",
        }
        allowed_modes = {"self_intro", "story", "persona_fact", "external", "emotional", "value", "casual"}
        allowed_recall = {"none", "identity", "persona", "story"}
        allowed_focus = {"general", "likes", "dislikes", "catchphrase", "personality", "self_intro"}

        result.intent = result.intent if result.intent in allowed_intents else "casual_chat"
        result.response_mode = result.response_mode if result.response_mode in allowed_modes else "casual"
        result.recall_mode = result.recall_mode if result.recall_mode in allowed_recall else "none"
        result.persona_focus = result.persona_focus if result.persona_focus in allowed_focus else "general"
        result.extracted_topic = str(result.extracted_topic or user_input or "").strip()
        result.extracted_keywords = [
            str(item or "").strip()
            for item in list(result.extracted_keywords or [])
            if str(item or "").strip()
        ][:10]

        if result.intent == "weather_query":
            result.needs_tool = True
            result.tool_name = "weather"
            result.response_mode = "external"
            result.recall_mode = "none"
            params = dict(result.tool_params or {})
            location = str(params.get("location", "") or "").strip()
            normalized = {"location": location} if location else {}
            confidence = str(params.get("location_confidence", "") or "").strip()
            if confidence:
                normalized["location_confidence"] = confidence
            result.tool_params = normalized
            return result

        if result.intent in {"web_search_query", "help_request"}:
            result.needs_tool = True
            result.tool_name = "web_search"
            result.response_mode = "external"
            result.recall_mode = "none"
            params = dict(result.tool_params or {})
            search_query = str(params.get("search_query", "") or result.extracted_topic or user_input or "").strip()
            result.tool_params = {"search_query": search_query, "language": "zh"}
            return result

        if result.intent == "character_related":
            result.needs_tool = False
            result.tool_name = None
            result.tool_params = {}
            if result.recall_mode == "none":
                result.recall_mode = "persona"
            if result.response_mode == "casual":
                result.response_mode = "persona_fact"
            return result

        if result.intent == "value_judgment":
            result.needs_tool = False
            result.tool_name = None
            result.tool_params = {}
            if result.response_mode == "casual":
                result.response_mode = "value"
            return result

        if result.intent == "emotional_interaction":
            result.needs_tool = False
            result.tool_name = None
            result.tool_params = {}
            if result.response_mode == "casual":
                result.response_mode = "emotional"
            result.recall_mode = "none"
            return result

        result.intent = "casual_chat"
        result.response_mode = "casual"
        result.recall_mode = "none"
        result.persona_focus = "general"
        result.needs_tool = False
        result.tool_name = None
        result.tool_params = {}
        return result

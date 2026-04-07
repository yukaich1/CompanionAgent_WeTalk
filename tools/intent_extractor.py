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
你是对话路由分析器。你要判断当前用户问题属于哪一类，并给后续检索和工具调用提供最有帮助的主题与关键词。

可选 intent 只能是：
- `character_related`: 用户在问当前角色本人的身份、设定、性格、喜恶、态度、经历、故事、口头禅、说话方式
- `weather_query`: 用户在问天气、气温、降雨、风力、体感、穿衣建议等天气信息
- `web_search_query`: 用户在问与当前角色资料无关的现实知识、新闻、百科、最新信息、价格、汇率、比赛结果等
- `casual_chat`: 普通闲聊，不需要工具，也不依赖角色事实资料
- `emotional_interaction`: 用户在倾诉、表达情绪、寻求安慰、陪伴或情绪回应
- `value_judgment`: 用户在问观点、评价、看法，需要角色表达态度
- `help_request`: 用户明确要求查询、搜索、总结某个现实主题，通常需要工具

`response_mode` 只能是：
- `self_intro`
- `story`
- `persona_fact`
- `external`
- `emotional`
- `value`
- `casual`

`recall_mode` 只能是：
- `none`
- `identity`
- `persona`
- `story`

`persona_focus` 只能是：
- `general`
- `likes`
- `dislikes`
- `catchphrase`
- `personality`
- `self_intro`

输出要求：
1. `extracted_topic` 用一句短语概括用户真正关心的主题。
2. `extracted_keywords` 给出 3 到 10 个最适合后续检索或搜索的关键词，优先保留用户原话里的实体、领域、限定条件。
3. 如果是 `weather_query`，请在 `tool_params.location` 中提取地点；如果用户没明确给地点，可以结合最近对话推断，但不要编造。
4. 如果是 `web_search_query` 或 `help_request`，请在 `tool_params.search_query` 中生成简洁可搜索的查询词。
5. 只有当用户问的是“当前角色本人”的身份、经历、设定、态度时，才能归为 `character_related`。
6. 如果用户问的是别的人、别的战队、别的作品、别的现实概念，即使句式里有“你认识吗”，也应归为 `web_search_query` 或 `help_request`。
7. 不要依赖固定候选词匹配，要按语义理解。
8. 只输出 JSON，不要输出解释性文字。

最近对话：
{recent_conversation}

当前角色名：
{character_name}

用户输入：
{user_input}
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
        result.extracted_topic = str(result.extracted_topic or user_input or "").strip()
        result.response_mode = str(result.response_mode or "casual").strip() or "casual"
        result.recall_mode = str(result.recall_mode or "none").strip() or "none"
        result.persona_focus = str(result.persona_focus or "general").strip() or "general"
        result.extracted_keywords = [
            str(item or "").strip()
            for item in list(result.extracted_keywords or [])
            if str(item or "").strip()
        ][:10]

        if result.intent in {"character_related", "value_judgment", "casual_chat", "emotional_interaction"}:
            result.needs_tool = False
            result.tool_name = None
            result.tool_params = {}
            if result.intent == "character_related" and result.recall_mode not in {"identity", "persona", "story"}:
                result.recall_mode = "persona"
            if result.intent == "value_judgment" and result.response_mode == "casual":
                result.response_mode = "value"
            if result.intent == "emotional_interaction" and result.response_mode == "casual":
                result.response_mode = "emotional"
            return result

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

        result.intent = "casual_chat"
        result.response_mode = "casual"
        result.recall_mode = "none"
        result.persona_focus = "general"
        result.needs_tool = False
        result.tool_name = None
        result.tool_params = {}
        return result

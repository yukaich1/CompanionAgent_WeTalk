from __future__ import annotations

from config import DEFAULT_CONFIG
from knowledge.knowledge_source import AssembledContext, DeduplicatedContext, RouteType


class ContextAssembler:
    """
    把人设、记忆、工具结果和思考结果整理成最终注入给模型的上下文。

    设计原则：
    - 角色事实和证据优先
    - 现实工具结果与角色证据分开
    - 记忆只提供关系和互动背景
    - 不重复注入已经由角色模板 prompt 承担的风格约束
    """

    def assemble(
        self,
        route_type: RouteType,
        deduped: DeduplicatedContext,
        thought_output: str = "",
        web_persona_context: str = "",
        web_reality_context: str = "",
    ) -> AssembledContext:
        slots = {
            "immutable_core": deduped.persona.integrated_context,
            "thought_output": thought_output,
            "web_persona_context": web_persona_context,
            "web_reality_context": web_reality_context,
            "semantic_memory": "\n".join(record.content for record in deduped.memory.semantic_records),
            "episodic_memory": "\n".join(
                f"[{record.inject_mode}] {record.content}" for record in deduped.memory.episodic_records
            ),
            "evidence_chunks": "\n".join(deduped.persona.evidence_chunks),
            "relation_state": str(deduped.memory.relation_state),
        }
        slots = self._apply_route_weights(route_type, slots)
        return AssembledContext(
            route_type=route_type,
            slots=slots,
            token_budget=DEFAULT_CONFIG.context_window_tokens,
            metadata={"coverage_score": deduped.persona.coverage_score},
        )

    def build_prompt_context(self, assembled: AssembledContext) -> str:
        slots = assembled.slots
        sections: list[str] = []

        fact_parts = [
            part
            for part in (
                slots.get("immutable_core", ""),
                slots.get("evidence_chunks", ""),
                slots.get("web_persona_context", ""),
            )
            if part
        ]
        if fact_parts:
            sections.append(
                "=== 角色相关事实与证据 ===\n"
                "下面的内容是和当前角色直接相关的依据。回答角色设定、经历、喜恶、口头禅、价值观等问题时，必须优先依赖这里。\n\n"
                + "\n\n".join(fact_parts)
            )

        if slots.get("web_reality_context"):
            sections.append(
                "=== 现实世界工具结果 ===\n"
                "下面是本轮查到的现实信息。涉及天气、新闻、人物、作品、比赛等现实问题时，必须优先依赖这里。\n\n"
                + slots["web_reality_context"]
            )

        memory_parts = []
        if slots.get("semantic_memory"):
            memory_parts.append("【长期印象】\n" + slots["semantic_memory"])
        if slots.get("episodic_memory"):
            memory_parts.append("【近期互动】\n" + slots["episodic_memory"])
        if slots.get("relation_state"):
            memory_parts.append("【关系状态】\n" + slots["relation_state"])
        if memory_parts:
            sections.append("=== 记忆与关系背景 ===\n\n" + "\n\n".join(memory_parts))

        if slots.get("thought_output"):
            sections.append("=== 当前内心推理摘要 ===\n\n" + slots["thought_output"])

        return "\n\n".join(section for section in sections if section).strip()

    def _apply_route_weights(self, route_type: RouteType, slots: dict[str, str]) -> dict[str, str]:
        adjusted = dict(slots)
        if route_type == RouteType.E1:
            adjusted["web_persona_context"] = ""
            adjusted["web_reality_context"] = ""
        elif route_type == RouteType.E2:
            adjusted["web_reality_context"] = ""
        elif route_type == RouteType.E2B:
            adjusted["web_persona_context"] = ""
            if not adjusted.get("web_reality_context"):
                adjusted["web_reality_context"] = "没有可靠外部资料时，必须保守回答。"
        elif route_type == RouteType.E4:
            adjusted["immutable_core"] = ""
            adjusted["evidence_chunks"] = ""
            if not adjusted.get("web_reality_context"):
                adjusted["web_reality_context"] = "当前没有工具查询结果。对于现实事实问题必须承认不确定，不得猜测或编造。"
        elif route_type == RouteType.E5:
            adjusted["web_persona_context"] = ""
            adjusted["web_reality_context"] = ""
        return adjusted

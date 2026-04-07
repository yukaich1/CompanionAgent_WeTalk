from __future__ import annotations

from config import DEFAULT_CONFIG
from knowledge.knowledge_source import AssembledContext, DeduplicatedContext, RouteType


class ContextAssembler:
    def assemble(
        self,
        route_type: RouteType,
        deduped: DeduplicatedContext,
        thought_output: str = "",
        web_persona_context: str = "",
        web_reality_context: str = "",
    ) -> AssembledContext:
        slots = {
            "persona_context": deduped.persona.integrated_context,
            "persona_evidence": "\n".join(deduped.persona.evidence_chunks),
            "memory_context": "\n".join(record.content for record in deduped.memory.episodic_records),
            "semantic_memory": "\n".join(record.content for record in deduped.memory.semantic_records),
            "relation_state": str(deduped.memory.relation_state),
            "web_persona_context": web_persona_context,
            "web_reality_context": web_reality_context,
            "thought_output": thought_output,
            "evidence_chunks": "\n".join(part for part in deduped.persona.evidence_chunks if part),
            "immutable_core": deduped.persona.integrated_context,
        }
        return AssembledContext(
            route_type=route_type,
            slots=self._apply_route_policy(route_type, slots),
            token_budget=DEFAULT_CONFIG.context_window_tokens,
            metadata={"coverage_score": deduped.persona.coverage_score},
        )

    def build_prompt_context(self, assembled: AssembledContext) -> str:
        slots = assembled.slots
        sections: list[str] = []

        if slots.get("persona_context") or slots.get("persona_evidence") or slots.get("web_persona_context"):
            persona_parts = [
                part
                for part in (
                    slots.get("persona_context", ""),
                    slots.get("persona_evidence", ""),
                    slots.get("web_persona_context", ""),
                )
                if part
            ]
            sections.append("=== 角色检索上下文 ===\n" + "\n\n".join(persona_parts))

        if slots.get("web_reality_context"):
            sections.append("=== 外部检索结果 ===\n" + slots["web_reality_context"])

        memory_parts = [
            part
            for part in (
                slots.get("memory_context", ""),
                slots.get("semantic_memory", ""),
                slots.get("relation_state", ""),
            )
            if part
        ]
        if memory_parts:
            sections.append("=== 记忆检索上下文 ===\n" + "\n\n".join(memory_parts))

        if slots.get("thought_output"):
            sections.append("=== 当前内部状态 ===\n" + slots["thought_output"])

        return "\n\n".join(section for section in sections if section).strip()

    def _apply_route_policy(self, route_type: RouteType, slots: dict[str, str]) -> dict[str, str]:
        adjusted = dict(slots)
        if route_type == RouteType.E1:
            adjusted["web_persona_context"] = ""
            adjusted["web_reality_context"] = ""
        elif route_type == RouteType.E2:
            adjusted["web_reality_context"] = ""
        elif route_type == RouteType.E2B:
            adjusted["web_persona_context"] = ""
        elif route_type == RouteType.E4:
            adjusted["persona_context"] = ""
            adjusted["persona_evidence"] = ""
            adjusted["evidence_chunks"] = ""
            adjusted["immutable_core"] = ""
        return adjusted

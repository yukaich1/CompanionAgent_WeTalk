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
            "immutable_core": deduped.persona.integrated_context,
            "thought_output": thought_output,
            "web_persona_context": web_persona_context,
            "slow_change_state": "",
            "web_reality_context": web_reality_context,
            "semantic_memory": "\n".join(record.content for record in deduped.memory.semantic_records),
            "episodic_memory": "\n".join(
                f"[{record.inject_mode}] {record.content}" for record in deduped.memory.episodic_records
            ),
            "evidence_chunks": "\n".join(deduped.persona.evidence_chunks),
            "relation_state": str(deduped.memory.relation_state),
        }
        return AssembledContext(
            route_type=route_type,
            slots=self._apply_route_weights(route_type, slots),
            token_budget=DEFAULT_CONFIG.context_window_tokens,
            metadata={"coverage_score": deduped.persona.coverage_score},
        )

    def _apply_route_weights(self, route_type: RouteType, slots: dict[str, str]) -> dict[str, str]:
        adjusted = dict(slots)
        if route_type == RouteType.E1:
            adjusted["web_persona_context"] = ""
            adjusted["web_reality_context"] = ""
        elif route_type == RouteType.E2:
            adjusted["web_reality_context"] = ""
        elif route_type == RouteType.E2B:
            adjusted["web_persona_context"] = ""
            adjusted["web_reality_context"] = "没有可靠外部资料时，必须保守回答。"
        elif route_type == RouteType.E4:
            adjusted["immutable_core"] = ""
            adjusted["evidence_chunks"] = ""
        elif route_type == RouteType.E5:
            adjusted["web_persona_context"] = ""
            adjusted["web_reality_context"] = ""
        return adjusted

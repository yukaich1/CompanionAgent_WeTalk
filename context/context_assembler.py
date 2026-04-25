from __future__ import annotations

from config import DEFAULT_CONFIG
from context.evidence_adapter import EvidenceAdapter
from context.memory_layers import MemoryLayerBuilder
from knowledge.knowledge_source import AssembledContext, DeduplicatedContext, RouteType


class ContextAssembler:
    def __init__(self) -> None:
        self.evidence_adapter = EvidenceAdapter()
        self.memory_layers = MemoryLayerBuilder()

    def _trim(self, text: str, limit: int) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "..."

    def _join(self, parts: list[str], limit: int) -> str:
        merged = "\n\n".join(part.strip() for part in parts if str(part or "").strip())
        return self._trim(merged, limit)

    def _render_slot(self, title: str, body: str) -> str:
        content = str(body or "").strip()
        if not content:
            return ""
        return f"=== {title} ===\n{content}"

    def _build_slot_values(
        self,
        deduped: DeduplicatedContext,
        thought_output: str,
        web_persona_context: str,
        web_reality_context: str,
    ) -> dict[str, str]:
        budget = DEFAULT_CONFIG.slot_budget
        persona_evidence_items = self.evidence_adapter.adapt_persona(deduped.persona)
        memory_evidence_items = self.evidence_adapter.adapt_memory(deduped.memory)
        persona_identity = next(
            (
                item.content
                for item in persona_evidence_items
                if item.source_kind == "persona_integrated" and str(item.content or "").strip()
            ),
            "",
        )
        persona_chunks = [
            item.content
            for item in persona_evidence_items
            if item.source_kind == "persona_chunk" and str(item.content or "").strip()
        ]
        persona_context = self._trim(persona_identity, budget.evidence_chunks)
        persona_evidence = self._join(persona_chunks, budget.evidence_chunks)
        layers = self.memory_layers.build(memory_evidence_items)
        internal_state = self._trim(thought_output, budget.thought_output)
        persona_web = self._trim(web_persona_context, budget.web_persona_context)
        reality_web = self._trim(web_reality_context, budget.web_reality_context)
        return {
            "layer0_identity": persona_context,
            "persona_evidence": persona_evidence,
            "story_evidence": "",
            "layer1_stable_memory": layers.get("layer1_stable_memory", ""),
            "layer2_topic_memory": layers.get("layer2_topic_memory", ""),
            "layer3_deep_memory": layers.get("layer3_deep_memory", ""),
            "web_persona_context": persona_web,
            "web_reality_context": reality_web,
            "thought_output": internal_state,
            "evidence_chunks": persona_evidence or persona_context,
            "evidence_total_count": str(len(persona_evidence_items) + len(memory_evidence_items)),
        }

    def _apply_route_policy(self, route_type: RouteType, slots: dict[str, str]) -> dict[str, str]:
        adjusted = dict(slots)
        if route_type == RouteType.E1:
            adjusted["web_persona_context"] = ""
            adjusted["web_reality_context"] = ""
            adjusted["story_evidence"] = ""
        elif route_type == RouteType.E2:
            adjusted["web_reality_context"] = ""
        elif route_type == RouteType.E2B:
            adjusted["web_persona_context"] = ""
            adjusted["story_evidence"] = adjusted.get("persona_evidence", "") or adjusted.get("layer0_identity", "")
            adjusted["persona_evidence"] = adjusted["story_evidence"]
            adjusted["evidence_chunks"] = adjusted["story_evidence"]
        elif route_type == RouteType.E4:
            adjusted["layer0_identity"] = ""
            adjusted["persona_evidence"] = ""
            adjusted["story_evidence"] = ""
            adjusted["evidence_chunks"] = ""
        return adjusted

    def _render_prompt_sections(self, slots: dict[str, str]) -> list[str]:
        sections = [
            self._render_slot("L0 Identity", slots.get("layer0_identity", "")),
            self._render_slot(
                "Persona Evidence",
                self._join(
                    [
                        slots.get("persona_evidence", ""),
                        slots.get("web_persona_context", ""),
                    ],
                    DEFAULT_CONFIG.slot_budget.evidence_chunks * 2,
                ),
            ),
            self._render_slot("Story Evidence", slots.get("story_evidence", "")),
            self._render_slot("L1 Stable Memory", slots.get("layer1_stable_memory", "")),
            self._render_slot("L2 Topic Recall", slots.get("layer2_topic_memory", "")),
            self._render_slot("L3 Deep Recall", slots.get("layer3_deep_memory", "")),
            self._render_slot("External Evidence", slots.get("web_reality_context", "")),
            self._render_slot("Internal State", slots.get("thought_output", "")),
        ]
        return [section for section in sections if section]

    def assemble(
        self,
        route_type: RouteType,
        deduped: DeduplicatedContext,
        thought_output: str = "",
        web_persona_context: str = "",
        web_reality_context: str = "",
    ) -> AssembledContext:
        slots = self._build_slot_values(
            deduped=deduped,
            thought_output=thought_output,
            web_persona_context=web_persona_context,
            web_reality_context=web_reality_context,
        )
        slots = self._apply_route_policy(route_type, slots)
        prompt_sections = self._render_prompt_sections(slots)
        metadata = {
            "coverage_score": deduped.persona.coverage_score,
            "slot_presence": {key: bool(value) for key, value in slots.items()},
            "prompt_section_count": len(prompt_sections),
            "memory_layers": {
                "l0": bool(slots.get("layer0_identity")),
                "l1": bool(slots.get("layer1_stable_memory")),
                "l2": bool(slots.get("layer2_topic_memory")),
                "l3": bool(slots.get("layer3_deep_memory")),
            },
            "evidence_total_count": int(str(slots.get("evidence_total_count", "0") or "0")),
        }
        slots["prompt_context"] = "\n\n".join(prompt_sections).strip()
        return AssembledContext(
            route_type=route_type,
            slots=slots,
            token_budget=DEFAULT_CONFIG.context_window_tokens,
            metadata=metadata,
        )

    def build_prompt_context(self, assembled: AssembledContext) -> str:
        prompt_context = str(assembled.slots.get("prompt_context", "") or "").strip()
        if prompt_context:
            return prompt_context
        return "\n\n".join(self._render_prompt_sections(assembled.slots)).strip()

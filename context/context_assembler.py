from __future__ import annotations

from config import DEFAULT_CONFIG
from knowledge.knowledge_source import AssembledContext, DeduplicatedContext, RouteType


class ContextAssembler:
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
        persona_context = self._trim(deduped.persona.integrated_context, budget.evidence_chunks)
        persona_evidence = self._join(list(deduped.persona.evidence_chunks or []), budget.evidence_chunks)
        episodic_memory = self._join(
            [record.content for record in deduped.memory.episodic_records if str(record.content or "").strip()],
            budget.episodic_memory,
        )
        semantic_memory = self._join(
            [record.content for record in deduped.memory.semantic_records if str(record.content or "").strip()],
            budget.semantic_memory,
        )
        relation_state = self._trim(str(deduped.memory.relation_state or ""), budget.relation_state)
        internal_state = self._trim(thought_output, budget.thought_output)
        persona_web = self._trim(web_persona_context, budget.web_persona_context)
        reality_web = self._trim(web_reality_context, budget.web_reality_context)
        return {
            "persona_context": persona_context,
            "persona_evidence": persona_evidence,
            "story_evidence": "",
            "episodic_memory": episodic_memory,
            "semantic_memory": semantic_memory,
            "relation_state": relation_state,
            "web_persona_context": persona_web,
            "web_reality_context": reality_web,
            "thought_output": internal_state,
            "evidence_chunks": persona_evidence or persona_context,
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
            adjusted["story_evidence"] = adjusted.get("persona_evidence", "") or adjusted.get("persona_context", "")
            adjusted["persona_evidence"] = adjusted["story_evidence"]
            adjusted["evidence_chunks"] = adjusted["story_evidence"]
        elif route_type == RouteType.E4:
            adjusted["persona_context"] = ""
            adjusted["persona_evidence"] = ""
            adjusted["story_evidence"] = ""
            adjusted["evidence_chunks"] = ""
        return adjusted

    def _render_prompt_sections(self, slots: dict[str, str]) -> list[str]:
        sections = [
            self._render_slot(
                "Persona Evidence",
                self._join(
                    [
                        slots.get("persona_context", ""),
                        slots.get("persona_evidence", ""),
                        slots.get("web_persona_context", ""),
                    ],
                    DEFAULT_CONFIG.slot_budget.evidence_chunks * 2,
                ),
            ),
            self._render_slot("Story Evidence", slots.get("story_evidence", "")),
            self._render_slot("External Evidence", slots.get("web_reality_context", "")),
            self._render_slot(
                "Memory and Relationship",
                self._join(
                    [
                        slots.get("episodic_memory", ""),
                        slots.get("semantic_memory", ""),
                        slots.get("relation_state", ""),
                    ],
                    DEFAULT_CONFIG.slot_budget.episodic_memory
                    + DEFAULT_CONFIG.slot_budget.semantic_memory
                    + DEFAULT_CONFIG.slot_budget.relation_state,
                ),
            ),
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

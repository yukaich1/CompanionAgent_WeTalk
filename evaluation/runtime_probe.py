from __future__ import annotations

from main import AISystem


class RuntimeProbe:
    def __init__(self, ai_system: AISystem | None = None) -> None:
        self.ai_system = ai_system or AISystem()

    def inspect_turn(self, user_input: str) -> dict:
        system = self.ai_system
        intent_stage = system._extract_intent_stage(user_input)
        persona_stage = system._recall_persona_stage(intent_stage)
        routing_stage = system._route_stage(intent_stage, persona_stage)
        memory_stage = system._recall_memory_stage(intent_stage.normalized_query)
        tool_stage = system._run_tool_stage(intent_stage.intent_result, routing_stage.route_decision)
        context_stage = system._assemble_context_stage(
            route_decision=routing_stage.route_decision,
            persona_recall=persona_stage.persona_recall,
            memory_result=memory_stage.memory_result,
            tool_stage=tool_stage,
        )
        grounding = system._build_grounding_contract(intent_stage.intent_result, persona_stage.persona_recall)
        story_hits = list((getattr(persona_stage.persona_recall, "metadata", {}) or {}).get("story_hits", []) or [])
        persona_context = context_stage.assembled_context.slots.get("evidence_chunks", "")
        tool_context_for_turn = (
            context_stage.assembled_context.slots.get("web_reality_context")
            or context_stage.assembled_context.slots.get("web_persona_context")
            or ""
        )
        context = system.response_generator._build_turn_context(
            user_input=user_input,
            thought_data={},
            response_mode=str(getattr(intent_stage.intent_result, "response_mode", "casual") or "casual"),
            persona_focus=str(getattr(intent_stage.intent_result, "persona_focus", "general") or "general"),
            persona_context=persona_context,
            tool_context=tool_context_for_turn,
            story_hits=story_hits,
            response_contract=str(grounding.get("response_contract", "") or ""),
            persona_focus_contract=str(grounding.get("persona_focus_contract", "") or ""),
            memory_slots=getattr(context_stage.assembled_context, "slots", {}) or {},
        )
        selected_context = system.response_generator.context_selector.select(context)
        plan = system.response_generator.response_planner.build(selected_context)
        return {
            "user_input": user_input,
            "route_type": str(getattr(routing_stage.route_decision, "type", "") or ""),
            "response_mode": str(getattr(intent_stage.intent_result, "response_mode", "") or ""),
            "persona_focus": str(getattr(intent_stage.intent_result, "persona_focus", "") or ""),
            "recall_query": intent_stage.recall_query,
            "selected_context_view": dict((selected_context.metadata or {}).get("selected_context_view", {}) or {}),
            "response_plan": {
                "task_label": plan.task_label,
                "evidence_kind": plan.evidence_kind,
                "evidence_required": plan.evidence_required,
                "evidence_ready": plan.evidence_ready,
                "constraints": list(plan.constraints),
            },
            "tool_policy": {
                "tool_type": str(getattr(tool_stage.tool_report, "tool_type", "") or ""),
                "inject_policy": str(getattr(tool_stage.tool_report, "inject_policy", "") or "none"),
                "persist_policy": str(getattr(tool_stage.tool_report, "persist_policy", "") or "ignore"),
            },
            "persistence_boundary": dict((selected_context.metadata or {}).get("persistence_boundary", {}) or {}),
            "stage_summary": {
                "persona_hit_count": len(list((getattr(persona_stage.persona_recall, "metadata", {}) or {}).get("hits", []) or [])),
                "story_hit_count": len(story_hits),
                "memory_episode_count": len(list(getattr(memory_stage.memory_result, "episode_records", []) or [])),
                "memory_stable_count": len(list(getattr(memory_stage.memory_result, "stable_records", []) or [])),
                "tool_call_count": len(list(getattr(tool_stage.tool_report, "calls", []) or [])),
                "memory_recall_policy": str(getattr(memory_stage, "recall_policy", "") or ""),
            },
        }

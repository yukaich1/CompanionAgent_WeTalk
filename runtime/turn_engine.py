from __future__ import annotations

import re
from datetime import datetime

from ai_runtime_support import derive_relation_impact, derive_topic_tags, estimate_pending_signal, record_debug_info, thought_signal
from reasoning.planner_persona_card import PlannerPersonaCardBuilder
from runtime.runtime_models import TurnArtifacts, TurnInput, TurnObjective
from runtime.turn_trace import TurnTrace
from utils import conversation_to_string


class TurnEngine:
    def __init__(self, system):
        self.system = system
        self.persona_card_builder = PlannerPersonaCardBuilder()

    def run_turn(self, turn_input: TurnInput) -> tuple[str, TurnTrace]:
        start = datetime.now()
        normalized_text = re.sub(r"\s+", " ", str(turn_input.user_text or "").strip())
        trace = TurnTrace(
            turn_id=f"{self.system.session_id}:{self.system.num_messages + 1}",
            session_id=self.system.session_id,
            input_text=str(turn_input.user_text or ""),
            normalized_text=normalized_text,
        )
        artifacts = TurnArtifacts(normalized_text=normalized_text)

        relation_before = self._relation_snapshot()
        emotion_before = self._emotion_snapshot()

        self.system.num_messages += 1
        self.system.buffer.add_message("user", turn_input.user_text)
        artifacts.recent_conversation = conversation_to_string(self.system.get_message_history(False)[-6:])

        self._run_stage_pipeline(artifacts)
        objective = self._build_objective(artifacts)
        trace.route_type = objective.route_type
        trace.response_mode = objective.response_mode
        trace.persona_focus = objective.persona_focus
        trace.recall_query = str(getattr(artifacts.intent_stage, "recall_query", "") or "")
        trace.stage_summary = self._stage_summary(artifacts)

        pending_signal = estimate_pending_signal(
            self.system,
            normalized_text,
            recent_conversation=artifacts.recent_conversation,
        )
        self.system.emotion_state_machine.queue_signal(pending_signal)

        self._run_retrieval_and_tools(artifacts)
        self._run_thought_and_response(artifacts)
        response = artifacts.response_text or "我现在没法把这件事说得太确定。"
        record_debug_info(
            self.system,
            artifacts.route_decision,
            artifacts.assembled_context,
            thought_data=artifacts.thought_data,
            intent_result=artifacts.intent_result,
            persona_recall=artifacts.persona_recall,
        )
        if str(getattr(artifacts.intent_result, "response_mode", "") or "") == "story":
            story_hits = list((getattr(artifacts.persona_recall, "metadata", {}) or {}).get("story_hits", []) or [])
            if story_hits:
                chunk_id = str(story_hits[0].get("chunk_id", "") or "").strip()
                if chunk_id:
                    self.system.recent_story_chunk_ids.append(chunk_id)

        self._commit_turn(
            user_input=normalized_text,
            response=response,
            artifacts=artifacts,
            pending_signal=pending_signal,
        )

        relation_after = self._relation_snapshot()
        emotion_after = self._emotion_snapshot()
        self.system.buffer.add_message("assistant", response)
        self.system.last_message = datetime.now()
        self.system.tick()

        working_snapshot = self.system.memory_system.get_working_memory_snapshot()
        self.system.session_context_manager.update_after_turn(
            active_topics=list(working_snapshot.get("active_topics", []) or []),
            active_thread_summaries=[
                str(thread.get("summary", "") or "").strip()
                for thread in list(working_snapshot.get("active_threads", []) or [])
                if str(thread.get("summary", "") or "").strip()
            ],
            relation_summary=str(working_snapshot.get("recent_relation_summary", "") or ""),
            emotion_summary=str(working_snapshot.get("recent_emotion_summary", "") or ""),
            pinned_facts=list(working_snapshot.get("pinned_facts", []) or []),
        )

        evidence_gate = dict((self.system.last_debug_info or {}).get("evidenceGate", {}) or {})
        response_plan = dict((self.system.last_debug_info or {}).get("responsePlan", {}) or {})
        trace.selected_evidence_sources = list(response_plan.get("selectedEvidenceSources", []) or [])
        trace.selected_memory_layers = list(response_plan.get("memoryLayers", []) or [])
        trace.evidence_gate = evidence_gate
        trace.tool_calls = self._tool_calls(artifacts)
        trace.tool_policy = self._tool_policy(artifacts.tool_report)
        trace.emotion_before = emotion_before
        trace.emotion_after = emotion_after
        trace.relation_before = relation_before
        trace.relation_after = relation_after
        trace.selected_context_view = dict((self.system.last_debug_info or {}).get("selectedContextView", {}) or {})
        trace.persistence_boundary = dict((self.system.last_debug_info or {}).get("persistenceBoundary", {}) or {})
        trace.memory_commit = dict((self.system.last_debug_info or {}).get("memoryCommit", {}) or {})
        trace.planner = {
            "surface_intent": str((artifacts.thought_data or {}).get("surface_intent", "") or ""),
            "latent_need": str((artifacts.thought_data or {}).get("latent_need", "") or ""),
            "response_goal": str((artifacts.thought_data or {}).get("response_goal", "") or ""),
            "reasoning_summary": str((artifacts.thought_data or {}).get("reasoning_summary", "") or ""),
        }
        trace.session_context = dict(self.system.session_context_manager.build_session_context() or {})
        trace.working_memory = dict(working_snapshot or {})
        trace.final_response = response
        trace.latency_ms = int((datetime.now() - start).total_seconds() * 1000)
        if evidence_gate.get("required") and not evidence_gate.get("ready"):
            trace.fallback_reason = str(evidence_gate.get("reason", "missing_evidence"))
        self.system.last_turn_trace = trace
        self.system.turn_traces.append(trace)
        return response, trace

    def _run_stage_pipeline(self, artifacts: TurnArtifacts) -> None:
        intent_stage = self.system._extract_intent_stage(artifacts.normalized_text)
        persona_stage = self.system._recall_persona_stage(intent_stage)
        routing_stage = self.system._route_stage(intent_stage, persona_stage)
        memory_stage = self.system._recall_memory_stage(intent_stage.normalized_query)
        tool_stage = self.system._run_tool_stage(intent_stage.intent_result, routing_stage.route_decision)
        context_stage = self.system._assemble_context_stage(
            route_decision=routing_stage.route_decision,
            persona_recall=persona_stage.persona_recall,
            memory_result=memory_stage.memory_result,
            tool_stage=tool_stage,
        )

        artifacts.intent_stage = intent_stage
        artifacts.persona_stage = persona_stage
        artifacts.routing_stage = routing_stage
        artifacts.memory_stage = memory_stage
        artifacts.tool_stage = tool_stage
        artifacts.context_stage = context_stage
        artifacts.route_decision = routing_stage.route_decision
        artifacts.intent_result = intent_stage.intent_result
        artifacts.persona_recall = persona_stage.persona_recall
        artifacts.memory_result = memory_stage.memory_result
        artifacts.tool_report = tool_stage.tool_report
        artifacts.assembled_context = context_stage.assembled_context

    def _build_objective(self, artifacts: TurnArtifacts) -> TurnObjective:
        route_decision = artifacts.route_decision
        intent_result = artifacts.intent_result
        tool_report = artifacts.tool_report
        response_mode = str(getattr(intent_result, "response_mode", "casual") or "casual").strip() or "casual"
        return TurnObjective(
            route_type=str(getattr(route_decision, "type", "") or ""),
            response_mode=response_mode,
            persona_focus=str(getattr(intent_result, "persona_focus", "general") or "general").strip() or "general",
            requires_tool=bool(list(getattr(tool_report, "calls", []) or []) or getattr(tool_report, "follow_up_message", "")),
            requires_persona=response_mode in {"self_intro", "persona_fact", "story", "value", "casual", "emotional"},
            requires_memory=response_mode in {"casual", "value", "emotional"},
            requires_story=response_mode == "story",
            requires_external=response_mode == "external",
        )

    def _stage_summary(self, artifacts: TurnArtifacts) -> dict:
        assembled_context = artifacts.assembled_context
        slot_presence = dict((getattr(assembled_context, "metadata", {}) or {}).get("slot_presence", {}) or {})
        persona_hits = list((getattr(getattr(artifacts, "persona_recall", None), "metadata", {}) or {}).get("hits", []) or [])
        story_hits = list((getattr(getattr(artifacts, "persona_recall", None), "metadata", {}) or {}).get("story_hits", []) or [])
        memory_result = getattr(artifacts, "memory_result", None)
        return {
            "intent_recall_mode": str(getattr(getattr(artifacts, "intent_result", None), "recall_mode", "") or ""),
            "intent_extracted_topic": str(getattr(getattr(artifacts, "intent_result", None), "extracted_topic", "") or ""),
            "persona_query_type": str(getattr(getattr(artifacts, "persona_stage", None), "preferred_query_type", "") or ""),
            "persona_hit_count": len(persona_hits),
            "story_hit_count": len(story_hits),
            "memory_recall_policy": str(getattr(getattr(artifacts, "memory_stage", None), "recall_policy", "") or ""),
            "memory_episode_count": len(list(getattr(memory_result, "episode_records", []) or [])),
            "memory_stable_count": len(list(getattr(memory_result, "stable_records", []) or [])),
            "tool_call_count": len(list(getattr(getattr(artifacts, "tool_report", None), "calls", []) or [])),
            "has_follow_up_message": bool(getattr(getattr(artifacts, "tool_report", None), "follow_up_message", "")),
            "context_slots": slot_presence,
        }

    def _run_retrieval_and_tools(self, artifacts: TurnArtifacts) -> None:
        if artifacts.tool_report and getattr(artifacts.tool_report, "follow_up_message", ""):
            artifacts.response_text = str(artifacts.tool_report.follow_up_message or "")

    def _run_thought_and_response(self, artifacts: TurnArtifacts) -> None:
        if artifacts.response_text:
            return
        assembled_context = artifacts.assembled_context
        intent_result = artifacts.intent_result
        persona_recall = artifacts.persona_recall
        tool_context_for_turn = (
            assembled_context.slots.get("web_reality_context")
            or assembled_context.slots.get("web_persona_context")
            or ""
        )
        persona_context = assembled_context.slots.get("evidence_chunks", "")
        thought_persona_context = self._planner_evidence_preview(assembled_context)
        grounding = self.system._build_grounding_contract(intent_result, persona_recall)
        artifacts.grounding = grounding

        memories, recalled_memories = self.system.memory_system.recall_memories(self.system.get_message_history(False))
        thought_data = self.system.thought_system.think(
            messages=self.system.get_message_history(False),
            memories=memories,
            recalled_memories=recalled_memories,
            last_message=self.system.last_message,
            persona_context=thought_persona_context,
            recent_conversation=artifacts.recent_conversation,
            session_context=str(self.system.session_context_manager.render() or ""),
            working_memory=self.system.memory_system.get_working_memory_snapshot(),
            relation_state=self._relation_snapshot(),
            intent_snapshot={
                "intent": str(getattr(intent_result, "intent", "") or ""),
                "response_mode": str(getattr(intent_result, "response_mode", "") or ""),
                "recall_mode": str(getattr(intent_result, "recall_mode", "") or ""),
                "persona_focus": str(getattr(intent_result, "persona_focus", "") or ""),
            },
            route_snapshot={
                "route_type": str(getattr(artifacts.route_decision, "type", "") or ""),
                "web_search_mode": str(getattr(artifacts.route_decision, "web_search_mode", "") or ""),
                "info_domain": str(getattr(artifacts.route_decision, "info_domain", "") or ""),
            },
            tool_snapshot=self._tool_policy(artifacts.tool_report),
            persona_decision_card=self.persona_card_builder.build(self.system),
        )
        self.system.emotion_state_machine.update_from_thought(thought_signal(thought_data))
        artifacts.thought_data = thought_data
        response_mode = str(getattr(intent_result, "response_mode", "casual") or "casual").strip()
        persona_focus = str(getattr(intent_result, "persona_focus", "general") or "general").strip()
        story_hits = list((getattr(persona_recall, "metadata", {}) or {}).get("story_hits", []) or [])
        evidence_status = self.system._evaluate_evidence_status(
            user_input=artifacts.normalized_text,
            intent_result=intent_result,
            grounding=grounding,
            tool_context=tool_context_for_turn,
            persona_context=persona_context,
            assembled_context=assembled_context,
            story_hits=story_hits,
        )
        self.system.last_debug_info = {
            **(self.system.last_debug_info or {}),
            "evidenceGate": evidence_status,
        }
        if evidence_status.get("required") and not evidence_status.get("ready"):
            self.system.last_debug_info = {
                **(self.system.last_debug_info or {}),
                "responsePlan": {
                    "mode": response_mode,
                    "personaFocus": persona_focus,
                    "task": str(grounding.get("response_contract", "") or ""),
                    "evidenceKind": evidence_status.get("reason", ""),
                    "evidenceRequired": True,
                    "evidenceReady": False,
                    "evidenceSources": [],
                    "selectedEvidenceSources": [],
                    "evidencePreview": "",
                    "memoryLayers": [],
                },
            }
            artifacts.response_text = self.system.response_generator.missing_evidence_reply(
                response_mode=response_mode,
                reason=str(evidence_status.get("reason", "")),
            )
            return
        mode_override = str(thought_data.get("response_mode_override", "") or "").strip()
        if mode_override in {"self_intro", "story", "persona_fact", "external", "emotional", "value", "casual"}:
            response_mode = mode_override
        focus_override = str(thought_data.get("persona_focus_override", "") or "").strip()
        if focus_override in {"general", "likes", "dislikes", "catchphrase", "personality", "self_intro"}:
            persona_focus = focus_override
        artifacts.response_text = self.system.response_generator.reply(
            user_input=artifacts.normalized_text,
            thought_data=thought_data,
            response_mode=response_mode,
            persona_focus=persona_focus,
            persona_context=persona_context,
            tool_context=tool_context_for_turn,
            story_hits=story_hits,
            response_contract=str(grounding.get("response_contract", "") or ""),
            persona_focus_contract=str(grounding.get("persona_focus_contract", "") or ""),
            memory_slots=getattr(assembled_context, "slots", {}) or {},
        )

    def _planner_evidence_preview(self, assembled_context) -> str:
        slots = getattr(assembled_context, "slots", {}) or {}
        identity = str(slots.get("l0_identity", "") or "").strip()
        evidence = str(slots.get("evidence_chunks", "") or "").strip()
        story = str(slots.get("story_chunks", "") or "").strip()
        parts: list[str] = []
        if identity and identity != "None":
            parts.append(f"[identity]\n{identity[:420]}")
        if evidence and evidence != "None":
            parts.append(f"[persona_evidence]\n{evidence[:520]}")
        if story and story != "None":
            parts.append(f"[story_evidence]\n{story[:520]}")
        return "\n\n".join(parts[:3])

    def _commit_turn(self, *, user_input: str, response: str, artifacts: TurnArtifacts, pending_signal) -> None:
        intent_result = artifacts.intent_result
        persona_recall = artifacts.persona_recall
        thought_data = artifacts.thought_data
        tool_policy = self._tool_policy(artifacts.tool_report)
        memory_capture = self.system._memory_capture_for_turn(
            user_input=user_input,
            response=str(response or ""),
            intent_result=intent_result,
            persona_recall=persona_recall,
        )
        relation_summary = self.system.memory_system.relation_summary() or self._relation_summary()
        emotion_summary = str(thought_data.get("emotion_reason", "") or thought_data.get("emotion", "") or "").strip()
        tool_summary = self._tool_summary(artifacts.tool_report)
        should_persist_memory = tool_policy.get("persist_policy", "ignore") != "session_only"
        if should_persist_memory:
            self.system.memory_writer.remember(
                self.system.new_memory_state,
                summary=memory_capture["summary"],
                user_text=memory_capture["user_text"],
                assistant_text=memory_capture["assistant_text"],
                verbatim_excerpt=memory_capture["verbatim_excerpt"],
                topic_tags=derive_topic_tags(user_input),
                relation_impact=derive_relation_impact(pending_signal),
                importance=0.55,
                character_emotion=str(thought_data.get("emotion", "平静")),
                scope="",
                source_session_id=self.system.session_id,
                source_turn_index=self.system.num_messages,
            )
            self.system.memory_system.set_state(self.system.new_memory_state)
        self.system.memory_system.update_working_memory(
            turn_index=self.system.num_messages,
            topic_hint=str(getattr(intent_result, "extracted_topic", "") or "").strip(),
            summary=memory_capture["summary"],
            relation_summary=relation_summary,
            emotion_summary=emotion_summary,
            tool_summary=tool_summary,
            pinned_facts=list((self.system.last_debug_info or {}).get("responsePlan", {}).get("selectedEvidenceSources", []) or []),
        )
        if self.system.num_messages % 6 == 0:
            self.system.memory_system.consolidate_memories()
        self.system._sync_persona_state()
        self.system.health_monitor.record_turn_metrics(
            coverage_score=artifacts.assembled_context.metadata.get("coverage_score", 0.0),
            evidence_backed=bool(((self.system.last_debug_info or {}).get("evidenceGate", {}) or {}).get("ready", False)),
        )
        self.system.last_debug_info = {
            **(self.system.last_debug_info or {}),
            "memoryCommit": {
                "persisted_long_term": should_persist_memory,
                "tool_policy": tool_policy,
                "working_memory_updated": True,
            },
        }

    def _tool_calls(self, artifacts: TurnArtifacts) -> list[dict]:
        tool_report = artifacts.tool_report
        if tool_report is None:
            return []
        calls = list(getattr(tool_report, "calls", []) or [])
        if not calls and not getattr(tool_report, "follow_up_message", ""):
            return []
        rendered = [
            {
                "tool_name": str(getattr(call, "name", "") or "tool_runtime"),
                "reason": str(getattr(call, "reason", "") or "").strip(),
            }
            for call in calls
        ]
        if getattr(tool_report, "follow_up_message", ""):
            rendered.append(
                {
                    "tool_name": "follow_up_message",
                    "reason": str(getattr(tool_report, "follow_up_message", "") or "").strip(),
                }
            )
        return rendered

    def _tool_summary(self, tool_report) -> str:
        if tool_report is None:
            return ""
        calls = list(getattr(tool_report, "calls", []) or [])
        tool_name = ", ".join(str(getattr(call, "name", "") or "").strip() for call in calls if str(getattr(call, "name", "") or "").strip())
        follow_up = str(getattr(tool_report, "follow_up_message", "") or "").strip()
        if tool_name and follow_up:
            return f"{tool_name}: {follow_up}"
        return tool_name or follow_up

    def _tool_policy(self, tool_report) -> dict:
        if tool_report is None:
            return {}
        return {
            "tool_type": str(getattr(tool_report, "tool_type", "") or ""),
            "inject_policy": str(getattr(tool_report, "inject_policy", "none") or "none"),
            "persist_policy": str(getattr(tool_report, "persist_policy", "ignore") or "ignore"),
        }

    def _relation_snapshot(self) -> dict:
        state = getattr(self.system, "new_memory_state", None)
        relation = getattr(state, "relation_state", None)
        if relation is None:
            return {}
        if hasattr(relation, "model_dump"):
            return relation.model_dump()
        if hasattr(relation, "dict"):
            return relation.dict()
        return {}

    def _emotion_snapshot(self) -> dict:
        machine = getattr(self.system, "emotion_state_machine", None)
        state = getattr(machine, "current_state", None)
        if state is None:
            return {}
        if hasattr(state, "model_dump"):
            return state.model_dump()
        if hasattr(state, "dict"):
            return state.dict()
        return {}

    def _relation_summary(self) -> str:
        relation = self._relation_snapshot()
        if not relation:
            return ""
        stage = str(relation.get("stage", "") or "").strip()
        trust = relation.get("trust")
        affection = relation.get("affection")
        familiarity = relation.get("familiarity")
        return (
            f"stage={stage or 'unknown'}, trust={trust:.2f}, affection={affection:.2f}, familiarity={familiarity:.2f}"
            if all(isinstance(item, (int, float)) for item in (trust, affection, familiarity))
            else stage
        )

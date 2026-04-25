from __future__ import annotations

from dataclasses import replace

from context.context_models import SelectedContextView, SessionContextView, StableContextView, TurnContextView
from turn_runtime import DynamicStateView, EvidenceBundle, TurnRuntimeContext


class ContextSelector:
    def select(self, context: TurnRuntimeContext) -> TurnRuntimeContext:
        mode = str(context.response_mode or "casual").strip() or "casual"
        filtered_evidence = self._select_evidence(mode, context.evidence)
        filtered_dynamic = self._select_dynamic_state(mode, context.dynamic_state)
        selected_view = self._build_selected_view(
            replace(context, evidence=filtered_evidence, dynamic_state=filtered_dynamic)
        )
        metadata = dict(context.metadata or {})
        metadata["selected_context_view"] = selected_view.as_dict()
        return replace(
            context,
            evidence=filtered_evidence,
            dynamic_state=filtered_dynamic,
            metadata=metadata,
        )

    def _select_evidence(self, mode: str, evidence: EvidenceBundle) -> EvidenceBundle:
        if mode == "self_intro":
            return EvidenceBundle(
                l0_identity=evidence.l0_identity,
                persona=evidence.persona,
            )
        if mode == "story":
            return EvidenceBundle(
                l0_identity=evidence.l0_identity,
                story=evidence.story,
            )
        if mode == "persona_fact":
            return EvidenceBundle(
                l0_identity=evidence.l0_identity,
                persona=evidence.persona,
            )
        if mode == "external":
            return EvidenceBundle(
                external=evidence.external,
            )
        if mode == "emotional":
            return EvidenceBundle(
                l0_identity=evidence.l0_identity,
                persona=evidence.persona,
            )
        if mode == "value":
            return EvidenceBundle(
                l0_identity=evidence.l0_identity,
                persona=evidence.persona,
            )
        return evidence

    def _select_dynamic_state(self, mode: str, dynamic_state: DynamicStateView) -> DynamicStateView:
        if mode == "external":
            return replace(
                dynamic_state,
                memory_snapshot="None",
            )
        if mode == "self_intro":
            return replace(
                dynamic_state,
                recent_dialogue="None",
            )
        if mode == "story":
            return replace(
                dynamic_state,
                recent_dialogue="None",
            )
        return dynamic_state

    def _build_selected_view(self, context: TurnRuntimeContext) -> SelectedContextView:
        session_state = dict((context.metadata or {}).get("session_context_state", {}) or {})
        topics = [str(item or "").strip() for item in list(session_state.get("active_topics", []) or []) if str(item or "").strip()]
        threads = [
            str(item or "").strip()
            for item in list(session_state.get("active_thread_summaries", []) or [])
            if str(item or "").strip()
        ]
        archived_threads = [
            str(item or "").strip()
            for item in list(session_state.get("archived_thread_summaries", []) or [])
            if str(item or "").strip()
        ]
        pinned = [
            str(item or "").strip()
            for item in list(session_state.get("pinned_facts", []) or [])
            if str(item or "").strip()
        ]
        relation_summary = str(session_state.get("relation_summary", "") or "").strip() or context.dynamic_state.relation_state
        emotion_summary = str(session_state.get("emotion_summary", "") or "").strip() or context.dynamic_state.user_emotion_hint
        evidence_preview = next(
            (
                item
                for item in [
                    context.evidence.external,
                    context.evidence.story,
                    context.evidence.persona,
                    context.evidence.l0_identity,
                ]
                if str(item or "").strip()
            ),
            "",
        )
        return SelectedContextView(
            stable=StableContextView(
                identity=str(context.evidence.l0_identity or "").strip(),
                style=str(context.style_prompt or "").strip(),
                rules=[
                    str(context.response_contract or "").strip(),
                    str(context.persona_focus_contract or "").strip(),
                ],
            ),
            session=SessionContextView(
                topics=topics,
                threads=threads,
                archived_threads=archived_threads,
                relation_summary=str(relation_summary or "").strip(),
                emotion_summary=str(emotion_summary or "").strip(),
                pinned_facts=pinned,
            ),
            turn=TurnContextView(
                user_input=str(context.user_input or "").strip(),
                response_mode=str(context.response_mode or "").strip(),
                evidence_sources=context.evidence.active_sources(),
                evidence_preview=str(evidence_preview or "").strip(),
                recent_dialogue=str(context.dynamic_state.recent_dialogue or "").strip(),
            ),
        )

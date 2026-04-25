from __future__ import annotations

from typing import Mapping

from ai_runtime_support import build_identity_reference
from turn_runtime import EvidenceBundle


class EvidencePackBuilder:
    def __init__(self, system) -> None:
        self.system = system

    def build(
        self,
        *,
        response_mode: str,
        user_input: str,
        persona_context: str = "",
        tool_context: str = "",
        story_hits: list[dict] | None = None,
        memory_slots: Mapping[str, object] | None = None,
    ) -> tuple[EvidenceBundle, list[str]]:
        del user_input
        normalized_mode = str(response_mode or "casual").strip() or "casual"
        l0_identity_prompt = str(build_identity_reference(self.system) or "").strip()
        story_text = self._select_story_text(story_hits)
        evidence = EvidenceBundle(
            l0_identity=l0_identity_prompt,
            persona=str(persona_context or "").strip(),
            story=story_text,
            external=str(tool_context or "").strip(),
        )
        return evidence, self._selected_memory_layers(normalized_mode, memory_slots)

    def _select_story_text(self, story_hits: list[dict] | None) -> str:
        for hit in list(story_hits or []):
            if not isinstance(hit, dict):
                continue
            content = str(hit.get("content", "") or hit.get("text", "") or "").strip()
            if content:
                return content
        return ""

    def _selected_memory_layers(self, response_mode: str, memory_slots: Mapping[str, object] | None = None) -> list[str]:
        slots = memory_slots if isinstance(memory_slots, Mapping) else {}
        layer_values = {
            "L1 Stable Memory": str(slots.get("layer1_stable_memory", "") or "").strip(),
            "L2 Topic Recall": str(slots.get("layer2_topic_memory", "") or "").strip(),
            "L3 Deep Recall": str(slots.get("layer3_deep_memory", "") or "").strip(),
        }
        selected_by_mode = {
            "self_intro": ("L1 Stable Memory",),
            "casual": ("L1 Stable Memory", "L2 Topic Recall"),
            "persona_fact": ("L1 Stable Memory",),
            "story": ("L1 Stable Memory", "L3 Deep Recall"),
            "external": ("L1 Stable Memory",),
            "emotional": ("L1 Stable Memory", "L2 Topic Recall"),
            "value": ("L1 Stable Memory", "L2 Topic Recall"),
        }
        layer_order = selected_by_mode.get(response_mode, ("L1 Stable Memory",))
        return [label for label in layer_order if layer_values.get(label)]

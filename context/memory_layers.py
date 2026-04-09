from __future__ import annotations

from config import DEFAULT_CONFIG
from knowledge.knowledge_source import UnifiedEvidenceItem


class MemoryLayerBuilder:
    def _trim(self, text: str, limit: int) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "..."

    def _join(self, parts: list[str], limit: int) -> str:
        merged = "\n\n".join(part.strip() for part in parts if str(part or "").strip())
        return self._trim(merged, limit)

    def build(self, evidence_items: list[UnifiedEvidenceItem]) -> dict[str, str]:
        budget = DEFAULT_CONFIG.slot_budget
        stable_items = [
            item.content
            for item in evidence_items
            if item.source_kind == "memory_stable" and str(item.content or "").strip()
        ]
        episodic_items = [
            item.content
            for item in evidence_items
            if item.source_kind == "memory_episode" and str(item.content or "").strip()
        ]
        relation_state = self._trim(
            next(
                (
                    item.content
                    for item in evidence_items
                    if item.source_kind == "relation_state" and str(item.content or "").strip()
                ),
                "",
            ),
            budget.relation_state,
        )
        topic_items = episodic_items[:2]
        deep_items = episodic_items[2:]

        layer1_parts = [*stable_items]
        if relation_state:
            layer1_parts.append(relation_state)

        return {
            "layer1_stable_memory": self._join(
                layer1_parts,
                budget.layer1_memory + budget.relation_state,
            ),
            "layer2_topic_memory": self._join(topic_items, budget.layer2_memory),
            "layer3_deep_memory": self._join(deep_items, budget.layer3_memory),
        }

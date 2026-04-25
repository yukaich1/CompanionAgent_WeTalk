from __future__ import annotations

import re

from config import DEFAULT_CONFIG
from knowledge.knowledge_source import UnifiedEvidenceItem


class MemoryLayerBuilder:
    def _normalize_key(self, text: str) -> str:
        value = re.sub(r"\s+", "", str(text or "")).strip().lower()
        return value[:160]

    def _trim(self, text: str, limit: int) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "..."

    def _join(self, parts: list[str], limit: int) -> str:
        merged = "\n\n".join(part.strip() for part in parts if str(part or "").strip())
        return self._trim(merged, limit)

    def _dedupe(self, parts: list[str], seen: set[str] | None = None) -> list[str]:
        seen_keys = seen if seen is not None else set()
        cleaned: list[str] = []
        for part in parts:
            value = str(part or "").strip()
            if not value:
                continue
            key = self._normalize_key(value)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            cleaned.append(value)
        return cleaned

    def build(self, evidence_items: list[UnifiedEvidenceItem]) -> dict[str, str]:
        budget = DEFAULT_CONFIG.slot_budget
        stable_items = self._dedupe([
            item.content
            for item in evidence_items
            if item.source_kind == "memory_stable" and str(item.content or "").strip()
        ])
        episodic_items = self._dedupe([
            item.content
            for item in evidence_items
            if item.source_kind == "memory_episode" and str(item.content or "").strip()
        ])
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
        layer1_seen = {self._normalize_key(item) for item in stable_items}
        if relation_state:
            layer1_seen.add(self._normalize_key(relation_state))

        topic_items = self._dedupe(episodic_items[:2], seen=set(layer1_seen))
        deep_items = self._dedupe(episodic_items[2:], seen=set(layer1_seen).union(self._normalize_key(item) for item in topic_items))

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

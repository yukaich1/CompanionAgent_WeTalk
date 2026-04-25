from __future__ import annotations

from memory.state_models import RelationState


class RelationStateManager:
    def render_summary(self, relation_state: RelationState) -> str:
        stage = str(getattr(relation_state, "stage", "") or "").strip() or "unknown"
        trust = float(getattr(relation_state, "trust", 0.0) or 0.0)
        affection = float(getattr(relation_state, "affection", 0.0) or 0.0)
        familiarity = float(getattr(relation_state, "familiarity", 0.0) or 0.0)
        last_event = str(getattr(relation_state, "last_significant_event", "") or "").strip()
        parts = [
            f"stage={stage}",
            f"trust={trust:.2f}",
            f"affection={affection:.2f}",
            f"familiarity={familiarity:.2f}",
        ]
        if last_event:
            parts.append(f"event={last_event}")
        return ", ".join(parts)

    def render_profile(self, relation_state: RelationState) -> dict[str, object]:
        trust = float(getattr(relation_state, "trust", 0.0) or 0.0)
        affection = float(getattr(relation_state, "affection", 0.0) or 0.0)
        familiarity = float(getattr(relation_state, "familiarity", 0.0) or 0.0)
        average = (trust + affection + familiarity) / 3.0
        if average >= 0.75:
            boundary = "允许更自然的亲近、回忆和轻度私密表达。"
        elif average >= 0.45:
            boundary = "可以自然靠近，但仍应避免默认过度亲密或越界代入。"
        else:
            boundary = "互动边界较远，避免默认进入私密或过分亲昵的表达。"
        return {
            "stage": str(getattr(relation_state, "stage", "") or "unknown").strip() or "unknown",
            "trust": round(trust, 3),
            "affection": round(affection, 3),
            "familiarity": round(familiarity, 3),
            "guidance": boundary,
            "last_significant_event": str(getattr(relation_state, "last_significant_event", "") or "").strip(),
        }

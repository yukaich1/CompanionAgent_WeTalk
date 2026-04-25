from __future__ import annotations

from typing import Any


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _label_from_score(value: float, *, low: float = 0.38, high: float = 0.66) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "medium"


def _style_label(warmth: float, directness: float) -> str:
    warmth_label = _label_from_score(warmth)
    direct_label = _label_from_score(directness)
    if warmth_label == "high" and direct_label == "high":
        return "warm-direct"
    if warmth_label == "high" and direct_label == "low":
        return "warm-soft"
    if warmth_label == "low" and direct_label == "high":
        return "cool-direct"
    if warmth_label == "low" and direct_label == "low":
        return "cool-soft"
    return "balanced"


class PlannerPersonaCardBuilder:
    def build(self, system) -> dict[str, Any]:
        persona = system.persona_system
        personality = system.personality_system
        base_template = getattr(persona, "base_template", {}) or {}
        background = base_template.get("00_BACKGROUND", {}) if isinstance(base_template, dict) else {}
        profile = background.get("profile", {}) if isinstance(background, dict) else {}

        identity_bits: list[str] = []
        for key in ("full_name", "name", "species", "archetype", "occupation", "role"):
            value = str(profile.get(key, "") or "").strip() if isinstance(profile, dict) else ""
            if value:
                identity_bits.append(f"{key}={value}")

        core_traits = [
            str(item or "").strip()
            for item in list(getattr(persona, "display_keywords", []) or [])
            if str(item or "").strip()
        ][:8]

        voice_card = str(getattr(persona, "character_voice_card", "") or "").strip()

        warmth_baseline = _clamp(0.45 + personality.agreeable * 0.28 + personality.extrovert * 0.12, 0.05, 0.95)
        directness_baseline = _clamp(0.5 + personality.conscientious * 0.18 - personality.agreeable * 0.1, 0.05, 0.95)
        emotional_openness = _clamp(0.42 + personality.extrovert * 0.15 + personality.open * 0.16, 0.05, 0.95)
        intimacy_baseline = _clamp(0.34 + personality.agreeable * 0.18 + personality.extrovert * 0.08, 0.05, 0.9)
        humor_tendency = _clamp(0.28 + personality.extrovert * 0.12 + personality.open * 0.08, 0.05, 0.9)
        restraint = _clamp(0.56 + personality.conscientious * 0.12 - personality.extrovert * 0.08, 0.05, 0.95)

        speech_style_hint = ""
        if voice_card:
            lowered = voice_card.lower()
            fragments: list[str] = []
            if any(token in lowered for token in ["温柔", "柔和", "轻声", "体贴"]):
                fragments.append("warm")
            if any(token in lowered for token in ["冷", "克制", "疏离", "淡"]):
                fragments.append("restrained")
            if any(token in lowered for token in ["直接", "干脆", "利落"]):
                fragments.append("direct")
            if any(token in lowered for token in ["玩笑", "调侃", "轻松", "俏皮"]):
                fragments.append("light-humor")
            speech_style_hint = ",".join(fragments[:4])

        return {
            "identity_core": "；".join(identity_bits[:3]) or system.config.name,
            "core_traits": core_traits,
            "speech_style_hint": speech_style_hint,
            "interaction_style": {
                "warmth_baseline": round(warmth_baseline, 3),
                "directness_baseline": round(directness_baseline, 3),
                "emotional_openness": round(emotional_openness, 3),
                "default_intimacy_ceiling": round(intimacy_baseline, 3),
                "humor_tendency": round(humor_tendency, 3),
                "restraint": round(restraint, 3),
                "style_label": _style_label(warmth_baseline, directness_baseline),
            },
            "decision_bias": {
                "comfort_first": "high" if warmth_baseline >= 0.62 else "medium",
                "clarity_first": "high" if directness_baseline >= 0.62 else "medium",
                "boundary_strictness": _label_from_score(restraint),
                "intimacy_progression": _label_from_score(intimacy_baseline, low=0.32, high=0.58),
            },
            "truthfulness_policy": "证据不足时宁可收住，也不要补写新设定或新经历。",
            "story_policy": "只有命中真实 story evidence 时才能讲故事，否则必须明确收束。",
            "memory_policy": "优先接住 working/session memory，不要把稳定人设误当作近期对话记忆。",
            "planner_note": "这张卡只描述角色如何做决策，不提供故事正文，也不替代当前证据。",
        }

from __future__ import annotations

import re
from datetime import datetime

from knowledge.knowledge_source import SearchMode
from persona_prompting import build_base_template_injection_prompt
from reasoning.emotion_state_machine import EmotionSignal
from utils import format_date, format_memories_to_string, format_time, time_since_last_message_string


def current_affinity_level(system) -> str:
    relation_state = getattr(getattr(system, "new_memory_state", None), "relation_state", None)
    if relation_state is not None:
        try:
            score = max(float(relation_state.affection), float(relation_state.familiarity))
        except Exception:
            score = 0.0
    else:
        friendliness = float(getattr(system.relation_system, "friendliness", 0.0))
        score = max(0.0, min(1.0, (friendliness + 100.0) / 200.0))

    if score >= 0.72:
        return "close"
    if score >= 0.35:
        return "familiar"
    return "stranger"


def build_persona_injection_prompt(system, thought_data: dict) -> str:
    persona = system.persona_system
    base_template = getattr(persona, "base_template", {}) or {}
    avoid_patterns = base_template.get("19_AVOID_PATTERNS", {}).get("patterns", [])
    current_emotion = str(thought_data.get("emotion", "平静") or "平静").strip() or "平静"
    selected_keywords = list(
        getattr(persona, "selected_keywords", []) or getattr(persona, "display_keywords", []) or []
    )
    return build_base_template_injection_prompt(
        character_name=system.config.name,
        character_voice_card=str(getattr(persona, "character_voice_card", "") or "").strip(),
        high_priority_rules=base_template,
        style_examples=list(getattr(persona, "style_examples", []) or []),
        avoid_patterns=avoid_patterns if isinstance(avoid_patterns, list) else [],
        current_affinity_level=current_affinity_level(system),
        current_emotion=current_emotion,
        selected_keywords=selected_keywords,
    )


def recent_assistant_context(system) -> str:
    recent_messages = system._recent_assistant_messages(limit=3)
    if not recent_messages:
        return "None"

    summaries: list[str] = []
    for text in recent_messages:
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        compact = " / ".join(lines[:3]) if lines else str(text).strip()
        summaries.append("- " + system._truncate_for_prompt(compact, 140))
    return system._truncate_for_prompt("\n".join(summaries), 420)


def build_format_data(system, content, thought_data, memories, persona_context, tool_context) -> dict:
    now = datetime.now()
    beliefs = system.get_beliefs()
    belief_str = "\n".join(f"- {belief}" for belief in beliefs) if beliefs else "None"
    memories_str = format_memories_to_string(memories, "You do not have any stable memory about this user yet.")
    memories_str = system._truncate_for_prompt(memories_str, 500)
    injection_prompt = build_persona_injection_prompt(system, thought_data)
    combined_persona_context = "\n\n".join(part for part in (injection_prompt, persona_context or "") if part)

    user_emotions = thought_data.get("possible_user_emotions", []) or []
    if user_emotions:
        user_emotion_str = "The user appears to be feeling: " + ", ".join(user_emotions)
    else:
        user_emotion_str = "The user does not appear to show a strong explicit emotion."

    return {
        "name": system.config.name,
        "personality_summary": system.personality_system.get_summary(),
        "persona_context": combined_persona_context,
        "tool_context": tool_context,
        "recent_assistant_context": recent_assistant_context(system),
        "persona_grounding_required": "yes" if system._requires_persona_grounding(content) else "no",
        "external_grounding_required": "yes" if system._requires_external_grounding(content) else "no",
        "tool_evidence_available": "yes" if system._has_tool_evidence(tool_context) else "no",
        "user_input": content,
        "emotion": thought_data.get("emotion", "平静"),
        "emotion_reason": thought_data.get("emotion_reason", "I feel this way based on the current conversation."),
        "memories": memories_str,
        "curr_date": format_date(now),
        "curr_time": format_time(now),
        "user_emotion_str": user_emotion_str,
        "beliefs": belief_str,
        "mood_long_desc": system.emotion_system.get_mood_long_description(),
        "mood_prompt": system.emotion_system.get_mood_prompt(),
        "last_interaction": time_since_last_message_string(system.last_message),
        "tone_register": thought_data.get("tone_register", "grounded-natural"),
        "evidence_status": thought_data.get("evidence_status", "evidence-backed"),
    }


def estimate_pending_signal(user_input) -> EmotionSignal:
    text = str(user_input or "")
    positive = ("喜欢", "谢谢", "开心", "高兴", "温暖", "信任", "陪我", "爱你")
    negative = ("讨厌", "烦", "滚", "讽刺", "失望", "生气", "恨", "无聊")
    sad = ("难过", "伤心", "低落", "沮丧", "孤独", "疲惫", "痛苦")
    if any(token in text for token in negative):
        return EmotionSignal(mood="受伤", intensity=0.35, valence=-0.45)
    if any(token in text for token in sad):
        return EmotionSignal(mood="关切", intensity=0.25, valence=-0.15)
    if any(token in text for token in positive):
        return EmotionSignal(mood="愉快", intensity=0.2, valence=0.25)
    return EmotionSignal(mood="平静", intensity=0.05, valence=0.0)


def thought_signal(thought_data) -> EmotionSignal:
    emotion_obj = thought_data.get("emotion_obj")
    raw_mood = str(thought_data.get("emotion", "平静") or "平静").strip()
    if (
        not raw_mood
        or len(raw_mood) > 12
        or "[" in raw_mood
        or "]" in raw_mood
        or "Intensity" in raw_mood
        or "event makes" in raw_mood.lower()
    ):
        raw_mood = "平静"
    if emotion_obj is None:
        return EmotionSignal(mood=raw_mood, intensity=0.1, valence=0.0)

    intensity = max(0.0, min(1.0, emotion_obj.get_intensity()))
    valence = max(-1.0, min(1.0, emotion_obj.pleasure))
    if raw_mood == "平静":
        if valence >= 0.25:
            raw_mood = "愉快"
        elif valence <= -0.35:
            raw_mood = "难过"
        elif valence <= -0.15:
            raw_mood = "关切"
    return EmotionSignal(mood=raw_mood, intensity=intensity, valence=valence)


def derive_relation_impact(signal: EmotionSignal) -> dict:
    valence = signal.valence
    if valence > 0.15:
        return {"trust_delta": 0.01, "affection_delta": 0.01, "familiarity_delta": 0.015}
    if valence < -0.15:
        return {"trust_delta": -0.01, "affection_delta": -0.012, "familiarity_delta": 0.0}
    return {"familiarity_delta": 0.01}


def derive_topic_tags(user_input) -> list[str]:
    text = str(user_input or "")
    tokens = [
        token
        for token in re.split(r"[\s，。！？!?；：、（）()\[\]\"“”]+", text)
        if 1 < len(token) <= 12
    ]
    return tokens[:5]


def split_tool_context_by_mode(tool_report, route_decision) -> tuple[str, str]:
    context = str(getattr(tool_report, "context", "") or "")
    if context in {"", "None"}:
        return "", ""
    if route_decision.web_search_mode == SearchMode.PERSONA_SEARCH:
        return context, ""
    if route_decision.web_search_mode == SearchMode.REALITY_SEARCH:
        return "", context
    if route_decision.web_search_mode == SearchMode.BOTH:
        return context, context
    return "", ""


def summarize_debug_lines(system, text, limit=3) -> list[str]:
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        line = str(raw).strip()
        if not line or line.endswith(":") or line.startswith("["):
            continue
        lines.append(system._truncate_for_prompt(line, 120))
        if len(lines) >= limit:
            break
    return lines


def record_debug_info(
    system,
    route_decision,
    assembled_context,
    tool_context_for_turn,
    local_precise_context="",
    local_story_context="",
) -> None:
    persona_lines: list[str] = []
    for source in (
        local_precise_context,
        local_story_context,
        assembled_context.slots.get("evidence_chunks", ""),
    ):
        for line in summarize_debug_lines(system, source, limit=4):
            if line not in persona_lines:
                persona_lines.append(line)
            if len(persona_lines) >= 4:
                break
        if len(persona_lines) >= 4:
            break

    tool_lines: list[str] = []
    for line in summarize_debug_lines(system, tool_context_for_turn, limit=4):
        if line not in tool_lines:
            tool_lines.append(line)
        if len(tool_lines) >= 4:
            break

    system.last_debug_info = {
        "routeType": str(getattr(route_decision, "type", "") or ""),
        "personaEvidence": persona_lines,
        "toolEvidence": tool_lines,
    }

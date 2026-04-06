from __future__ import annotations

import re
from datetime import datetime

from knowledge.knowledge_source import SearchMode
from persona_prompting import build_base_template_injection_prompt
from reasoning.emotion_state_machine import EmotionSignal
from utils import format_date, format_memories_to_string, format_time, time_since_last_message_string


def _clean_prompt_line(text: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n-:：；，。！？?()[]")
    if not value:
        return ""
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


def relation_metrics(system) -> dict:
    relation_state = getattr(getattr(system, "new_memory_state", None), "relation_state", None)
    try:
        trust = float(getattr(relation_state, "trust", 0.0) or 0.0)
        affection = float(getattr(relation_state, "affection", 0.0) or 0.0)
        familiarity = float(getattr(relation_state, "familiarity", 0.0) or 0.0)
    except Exception:
        trust = affection = familiarity = 0.0
    warmth = max(0.0, min(1.0, (trust * 0.35) + (affection * 0.45) + (familiarity * 0.20)))
    openness = max(0.0, min(1.0, (trust * 0.30) + (familiarity * 0.45) + (affection * 0.25)))
    return {
        "trust": round(trust, 3),
        "affection": round(affection, 3),
        "familiarity": round(familiarity, 3),
        "warmth": round(warmth, 3),
        "openness": round(openness, 3),
    }


def _affinity_level_for_prompt(system) -> str:
    metrics = relation_metrics(system)
    warmth = float(metrics.get("warmth", 0.0) or 0.0)
    familiarity = float(metrics.get("familiarity", 0.0) or 0.0)
    if warmth >= 0.72 or familiarity >= 0.78:
        return "close"
    if warmth >= 0.42 or familiarity >= 0.45:
        return "familiar"
    return "stranger"


def relation_state_summary(system) -> str:
    metrics = relation_metrics(system)
    affinity_level = _affinity_level_for_prompt(system)
    guidance = {
        "stranger": "保持礼貌和适度距离，不主动越界到私密话题，不使用过分亲昵的称呼。",
        "familiar": "语气可以自然一些，允许轻微玩笑和熟悉感，但仍保留边界。",
        "close": "可以更柔和、更直接地表达在意与关心，允许自然流露亲近感。",
    }
    return (
        f"level={affinity_level}; trust={metrics['trust']}; affection={metrics['affection']}; "
        f"familiarity={metrics['familiarity']}; guidance={guidance[affinity_level]}"
    )


def build_identity_reference(system) -> str:
    base_template = getattr(system.persona_system, "base_template", {}) or {}
    background = base_template.get("00_BACKGROUND", {}) if isinstance(base_template, dict) else {}
    profile = background.get("profile", {}) if isinstance(background, dict) else {}
    lines: list[str] = []
    if isinstance(profile, dict):
        for key, value in profile.items():
            clean = _clean_prompt_line(value, 80)
            if clean:
                lines.append(f"{key}: {clean}")
    return "\n".join(lines[:8]).strip()


def build_persona_injection_prompt(system, thought_data: dict) -> str:
    persona = system.persona_system
    current_emotion = str(thought_data.get("emotion", "平静") or "平静").strip() or "平静"
    affinity_level = _affinity_level_for_prompt(system)
    return str(
        build_base_template_injection_prompt(
            character_name=system.config.name,
            character_voice_card=str(getattr(persona, "character_voice_card", "") or ""),
            base_template=getattr(persona, "base_template", {}) or {},
            style_examples=list(getattr(persona, "style_examples", []) or []),
            current_affinity_level=affinity_level,
            current_emotion=current_emotion,
            display_keywords=list(getattr(persona, "display_keywords", []) or []),
        )
        or ""
    ).strip()


def _split_evidence_blocks(persona_context: str) -> list[str]:
    text = str(persona_context or "").strip()
    if not text:
        return []
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()] or [text]


def _select_evidence_prompt(persona_context: str, response_mode: str) -> str:
    blocks = _split_evidence_blocks(persona_context)
    if not blocks:
        return ""
    if response_mode == "story_retelling":
        return blocks[0]
    if response_mode == "persona_grounded":
        return "\n\n".join(blocks[:3])
    return "\n\n".join(blocks[:4])


def _response_mode_and_contract(system, content: str, grounding: dict | None, tool_context: str) -> tuple[str, str]:
    grounding = grounding or {}
    if grounding.get("needs_story_grounding"):
        return (
            "story_retelling",
            "只选一条命中的故事证据，用第一人称直接讲，不比较，不合并，不补新事实。",
        )
    if grounding.get("needs_external_grounding") and str(tool_context or "").strip() not in {"", "None"}:
        return (
            "external_fact",
            "只依据 tool_context 回答现实事实，可以保留角色语气，但不能补充额外知识或想象场景。",
        )
    if grounding.get("is_self_intro") and grounding.get("has_identity_reference"):
        return (
            "self_intro",
            "只用身份背景信息回答‘你是谁’一类问题，不扩写未证实经历。",
        )
    if grounding.get("needs_persona_grounding"):
        return (
            "persona_grounded",
            "只依据角色证据回答，不扩展 unsupported 的喜好、习惯、过去和故事。",
        )
    return ("free_chat", "自然地以角色口吻聊天；没有证据支撑的具体事实不要编。")


def _persona_focus_contract(system, content: str, response_mode: str) -> tuple[str, str]:
    if response_mode != "persona_grounded":
        return "", ""
    focus = str(system.persona_policy.persona_query_focus(content) or "").strip()
    mapping = {
        "likes": "只回答证据里明确支持的喜欢与偏好。",
        "dislikes": "只回答证据里明确支持的讨厌、禁忌与回避。",
        "catchphrase": "只回答证据支持的固定说法、习惯句式或口头禅。",
        "personality": "只回答证据支持的性格与行为倾向。",
        "self_intro": "只回答身份背景信息。",
    }
    return focus, mapping.get(focus, "")


def recent_assistant_context(system) -> str:
    recent_messages = system._recent_assistant_messages(limit=3)
    if not recent_messages:
        return "None"
    summaries: list[str] = []
    for text in recent_messages:
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        compact = " / ".join(lines[:3]) if lines else str(text).strip()
        summaries.append("- " + system._truncate_for_prompt(compact, 120))
    return system._truncate_for_prompt("\n".join(summaries), 360)


def build_format_data(system, content, thought_data, memories, persona_context, tool_context, grounding=None) -> dict:
    now = datetime.now()
    beliefs = system.get_beliefs()
    belief_str = "\n".join(f"- {belief}" for belief in beliefs) if beliefs else "None"
    memories_str = format_memories_to_string(memories, "You do not have any stable memory about this user yet.")
    memories_str = system._truncate_for_prompt(memories_str, 420)

    response_mode, response_contract = _response_mode_and_contract(system, content, grounding, tool_context)
    persona_focus, persona_focus_contract = _persona_focus_contract(system, content, response_mode)
    style_prompt = system._truncate_for_prompt(build_persona_injection_prompt(system, thought_data), 2200)
    identity_prompt = system._truncate_for_prompt(build_identity_reference(system), 700)
    evidence_prompt = system._truncate_for_prompt(_select_evidence_prompt(persona_context, response_mode), 2000)
    tool_context = system._truncate_for_prompt(tool_context, 900)

    user_emotions = thought_data.get("possible_user_emotions", []) or []
    user_emotion_str = "The user appears to be feeling: " + ", ".join(user_emotions) if user_emotions else "The user does not show a strong explicit emotion."

    return {
        "name": system.config.name,
        "personality_summary": system.personality_system.get_summary(),
        "style_prompt": style_prompt,
        "identity_prompt": identity_prompt or "None",
        "evidence_prompt": evidence_prompt or "None",
        "tool_context": tool_context or "None",
        "response_mode": response_mode,
        "response_contract": response_contract,
        "persona_focus": persona_focus or "general",
        "persona_focus_contract": persona_focus_contract or "None",
        "recent_assistant_context": recent_assistant_context(system),
        "story_grounding_required": "yes" if system._requires_story_grounding(content) else "no",
        "story_answer_mode": "direct_retelling" if response_mode == "story_retelling" else "normal",
        "persona_grounding_required": "yes" if system._requires_persona_grounding(content) else "no",
        "external_grounding_required": "yes" if system._requires_external_grounding(content) else "no",
        "tool_evidence_available": "yes" if system._has_tool_evidence(tool_context) else "no",
        "user_input": content,
        "emotion": thought_data.get("emotion", "平静"),
        "relationship_state": relation_state_summary(system),
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
    negative = ("讨厌", "恶心", "讽刺", "失望", "生气", "恨", "无聊")
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
    raw_mood = str(thought_data.get("emotion", "平静") or "平静").strip() or "平静"
    if emotion_obj is None:
        return EmotionSignal(mood=raw_mood, intensity=0.1, valence=0.0)
    intensity = max(0.0, min(1.0, emotion_obj.get_intensity()))
    valence = max(-1.0, min(1.0, emotion_obj.pleasure))
    return EmotionSignal(mood=raw_mood, intensity=intensity, valence=valence)


def derive_relation_impact(signal: EmotionSignal) -> dict:
    valence = signal.valence
    if valence > 0.15:
        return {"trust_delta": 0.012, "affection_delta": 0.014, "familiarity_delta": 0.016}
    if valence < -0.15:
        return {"trust_delta": -0.010, "affection_delta": -0.012, "familiarity_delta": 0.003}
    return {"familiarity_delta": 0.01}


def derive_topic_tags(user_input) -> list[str]:
    text = str(user_input or "")
    tokens = [token for token in re.split(r"[\s，。！？；：、（）()\"'“”‘’]+", text) if 1 < len(token) <= 12]
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


def summarize_thoughts(system, thought_data, limit=5) -> list[str]:
    thoughts = thought_data.get("thoughts", []) if isinstance(thought_data, dict) else []
    lines: list[str] = []
    for item in thoughts:
        text = str(item.get("content", "") or "").strip() if isinstance(item, dict) else str(item or "").strip()
        if not text:
            continue
        lines.append(system._truncate_for_prompt(text, 120))
        if len(lines) >= limit:
            break
    return lines


def record_debug_info(system, route_decision, assembled_context, tool_context_for_turn, thought_data=None, local_precise_context="", local_story_context="") -> None:
    persona_lines: list[str] = []
    for source in (local_precise_context, local_story_context, assembled_context.slots.get("evidence_chunks", "")):
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
        "thoughts": summarize_thoughts(system, thought_data, limit=5),
    }

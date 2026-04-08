from __future__ import annotations

import re
from datetime import datetime

from knowledge.knowledge_source import SearchMode
from persona_prompting import build_base_template_injection_prompt
from reasoning.emotion_state_machine import EmotionSignal
from utils import format_date, format_memories_to_string, format_time, time_since_last_message_string


TURN_IMPACT_SCHEMA = {
    "mood": "str",
    "intensity": "float",
    "valence": "float",
    "trust_delta": "float",
    "affection_delta": "float",
    "familiarity_delta": "float",
    "rationale": "str",
}


def _clean_prompt_line(text: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n-:：；，。！？、\"'[]")
    if not value:
        return ""
    return value if len(value) <= limit else value[:limit].rstrip() + "..."


def relation_metrics(system) -> dict:
    relation_state = getattr(getattr(system, "new_memory_state", None), "relation_state", None)
    try:
        trust = float(getattr(relation_state, "trust", 0.0) or 0.0)
        affection = float(getattr(relation_state, "affection", 0.0) or 0.0)
        familiarity = float(getattr(relation_state, "familiarity", 0.0) or 0.0)
    except Exception:
        trust = affection = familiarity = 0.0

    warmth = max(0.0, min(1.0, trust * 0.35 + affection * 0.45 + familiarity * 0.20))
    openness = max(0.0, min(1.0, trust * 0.30 + familiarity * 0.45 + affection * 0.25))
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
        "stranger": "保持礼貌和适度距离，不要过分越界到私密话题，也不要使用过分亲昵的称呼。",
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
    experiences = background.get("key_experiences", []) if isinstance(background, dict) else []
    lines: list[str] = []
    if isinstance(profile, dict):
        for key, value in profile.items():
            clean = _clean_prompt_line(value, 80)
            if clean:
                lines.append(f"{key}: {clean}")
    for item in list(experiences or [])[:4]:
        clean = _clean_prompt_line(item, 100)
        if clean:
            lines.append(f"经历: {clean}")
    return "\n".join(lines[:10]).strip()


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
    if response_mode == "story":
        return blocks[0]
    if response_mode in {"persona_fact", "self_intro"}:
        return "\n\n".join(blocks[:3])
    return "\n\n".join(blocks[:4])


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
    memories_str = format_memories_to_string(memories, "你暂时还没有关于这位用户的稳定记忆。")
    memories_str = system._truncate_for_prompt(memories_str, 420)

    grounding = grounding or {}
    response_mode = str(grounding.get("response_mode", "casual") or "casual")
    persona_focus = str(grounding.get("persona_focus", "general") or "general")
    response_contract = str(grounding.get("response_contract", "") or "").strip()
    persona_focus_contract = str(grounding.get("persona_focus_contract", "") or "").strip()

    style_prompt = system._truncate_for_prompt(build_persona_injection_prompt(system, thought_data), 2200)
    identity_prompt = system._truncate_for_prompt(build_identity_reference(system), 700)
    evidence_prompt = system._truncate_for_prompt(_select_evidence_prompt(persona_context, response_mode), 2000)
    tool_context = system._truncate_for_prompt(tool_context, 900)

    user_emotions = thought_data.get("possible_user_emotions", []) or []
    if user_emotions:
        user_emotion_str = "用户可能正在感受：" + "、".join(user_emotions)
    else:
        user_emotion_str = "用户没有表现出非常强烈的显性情绪。"

    return {
        "name": system.config.name,
        "personality_summary": system.personality_system.get_summary(),
        "style_prompt": style_prompt,
        "identity_prompt": identity_prompt or "None",
        "evidence_prompt": evidence_prompt or "None",
        "tool_context": tool_context or "None",
        "response_mode": response_mode,
        "response_contract": response_contract or "自然回答，但不要编造没有证据的事实。",
        "persona_focus": persona_focus or "general",
        "persona_focus_contract": persona_focus_contract or "None",
        "recent_assistant_context": recent_assistant_context(system),
        "story_grounding_required": "yes" if response_mode == "story" else "no",
        "story_answer_mode": "direct_retelling" if response_mode == "story" else "normal",
        "persona_grounding_required": "yes" if response_mode in {"self_intro", "persona_fact", "story"} else "no",
        "external_grounding_required": "yes" if response_mode == "external" else "no",
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


def estimate_pending_signal(system, user_input, recent_conversation: str = "") -> EmotionSignal:
    text = str(user_input or "").strip()
    if not text:
        return EmotionSignal()

    prompt = f"""
你在做“本轮互动影响评估”。请根据用户这句话及最近对话，判断这轮互动对角色当前情绪和关系状态的影响。

要求：
1. 不要靠关键词机械匹配，要理解语义、语气、赞美、安抚、依赖、冷淡、攻击、命令、调情、疏离等互动姿态。
2. 只评估“这一轮互动影响”，不要总结角色设定，不要写长解释。
3. 如果用户是在夸赞、关心、表达信任、示好、依赖，允许给出正向 affection/trust 变化。
4. 如果用户是在攻击、嘲讽、冷漠命令、贬低，允许给出负向变化。
5. 如果只是普通信息提问或平淡闲聊，变化应接近 0。
6. 数值必须保守：
   - intensity: 0 到 1
   - valence: -1 到 1
   - trust_delta / affection_delta: -0.08 到 0.08
   - familiarity_delta: -0.04 到 0.04
7. mood 用简短中文，如：平静、愉快、受伤、关切、警惕、害羞、无奈、放松、期待。

最近对话：
{str(recent_conversation or "无").strip()}

用户本轮输入：
{text}
""".strip()

    try:
        payload = system.model.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=320,
            return_json=True,
            schema=TURN_IMPACT_SCHEMA,
        )
        if not isinstance(payload, dict):
            return EmotionSignal()
        return EmotionSignal(
            mood=str(payload.get("mood", "平静") or "平静").strip() or "平静",
            intensity=max(0.0, min(1.0, float(payload.get("intensity", 0.0) or 0.0))),
            valence=max(-1.0, min(1.0, float(payload.get("valence", 0.0) or 0.0))),
            trust_delta=max(-0.08, min(0.08, float(payload.get("trust_delta", 0.0) or 0.0))),
            affection_delta=max(-0.08, min(0.08, float(payload.get("affection_delta", 0.0) or 0.0))),
            familiarity_delta=max(-0.04, min(0.04, float(payload.get("familiarity_delta", 0.0) or 0.0))),
            rationale=_clean_prompt_line(payload.get("rationale", ""), 160),
        )
    except Exception:
        return EmotionSignal()


def thought_signal(thought_data) -> EmotionSignal:
    emotion_obj = thought_data.get("emotion_obj")
    raw_mood = str(thought_data.get("emotion", "平静") or "平静").strip() or "平静"
    if emotion_obj is None:
        return EmotionSignal(mood=raw_mood, intensity=0.1, valence=0.0)
    intensity = max(0.0, min(1.0, emotion_obj.get_intensity()))
    valence = max(-1.0, min(1.0, emotion_obj.pleasure))
    return EmotionSignal(mood=raw_mood, intensity=intensity, valence=valence)


def derive_relation_impact(signal: EmotionSignal) -> dict:
    trust_delta = float(getattr(signal, "trust_delta", 0.0) or 0.0)
    affection_delta = float(getattr(signal, "affection_delta", 0.0) or 0.0)
    familiarity_delta = float(getattr(signal, "familiarity_delta", 0.0) or 0.0)
    return {
        "trust_delta": max(-0.08, min(0.08, trust_delta)),
        "affection_delta": max(-0.08, min(0.08, affection_delta)),
        "familiarity_delta": max(-0.04, min(0.04, familiarity_delta)),
    }


def derive_topic_tags(user_input) -> list[str]:
    text = str(user_input or "")
    tokens = [
        token
        for token in re.split(r"[\s，。！？；：、】【（）\"“”‘’]+", text)
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


def summarize_internal_feelings(system, thought_data, limit=4) -> list[str]:
    feelings = thought_data.get("internal_feelings", []) if isinstance(thought_data, dict) else []
    lines: list[str] = []
    for item in feelings:
        text = str(item or "").strip()
        if not text:
            continue
        lines.append(system._truncate_for_prompt(text, 120))
        if len(lines) >= limit:
            break
    return lines


def _terminal_debug_report(system, payload: dict) -> None:
    print("[DEBUG] ===== Turn Trace =====")
    print(f"[DEBUG] route={payload.get('route')} intent={payload.get('intent')} tool={payload.get('tool')}")
    print(f"[DEBUG] topic={payload.get('topic')} keywords={payload.get('keywords')}")
    print(f"[DEBUG] recall_query={payload.get('recall_query')}")
    print(f"[DEBUG] coverage={payload.get('coverage')} vector_unavailable={payload.get('vector_unavailable')}")
    print(
        f"[DEBUG] query_type={payload.get('query_type')} "
        f"direct={payload.get('direct_query')} multi={payload.get('multi_query_count')}"
    )

    hits = payload.get("hits") or []
    print("[DEBUG] retrieved chunks:")
    if hits:
        for idx, hit in enumerate(hits[:6], start=1):
            print(f"  [{idx}] score={hit.get('score')} title={hit.get('title')}")
            print(f"      path={hit.get('path')}")
            print(f"      text={hit.get('text')}")
    else:
        print("  (none)")

    thoughts = payload.get("thoughts") or []
    print("[DEBUG] slow thoughts:")
    if thoughts:
        for line in thoughts:
            print(f"  - {line}")
    else:
        print("  (none)")

    internal_feelings = payload.get("internal_feelings") or []
    print("[DEBUG] internal feelings:")
    if internal_feelings:
        for line in internal_feelings:
            print(f"  - {line}")
    else:
        print("  (none)")


def record_debug_info(
    system,
    route_decision,
    assembled_context,
    local_precise_context="",
    tool_report=None,
    intent_result=None,
    persona_recall=None,
    thought_data=None,
) -> None:
    metadata = getattr(persona_recall, "metadata", {}) or {}
    query_plan = metadata.get("query_plan", {}) or {}
    raw_hits = metadata.get("hits", []) or []
    hits = []
    for hit in raw_hits[:6]:
        if not isinstance(hit, dict):
            continue
        hits.append(
            {
                "score": round(float(hit.get("score", 0.0) or 0.0), 3),
                "title": str(hit.get("title", "") or ""),
                "path": " > ".join(list(hit.get("markdown_path", []) or [])) or str(hit.get("source_label", "") or ""),
                "text": system._truncate_for_prompt(hit.get("content", ""), 320),
            }
        )

    payload = {
        "route": getattr(route_decision, "type", None),
        "intent": getattr(intent_result, "intent", None),
        "tool": getattr(intent_result, "tool_name", "none") if intent_result else "none",
        "topic": getattr(intent_result, "extracted_topic", ""),
        "keywords": list(getattr(intent_result, "extracted_keywords", []) or []),
        "recall_query": (system.last_debug_info or {}).get("recallQuery", ""),
        "coverage": round(float(getattr(persona_recall, "coverage_score", 0.0) or 0.0), 3),
        "vector_unavailable": bool(metadata.get("vector_unavailable", False)),
        "query_type": str(query_plan.get("query_type", "") or ""),
        "direct_query": str(query_plan.get("direct_query", "") or ""),
        "multi_query_count": len(list(query_plan.get("multi_queries", []) or [])),
        "hits": hits,
        "thoughts": summarize_thoughts(system, thought_data),
        "internal_feelings": summarize_internal_feelings(system, thought_data),
    }
    system.last_debug_info = {**(system.last_debug_info or {}), "turnTrace": payload}
    _terminal_debug_report(system, payload)

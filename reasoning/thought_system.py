from __future__ import annotations

import json
from collections import deque
from datetime import datetime

from const import EMOTION_MAP
from reasoning.emotion_state_machine import Emotion


PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "surface_intent": {"type": "string"},
        "latent_need": {"type": "string"},
        "possible_user_emotions": {"type": "array", "items": {"type": "string"}},
        "character_emotion": {"type": "string"},
        "emotion_reason": {"type": "string"},
        "emotion_intensity": {"type": "integer"},
        "tone_register": {"type": "string"},
        "evidence_status": {"type": "string"},
        "response_goal": {"type": "string"},
        "response_mode_override": {"type": "string"},
        "persona_focus_override": {"type": "string"},
        "thought_summary": {"type": "array", "items": {"type": "string"}},
        "internal_feelings": {"type": "array", "items": {"type": "string"}},
        "reasoning_summary": {"type": "string"},
        "intimacy_ceiling": {"type": "number"},
        "warmth": {"type": "number"},
        "directness": {"type": "number"},
    },
    "required": [
        "surface_intent",
        "latent_need",
        "possible_user_emotions",
        "character_emotion",
        "emotion_reason",
        "emotion_intensity",
        "tone_register",
        "evidence_status",
        "response_goal",
        "response_mode_override",
        "persona_focus_override",
        "thought_summary",
        "internal_feelings",
        "reasoning_summary",
        "intimacy_ceiling",
        "warmth",
        "directness",
    ],
}


class ThoughtSystem:
    def __init__(self, config, emotion_system, memory_system, relation_system, personality_system, model=None):
        self.config = config
        self.emotion_system = emotion_system
        self.memory_system = memory_system
        self.relation_system = relation_system
        self.personality_system = personality_system
        self.model = model
        self.show_thoughts = False
        self.last_reflection = datetime.now()

    def can_reflect(self):
        return False

    def reflect(self):
        return None

    def _fallback_thought_output(self):
        return {
            "surface_intent": "",
            "latent_need": "",
            "thoughts": [],
            "possible_user_emotions": [],
            "tone_register": "grounded-natural",
            "evidence_status": "unknown",
            "emotion_intensity": 2,
            "emotion": "Neutral",
            "emotion_reason": "planner_unavailable",
            "response_goal": "优先保守、贴着证据回应，不额外编造细节。",
            "response_mode_override": "",
            "persona_focus_override": "",
            "internal_feelings": [],
            "reasoning_summary": "",
            "intimacy_ceiling": 0.4,
            "warmth": 0.5,
            "directness": 0.5,
            "relationship_change": {"friendliness": 0.0, "dominance": 0.0},
            "emotion_obj": Emotion(),
        }

    def _last_content(self, messages) -> str:
        if not messages:
            return ""
        last_message_payload = messages[-1].get("content", "")
        if isinstance(last_message_payload, list):
            text_parts = []
            for item in last_message_payload:
                if isinstance(item, dict) and item.get("text"):
                    text_parts.append(str(item["text"]))
            return "\n".join(text_parts)
        return str(last_message_payload or "")

    def _json_safe(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, deque):
            return [self._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "model_dump"):
            return self._json_safe(value.model_dump())
        if hasattr(value, "dict"):
            return self._json_safe(value.dict())
        if hasattr(value, "as_dict"):
            return self._json_safe(value.as_dict())
        return str(value)

    def _planner_prompt(
        self,
        *,
        last_content: str,
        recent_conversation: str,
        persona_context: str,
        session_context: str,
        working_memory: dict,
        relation_state: dict,
        intent_snapshot: dict,
        route_snapshot: dict,
        tool_snapshot: dict,
        persona_decision_card: dict,
    ) -> str:
        working_summary = json.dumps(self._json_safe(working_memory or {}), ensure_ascii=False, indent=2)
        relation_summary = json.dumps(self._json_safe(relation_state or {}), ensure_ascii=False, indent=2)
        intent_summary = json.dumps(self._json_safe(intent_snapshot or {}), ensure_ascii=False, indent=2)
        route_summary = json.dumps(self._json_safe(route_snapshot or {}), ensure_ascii=False, indent=2)
        tool_summary = json.dumps(self._json_safe(tool_snapshot or {}), ensure_ascii=False, indent=2)
        persona_card_summary = json.dumps(self._json_safe(persona_decision_card or {}), ensure_ascii=False, indent=2)
        evidence_preview = str(persona_context or "").strip()[:1200] or "None"
        session_preview = str(session_context or "").strip()[:800] or "None"
        recent_preview = str(recent_conversation or "").strip()[:1000] or "None"
        user_text = str(last_content or "").strip() or "None"

        return f"""
你是角色陪伴型对话系统里的“慢思考规划器”。你的任务不是直接回复用户，而是基于当前状态输出一份结构化决策。

必须遵守：
1. 你只输出 JSON。
2. 你的职责是判断：用户表层意图、隐含需求、当前情绪/关系语境、语气策略、证据使用优先级、以及这轮回应目标。
3. 不要复述系统规则，不要生成最终对用户的回答。
4. 不要为了角色感而胡乱扩展故事或事实。story 没证据时必须保守。
5. 人设信息只用来帮助决策风格，不是让你复述设定。
6. 角色决策卡是通用的决策抽象，不是某个角色的专属硬规则；你要根据当前用户输入、关系状态、证据状态来灵活判断。

[用户输入]
{user_text}

[最近对话摘要]
{recent_preview}

[会话上下文]
{session_preview}

[工作记忆]
{working_summary}

[关系状态]
{relation_summary}

[初始意图判断]
{intent_summary}

[初始路由判断]
{route_summary}

[工具状态]
{tool_summary}

[角色决策卡（决策抽象，不是设定原文）]
{persona_card_summary}

[当前命中的人物/故事证据摘要]
{evidence_preview}

字段要求：
- surface_intent: 这句表层上在问什么
- latent_need: 这句更深层可能在试探、寻求、确认什么
- possible_user_emotions: 用户可能情绪标签，最多 4 个
- character_emotion: 角色此刻更接近哪种 OCC/PAD 对应情绪，只能用现有情绪名；没有明显波动就用 Neutral
- emotion_reason: 为什么会出现这个情绪
- emotion_intensity: 1 到 5
- tone_register: 简短描述这轮说话基调，例如 grounded-natural / warm-contained / gentle-direct
- evidence_status: unsupported / partial / grounded 三选一
- response_goal: 这轮回答最应该先完成什么
- response_mode_override: 如果你认为当前初始 mode 不理想，可以给出建议的 mode；否则留空
- persona_focus_override: 如有必要给出 persona focus 建议，否则留空
- thought_summary: 3 到 4 条简短决策摘要，供 debug 展示
- internal_feelings: 1 到 3 条角色内部感受摘要，供 debug 展示
- reasoning_summary: 一句总括性的慢思考总结
- intimacy_ceiling / warmth / directness: 0 到 1 之间

注意：
- 不要因为角色决策卡里有 identity_core 或 core_traits，就把这轮回答变成设定复述。
- 决策时，用户当下需求和证据边界优先于角色风格偏置。
""".strip()

    def _normalize_result(self, raw: dict | None) -> dict:
        data = self._fallback_thought_output()
        payload = dict(raw or {})

        emotion_name = str(payload.get("character_emotion", "Neutral") or "Neutral").strip() or "Neutral"
        if emotion_name not in EMOTION_MAP:
            emotion_name = "Neutral"
        intensity = int(payload.get("emotion_intensity", 2) or 2)
        intensity = max(1, min(5, intensity))

        thought_summary = [
            {"content": str(item or "").strip()}
            for item in list(payload.get("thought_summary", []) or [])
            if str(item or "").strip()
        ][:4]
        internal_feelings = [
            str(item or "").strip()
            for item in list(payload.get("internal_feelings", []) or [])
            if str(item or "").strip()
        ][:4]

        data.update(
            {
                "surface_intent": str(payload.get("surface_intent", "") or "").strip(),
                "latent_need": str(payload.get("latent_need", "") or "").strip(),
                "possible_user_emotions": [
                    str(item or "").strip()
                    for item in list(payload.get("possible_user_emotions", []) or [])
                    if str(item or "").strip()
                ][:4],
                "emotion": emotion_name,
                "emotion_reason": str(payload.get("emotion_reason", "") or "").strip() or "planner_unavailable",
                "emotion_intensity": intensity,
                "tone_register": str(payload.get("tone_register", "grounded-natural") or "grounded-natural").strip(),
                "evidence_status": str(payload.get("evidence_status", "unknown") or "unknown").strip(),
                "response_goal": str(payload.get("response_goal", "") or "").strip(),
                "response_mode_override": str(payload.get("response_mode_override", "") or "").strip(),
                "persona_focus_override": str(payload.get("persona_focus_override", "") or "").strip(),
                "thoughts": thought_summary,
                "internal_feelings": internal_feelings,
                "reasoning_summary": str(payload.get("reasoning_summary", "") or "").strip(),
                "intimacy_ceiling": max(0.0, min(1.0, float(payload.get("intimacy_ceiling", 0.4) or 0.4))),
                "warmth": max(0.0, min(1.0, float(payload.get("warmth", 0.5) or 0.5))),
                "directness": max(0.0, min(1.0, float(payload.get("directness", 0.5) or 0.5))),
            }
        )

        if emotion_name == "Neutral":
            data["emotion_obj"] = Emotion()
        else:
            data["emotion_obj"] = self.emotion_system.experience_emotion(emotion_name, intensity / 10.0)
        return data

    def think(
        self,
        messages,
        memories,
        recalled_memories,
        last_message,
        persona_context="",
        *,
        recent_conversation: str = "",
        session_context: str = "",
        working_memory: dict | None = None,
        relation_state: dict | None = None,
        intent_snapshot: dict | None = None,
        route_snapshot: dict | None = None,
        tool_snapshot: dict | None = None,
        persona_decision_card: dict | None = None,
    ):
        data = self._fallback_thought_output()
        if self.model is None:
            return data

        prompt = self._planner_prompt(
            last_content=self._last_content(messages),
            recent_conversation=recent_conversation,
            persona_context=persona_context,
            session_context=session_context,
            working_memory=working_memory or {},
            relation_state=relation_state or {},
            intent_snapshot=intent_snapshot or {},
            route_snapshot=route_snapshot or {},
            tool_snapshot=tool_snapshot or {},
            persona_decision_card=persona_decision_card or {},
        )
        try:
            payload = self.model.generate(
                prompt,
                return_json=True,
                schema=PLANNER_SCHEMA,
                temperature=0.15,
                max_tokens=520,
            )
            return self._normalize_result(payload if isinstance(payload, dict) else {})
        except Exception:
            return data

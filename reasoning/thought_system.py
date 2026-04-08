from __future__ import annotations

from datetime import datetime

from reasoning.emotion_state_machine import Emotion


class ThoughtSystem:
    def __init__(self, config, emotion_system, memory_system, relation_system, personality_system):
        self.config = config
        self.emotion_system = emotion_system
        self.memory_system = memory_system
        self.relation_system = relation_system
        self.personality_system = personality_system
        self.show_thoughts = False
        self.last_reflection = datetime.now()

    def can_reflect(self):
        return False

    def reflect(self):
        return None

    def _fallback_thought_output(self):
        return {
            "thoughts": [
                {"content": "先接住用户真正想问的东西，不绕远。"},
                {"content": "如果有证据，就沿着证据说；没有证据，就别硬补。"},
                {"content": "保持第一人称，像角色本人，而不是旁白。"},
                {"content": "把重点说清楚，同时别把角色味道压扁。"},
                {"content": "证据足够时可以自然展开，不必总缩成一句话。"},
            ],
            "possible_user_emotions": [],
            "emotion_mult": {},
            "tone_register": "grounded-natural",
            "evidence_status": "unsupported",
            "emotion_intensity": 3,
            "emotion": "Neutral",
            "emotion_reason": "当前先保持克制和稳定。",
            "next_action": "final_answer",
            "relationship_change": {"friendliness": 0.0, "dominance": 0.0},
        }

    def _infer_user_emotions(self, text: str) -> list[str]:
        emotions: list[str] = []
        if any(token in text for token in ("开心", "高兴", "喜欢", "期待", "谢谢", "太好了")):
            emotions.append("positive")
        if any(token in text for token in ("难过", "伤心", "失落", "低落", "委屈", "孤独")):
            emotions.append("sad")
        if any(token in text for token in ("讨厌", "恶心", "生气", "愤怒", "失望")):
            emotions.append("negative")
        return emotions[:3]

    def _infer_character_emotion(self, text: str) -> tuple[str, str, int]:
        if any(token in text for token in ("谢谢", "喜欢你", "辛苦了", "抱抱", "晚安")):
            return "Gratitude", "用户表达偏温和或亲近，回应可以更自然柔和一些。", 4
        if any(token in text for token in ("讨厌", "闭嘴", "滚", "烦死了", "恶心")):
            return "Reproach", "用户措辞带有明显攻击性，回应需要收敛并降温。", 5
        if any(token in text for token in ("难过", "伤心", "失落", "低落", "委屈")):
            return "Pity", "用户状态偏低落，回应应更克制、更关照。", 4
        return "Neutral", "当前没有特别强的情绪波动，保持平静自然。", 3

    def _build_internal_thoughts(self, text: str, persona_context: str) -> list[dict]:
        evidence_status = "evidence-backed" if str(persona_context or "").strip() else "unsupported"
        thoughts = [
            "先顺着这句话的核心意思回答，不绕远。",
            "如果这是现实信息问题，就只吃工具结果，不拿人设去乱补。",
            "如果有角色证据，就让语气和立场自然落在证据上。",
            "保持第一人称，别把自己说成档案。",
            "先把重点说清楚，但别把表达压得太短。",
        ]
        if evidence_status == "evidence-backed":
            thoughts[2] = "这句可以吃到现有证据，重点是让角色语气稳定落地。"
        return [{"content": item} for item in thoughts]

    def think(self, messages, memories, recalled_memories, last_message, persona_context=""):
        data = self._fallback_thought_output()

        last_content = ""
        if messages:
            last_message_payload = messages[-1].get("content", "")
            if isinstance(last_message_payload, list):
                text_parts = []
                for item in last_message_payload:
                    if isinstance(item, dict) and item.get("text"):
                        text_parts.append(str(item["text"]))
                last_content = "\n".join(text_parts)
            else:
                last_content = str(last_message_payload or "")

        emotion_name, emotion_reason, intensity = self._infer_character_emotion(last_content)
        data["possible_user_emotions"] = self._infer_user_emotions(last_content)
        data["emotion"] = emotion_name
        data["emotion_reason"] = emotion_reason
        data["emotion_intensity"] = intensity
        data["thoughts"] = self._build_internal_thoughts(last_content, persona_context)
        data["evidence_status"] = "evidence-backed" if str(persona_context or "").strip() else "unsupported"

        if emotion_name == "Neutral":
            data["emotion_obj"] = Emotion()
        else:
            data["emotion_obj"] = self.emotion_system.experience_emotion(emotion_name, intensity / 10.0)
        return data

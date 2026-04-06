from __future__ import annotations

from datetime import datetime

from reasoning.emotion_state_machine import Emotion


class ThoughtSystem:
    """轻量本地慢思考系统。

    这里不再额外请求 LLM，而是保留一层可见的本地思考链：
    情绪判断 -> 证据判断 -> 回答策略。
    这样既能保留“慢思考过程”的可见性，也不会把超时风险再放大。
    """

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
                {"content": "先把用户这句话的核心问题接住，不急着铺设定。"},
                {"content": "如果资料里有明确证据，就顺着证据回答。"},
                {"content": "如果现实信息没有查证，不强行下结论。"},
                {"content": "保持第一人称，像角色本人，而不是分析报告。"},
                {"content": "先自然回应，再考虑补充细节。"},
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
            return "Gratitude", "用户表达偏温和或亲近，回应可以自然柔和一些。", 4
        if any(token in text for token in ("讨厌", "闭嘴", "滚", "烦死了", "恶心")):
            return "Reproach", "用户措辞带有明显攻击性，回应需要收敛并降温。", 5
        if any(token in text for token in ("难过", "伤心", "失落", "低落", "委屈")):
            return "Pity", "用户状态偏低落，回应应更克制、更关照。", 4
        return "Neutral", "当前没有特别强的情绪波动，保持平静自然。", 3

    def _build_internal_thoughts(self, text: str, persona_context: str) -> list[dict]:
        evidence_status = "evidence-backed" if str(persona_context or "").strip() else "unsupported"
        thoughts = [
            "先顺着这句话的核心意思回答，不绕远。",
            "如果角色资料里有依据，就让语气和立场自然落在那些依据上。",
            "如果是现实信息问题，没有证据就别硬说。",
            "保持第一人称，别把自己说成档案。",
            "这次回答尽量短一点，先把最该说的说清楚。",
        ]
        if evidence_status == "evidence-backed":
            thoughts[1] = "这句可以吃到现有人设证据，重点是让角色声音稳定落地。"
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

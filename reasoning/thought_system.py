import json
from datetime import datetime

from const import (
    ADDED_CONTEXT_TEMPLATE,
    EMOTION_MAP,
    HIGHER_ORDER_THOUGHTS,
    MAX_THOUGHT_STEPS,
    MEMORY_RETRIEVAL_TOP_K,
    REFLECT_GEN_INSIGHTS,
    REFLECT_GEN_TOPICS,
    THOUGHT_PROMPT,
    THOUGHT_SCHEMA,
)
from llm import FallbackMistralLLM
from reasoning.emotion_state_machine import Emotion
from safe_colored import Fore, Style
from utils import format_date, format_memories_to_string, format_time, time_since_last_message_string


class ThoughtSystem:
    def __init__(self, config, emotion_system, memory_system, relation_system, personality_system):
        self.model = FallbackMistralLLM()
        self.config = config
        self.emotion_system = emotion_system
        self.memory_system = memory_system
        self.relation_system = relation_system
        self.personality_system = personality_system
        self.show_thoughts = True
        self.last_reflection = datetime.now()

    def can_reflect(self):
        return (
            self.memory_system.importance_counter >= 10
            and (datetime.now() - self.last_reflection).total_seconds() > 6 * 3600
            and len(self.memory_system.get_short_term_memories()) >= 5
        )

    def reflect(self):
        recent_memories = self.memory_system.get_short_term_memories()
        memories_str = "\n".join(mem.format_memory() for mem in recent_memories)
        questions = self.model.generate(
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": REFLECT_GEN_TOPICS.format(memories=memories_str)},
            ],
            temperature=0.1,
            return_json=True,
        ).get("questions", [])
        for question in questions[:3]:
            relevant_memories = self.memory_system.get_short_term_memories() + self.memory_system.retrieve_long_term(question, 12)
            memory_text = "\n".join(mem.format_memory() for mem in relevant_memories)
            insights = self.model.generate(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": REFLECT_GEN_INSIGHTS.format(memories=memory_text, question=question)},
                ],
                temperature=0.1,
                return_json=True,
            ).get("insights", [])
            for insight in insights[:3]:
                self.memory_system.remember(f"I gained an insight after reflection: {insight}", is_insight=True)
        self.memory_system.reset_importance()
        self.last_reflection = datetime.now()

    def _check_and_fix_thought_output(self, data):
        data = dict(data or {})
        data.setdefault("thoughts", [])
        if len(data["thoughts"]) < 5:
            defaults = self._fallback_thought_output()["thoughts"]
            data["thoughts"] = (data["thoughts"] + defaults)[:5]
        data.setdefault("possible_user_emotions", [])
        data.setdefault("emotion_mult", {})
        data.setdefault("tone_register", "grounded-natural")
        data.setdefault("evidence_status", "evidence-backed")
        data.setdefault("emotion_intensity", 5)
        data["emotion_intensity"] = max(1, min(10, int(data["emotion_intensity"])))
        data.setdefault("emotion", "Neutral")
        data.setdefault("emotion_reason", "I feel this way based on the current conversation.")
        if data["emotion"] not in EMOTION_MAP:
            data["emotion"] = "Neutral"
        data.setdefault("next_action", "final_answer")
        data.setdefault("relationship_change", {"friendliness": 0.0, "dominance": 0.0})
        return data

    def _fallback_thought_output(self):
        return {
            "thoughts": [
                {"content": "我先保持冷静，不要因为信息不足就把回答说满。"},
                {"content": "如果角色资料里已经有依据，我应该优先顺着那些内容来回应。"},
                {"content": "如果这是现实事实问题，我需要更看重工具结果，而不是靠感觉回答。"},
                {"content": "语气可以保留角色本身的味道，但内容必须尽量稳妥。"},
                {"content": "先把话说清楚、说自然，再考虑额外延展。"},
            ],
            "possible_user_emotions": [],
            "emotion_mult": {},
            "tone_register": "conservative-unsure",
            "evidence_status": "unsupported",
            "emotion_intensity": 4,
            "emotion": "Neutral",
            "emotion_reason": "当前缺少稳定依据，因此保持克制。",
            "next_action": "final_answer",
            "relationship_change": {"friendliness": 0.0, "dominance": 0.0},
        }

    def think(self, messages, memories, recalled_memories, last_message, persona_context=""):
        memories_str = format_memories_to_string(memories, "You don't have any memories of this user yet.")

        memory_emotion = Emotion()
        if recalled_memories:
            total_weight = 0.0
            for memory in recalled_memories:
                weight = memory.get_recency_factor(True)
                memory_emotion += memory.emotion * weight
                total_weight += weight
            if total_weight > 0:
                memory_emotion /= total_weight
                self.emotion_system.add_emotion(memory_emotion * 0.3)

        content = messages[-1]["content"]
        img_data = None
        if isinstance(content, list):
            text_content = content[0]["text"] + "\n\n((The user attached an image to this message - please see the attached image.))"
            img_data = content[1]
        else:
            text_content = content

        beliefs = self.memory_system.get_beliefs()
        belief_str = "\n".join(f"- {belief}" for belief in beliefs) if beliefs else "None"

        appraisal = self.emotion_system.appraisal(messages, memories, beliefs)
        appraisal_str = ", ".join(
            f"{emotion} (Intensity {round(intensity * 100)}%)" for emotion, intensity in appraisal if intensity >= 0.05
        )
        appraisal_hint = f"[This event makes {self.config.name} feel: {appraisal_str}]" if appraisal_str else ""

        prompt = THOUGHT_PROMPT.format(
            name=self.config.name,
            user_input=text_content,
            personality_summary=self.personality_system.get_summary(),
            persona_context=persona_context or "None",
            mood_long_desc=self.emotion_system.get_mood_long_description(),
            curr_date=format_date(datetime.now()),
            curr_time=format_time(datetime.now()),
            mood_prompt=self.emotion_system.get_mood_prompt(),
            memories=memories_str,
            relationship_str=self.relation_system.get_string(),
            beliefs=belief_str,
            last_interaction=time_since_last_message_string(last_message),
            appraisal_hint=appraisal_hint,
        )
        prompt_content = prompt if img_data is None else [{"type": "text", "text": prompt}, img_data]

        thought_history = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": "[START OF PREVIOUS CHAT HISTORY]"},
            *messages[:-1],
            {"role": "user", "content": "[END OF PREVIOUS CHAT HISTORY]"},
            {"role": "user", "content": prompt_content},
        ]

        try:
            data = self.model.generate(thought_history, temperature=0.6, return_json=True, schema=THOUGHT_SCHEMA)
        except Exception:
            data = self._fallback_thought_output()
        data = self._check_and_fix_thought_output(data)

        if self.show_thoughts:
            print("思考中：")
            for thought in data["thoughts"]:
                print(Fore.magenta + thought["content"] + Style.reset)

        num_steps = 0
        while data.get("next_action", "").lower() == "continue_thinking" and num_steps < MAX_THOUGHT_STEPS:
            num_steps += 1
            related = self.memory_system.retrieve_long_term(" ".join(t["content"] for t in data["thoughts"]), MEMORY_RETRIEVAL_TOP_K)
            added_context = ADDED_CONTEXT_TEMPLATE.format(memories="\n".join(mem.format_memory() for mem in related)) if related else ""
            thought_history.append({"role": "user", "content": HIGHER_ORDER_THOUGHTS.format(added_context=added_context)})
            try:
                new_data = self.model.generate(thought_history, temperature=0.6, return_json=True, schema=THOUGHT_SCHEMA)
            except Exception:
                break
            new_data = self._check_and_fix_thought_output(new_data)
            data["thoughts"].extend(new_data["thoughts"])
            data["next_action"] = new_data["next_action"]

        appraised_emotions = [emotion for emotion, intensity in appraisal if intensity >= 0.05]
        emotion_mult = data.get("emotion_mult", {})
        for key in list(emotion_mult.keys()):
            if key in appraised_emotions:
                emotion_mult[key] = max(0.5, min(1.5, emotion_mult[key]))
            else:
                del emotion_mult[key]
        if not appraisal and data["emotion"] != "Neutral":
            appraisal = [(data["emotion"], data["emotion_intensity"] / 10)]

        total_emotion = Emotion()
        for emotion, intensity in appraisal:
            intensity *= emotion_mult.get(emotion, 1.0)
            total_emotion += self.emotion_system.experience_emotion(emotion, intensity)
        self.emotion_system.clamp_mood()

        data["appraisal_hint"] = appraisal_hint
        data["emotion_obj"] = total_emotion
        return data

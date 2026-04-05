from __future__ import annotations

import math
import random
from datetime import datetime

from pydantic import BaseModel, Field

from const import (
    APPRAISAL_PROMPT,
    APPRAISAL_SCHEMA,
    EMOTION_HALF_LIFE,
    EMOTION_APPRAISAL_CONTEXT_TEMPLATE,
    EMOTION_MAP,
    EMOTION_PROMPTS,
    MODD_INTENSITY_FACTOR,
    MOOD_HALF_LIFE,
    PERSONALITY_INTENSITY_FACTOR,
    SUMMARIZE_PERSONALITY,
)
from llm import FallbackMistralLLM
from safe_colored import Fore
from utils import conversation_to_string, format_memories_to_string, num_to_str_sign, val_to_symbol_color


def get_default_mood(openness, conscientious, extrovert, agreeable, neurotic):
    pleasure = 0.12 * extrovert + 0.59 * agreeable - 0.19 * neurotic
    arousal = 0.15 * openness + 0.3 * agreeable + 0.57 * neurotic
    dominance = 0.25 * openness + 0.17 * conscientious + 0.6 * extrovert - 0.32 * agreeable
    return pleasure, arousal, dominance


def summarize_personality(openness, conscientious, extrovert, agreeable, neurotic):
    model = FallbackMistralLLM()
    personality_str = "\n".join(
        [
            f"Openness: {round((openness + 1) * 50)}/100",
            f"Conscientiousness: {round((conscientious + 1) * 50)}/100",
            f"Extroversion: {round((extrovert + 1) * 50)}/100",
            f"Agreeableness: {round((agreeable + 1) * 50)}/100",
            f"Emotional Stability: {round((1 - neurotic) * 50)}/100",
        ]
    )
    return model.generate(SUMMARIZE_PERSONALITY.format(personality_values=personality_str), temperature=0.1)


class PersonalitySystem:
    def __init__(self, openness, conscientious, extrovert, agreeable, neurotic):
        self.open = openness
        self.conscientious = conscientious
        self.extrovert = extrovert
        self.agreeable = agreeable
        self.neurotic = neurotic
        self.summary = ""

    def get_summary(self):
        if not self.summary:
            self.summary = summarize_personality(
                self.open,
                self.conscientious,
                self.extrovert,
                self.agreeable,
                self.neurotic,
            )
        return self.summary


class Emotion:
    def __init__(self, pleasure=0.0, arousal=0.0, dominance=0.0):
        self.pleasure = pleasure
        self.arousal = arousal
        self.dominance = dominance

    @classmethod
    def from_personality(cls, openness, conscientious, extrovert, agreeable, neurotic):
        return cls(*get_default_mood(openness, conscientious, extrovert, agreeable, neurotic))

    def __add__(self, other):
        if isinstance(other, Emotion):
            return Emotion(self.pleasure + other.pleasure, self.arousal + other.arousal, self.dominance + other.dominance)
        return NotImplemented

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Emotion):
            self.pleasure += other.pleasure
            self.arousal += other.arousal
            self.dominance += other.dominance
            return self
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Emotion(self.pleasure * other, self.arousal * other, self.dominance * other)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Emotion(self.pleasure / other, self.arousal / other, self.dominance / other)
        return NotImplemented

    def dot(self, other):
        return self.pleasure * other.pleasure + self.arousal * other.arousal + self.dominance * other.dominance

    def get_intensity(self):
        return math.sqrt(self.pleasure**2 + self.arousal**2 + self.dominance**2) / math.sqrt(3)

    def get_norm(self):
        return max(abs(self.pleasure), abs(self.arousal), abs(self.dominance))

    def clamp(self):
        norm = self.get_norm()
        if norm > 1:
            self.pleasure /= norm
            self.arousal /= norm
            self.dominance /= norm

    def copy(self):
        return Emotion(self.pleasure, self.arousal, self.dominance)


class RelationshipSystem:
    def __init__(self):
        self.friendliness = 0.0
        self.dominance = 0.0
        self.interaction_count = 0

    def set_relation(self, friendliness=None, dominance=None):
        if friendliness is not None:
            self.friendliness = max(-100.0, min(100.0, friendliness))
        if dominance is not None:
            self.dominance = max(-100.0, min(100.0, dominance))

    def tick(self, dt):
        num_days = dt / 86400.0
        self.friendliness *= math.exp(-num_days / 90.0)
        self.dominance *= math.exp(-num_days / 120.0)

    def on_user_message(self, text):
        text = str(text or "").strip()
        if not text:
            return
        self.interaction_count += 1
        friendliness_delta = 0.0
        dominance_delta = 0.0
        if any(token in text for token in ("谢谢", "喜欢", "信任", "陪我", "晚安", "辛苦了")):
            friendliness_delta += 1.8
        if any(token in text for token in ("讨厌", "闭嘴", "滚", "烦死了", "不喜欢你")):
            friendliness_delta -= 3.2
        if any(token in text for token in ("命令", "立刻", "必须", "现在就")):
            dominance_delta += 1.2
        self.set_relation(self.friendliness + friendliness_delta, self.dominance + dominance_delta)

    def on_emotion(self, emotion, intensity):
        if emotion in {"Gratitude", "HappyFor", "Joy", "Love"}:
            self.friendliness = min(100.0, self.friendliness + 8 * intensity)
        elif emotion in {"Anger", "Reproach", "Hate"}:
            self.friendliness = max(-100.0, self.friendliness - 9 * intensity)

    def get_string(self):
        return f"Friendliness: {self.friendliness:.2f}\nDominance: {self.dominance:.2f}"

    def print_relation(self):
        string = val_to_symbol_color(self.friendliness / 100.0, 20, Fore.green, Fore.red)
        print(f"友好度：{string}")
        string = val_to_symbol_color(self.dominance / 100.0, 20, Fore.cyan, Fore.light_magenta)
        print(f"支配度：{string}")


class EmotionSystem:
    def __init__(self, personality_system, relation, config):
        self.personality_system = personality_system
        self.relation = relation
        self.config = config
        self.base_mood = Emotion.from_personality(
            personality_system.open,
            personality_system.conscientious,
            personality_system.extrovert,
            personality_system.agreeable,
            personality_system.neurotic,
        )
        self.mood = self.base_mood.copy()
        self.model = FallbackMistralLLM()

    def appraisal(self, messages, memories, beliefs):
        try:
            context = EMOTION_APPRAISAL_CONTEXT_TEMPLATE.format(
                conversation=conversation_to_string(messages[-4:]),
                memories=format_memories_to_string(memories, "None"),
                beliefs="\n".join(f"- {belief}" for belief in beliefs) if beliefs else "None",
            )
            data = self.model.generate(
                APPRAISAL_PROMPT.format(context=context),
                temperature=0.1,
                return_json=True,
                schema=APPRAISAL_SCHEMA,
            )
            items = data.get("appraisal", [])
        except Exception:
            return []
        result = []
        for item in items:
            emotion = str(item.get("emotion", "") or "")
            intensity = float(item.get("intensity", 0.0) or 0.0)
            if emotion in EMOTION_MAP and intensity > 0:
                result.append((emotion, max(0.0, min(1.0, intensity))))
        return result[:4]

    def set_emotion(self, pleasure=None, arousal=None, dominance=None):
        if pleasure is not None:
            self.mood.pleasure = max(-1.0, min(1.0, pleasure))
        if arousal is not None:
            self.mood.arousal = max(-1.0, min(1.0, arousal))
        if dominance is not None:
            self.mood.dominance = max(-1.0, min(1.0, dominance))

    def add_emotion(self, emotion):
        if isinstance(emotion, Emotion):
            self.mood += emotion
            self.clamp_mood()

    def apply_user_signal(self, text):
        text = str(text or "").strip()
        if not text:
            return
        delta = Emotion()
        if any(token in text for token in ("开心", "高兴", "喜欢", "谢谢", "期待", "安心", "太好了")):
            delta += Emotion(0.035, 0.02, -0.005)
        if any(token in text for token in ("难过", "伤心", "失落", "低落", "沮丧", "委屈", "孤独", "累")):
            delta += Emotion(-0.03, -0.02, -0.015)
        if any(token in text for token in ("讨厌你", "烦死了", "闭嘴", "滚", "不喜欢你", "讽刺", "恶心", "失望")):
            delta += Emotion(-0.045, 0.03, 0.025)
        if delta.get_intensity() > 0:
            self.mood += delta
            self.clamp_mood()

    def reset_mood(self):
        self.mood = self.get_base_mood()

    def _get_adv(self, val):
        if abs(val) > 0.9:
            return "extremely"
        if abs(val) > 0.65:
            return "very"
        if abs(val) > 0.35:
            return "moderately"
        return "slightly"

    def _get_mood_word(self, val, pos_str, neg_str):
        if abs(val) < 0.04:
            return "neutral"
        return self._get_adv(val) + " " + (pos_str if val >= 0 else neg_str)

    def get_mood_long_description(self):
        mood = self.mood
        pleasure_desc = f"Pleasure: {num_to_str_sign(mood.pleasure, 2)} ({self._get_mood_word(mood.pleasure, 'pleasant', 'unpleasant')})"
        arousal_desc = f"Arousal: {num_to_str_sign(mood.arousal, 2)} ({self._get_mood_word(mood.arousal, 'energized', 'soporific')})"
        dominance_desc = f"Dominance: {num_to_str_sign(mood.dominance, 2)} ({self._get_mood_word(mood.dominance, 'dominant', 'submissive')})"
        return "\n".join([pleasure_desc, arousal_desc, dominance_desc])

    def print_mood(self):
        mood = self.mood
        print("心境：")
        print("--------")
        print(f"愉悦度：{val_to_symbol_color(mood.pleasure, 20, Fore.green, Fore.red)}")
        print(f"唤醒度：{val_to_symbol_color(mood.arousal, 20, Fore.yellow, Fore.cornflower_blue)}")
        print(f"支配度：{val_to_symbol_color(mood.dominance, 20, Fore.cyan, Fore.light_magenta)}")

    def get_mood_name(self):
        mood = self.mood
        if mood.get_intensity() < 0.05:
            return "neutral"
        if mood.pleasure >= 0:
            return "exuberant" if mood.arousal >= 0 else "relaxed"
        return "hostile" if mood.arousal >= 0 else "bored"

    def get_mood_description(self):
        mood_name = self.get_mood_name()
        if mood_name != "neutral":
            mood_name = self._get_adv(self.mood.get_intensity()) + " " + mood_name
        return mood_name

    def get_mood_prompt(self):
        return f"{self.get_mood_description()} - {EMOTION_PROMPTS[self.get_mood_name()]}"

    def experience_emotion(self, name, intensity):
        emotion = Emotion(*EMOTION_MAP[name])
        mood_align = emotion.dot(self.mood)
        personality_align = emotion.dot(self.get_base_mood())
        intensity *= 1.0 + (MODD_INTENSITY_FACTOR * mood_align + PERSONALITY_INTENSITY_FACTOR * personality_align)
        intensity = max(0.02, min(0.55, intensity))
        self.relation.on_emotion(name, intensity)
        emotion *= intensity
        self.mood += emotion
        return emotion

    def clamp_mood(self):
        self.mood.clamp()

    def get_base_mood(self):
        base_mood = self.base_mood.copy()
        base_mood.pleasure += self.relation.friendliness / 100.0
        base_mood.dominance += self.relation.dominance / 100.0
        base_mood.clamp()
        return base_mood

    def tick(self, dt=None):
        dt = 0.0 if dt is None else float(dt)
        self.relation.tick(dt)
        decay = 0.5 ** (dt / MOOD_HALF_LIFE) if dt > 0 else 1.0
        base = self.get_base_mood()
        blend = 1 - decay
        self.mood.pleasure += (base.pleasure - self.mood.pleasure) * blend
        self.mood.arousal += (base.arousal - self.mood.arousal) * blend
        self.mood.dominance += (base.dominance - self.mood.dominance) * blend
        self.clamp_mood()


class EmotionSignal(BaseModel):
    mood: str = "平静"
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)


class EmotionState(BaseModel):
    mood: str = "平静"
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    updated_at: datetime = Field(default_factory=datetime.now)


class EmotionStateMachine:
    def __init__(self):
        self.baseline = EmotionState()
        self.current_state = EmotionState()
        self.pending_signal = EmotionSignal()

    def queue_signal(self, signal: EmotionSignal) -> None:
        self.pending_signal = signal

    def update_from_thought(self, thought_emotion: EmotionSignal) -> EmotionState:
        valence = thought_emotion.valence * 0.7 + self.pending_signal.valence * 0.3
        intensity = min(1.0, thought_emotion.intensity * 0.7 + self.pending_signal.intensity * 0.3)
        self.current_state = EmotionState(
            mood=thought_emotion.mood or self.pending_signal.mood or self.baseline.mood,
            intensity=intensity,
            valence=valence,
            updated_at=datetime.now(),
        )
        self.pending_signal = EmotionSignal()
        self._drift_to_baseline()
        return self.current_state

    def _drift_to_baseline(self) -> None:
        self.current_state.valence = self.current_state.valence * 0.9 + self.baseline.valence * 0.1
        self.current_state.intensity = self.current_state.intensity * 0.9 + self.baseline.intensity * 0.1

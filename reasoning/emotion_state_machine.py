from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, Field

from const import EMOTION_MAP, EMOTION_PROMPTS, MODD_INTENSITY_FACTOR, MOOD_HALF_LIFE, PERSONALITY_INTENSITY_FACTOR
from safe_colored import Fore
from utils import num_to_str_sign, val_to_symbol_color


def get_default_mood(openness, conscientious, extrovert, agreeable, neurotic):
    pleasure = 0.12 * extrovert + 0.59 * agreeable - 0.19 * neurotic
    arousal = 0.15 * openness + 0.3 * agreeable + 0.57 * neurotic
    dominance = 0.25 * openness + 0.17 * conscientious + 0.6 * extrovert - 0.32 * agreeable
    return pleasure, arousal, dominance


def summarize_personality(openness, conscientious, extrovert, agreeable, neurotic):
    traits: list[str] = []
    if extrovert >= 0.35:
        traits.append("表达主动")
    elif extrovert <= -0.2:
        traits.append("表达偏内收")
    if agreeable >= 0.45:
        traits.append("整体温和")
    elif agreeable <= -0.1:
        traits.append("立场偏硬")
    if openness >= 0.25:
        traits.append("愿意接受新话题")
    if conscientious >= 0.2:
        traits.append("表达较有条理")
    if neurotic >= 0.25:
        traits.append("容易在细节上紧张")
    elif neurotic <= 0:
        traits.append("情绪相对稳定")
    return "、".join(traits) if traits else "表达自然"


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

    def set_relation(self, friendliness=None, dominance=None):
        if friendliness is not None:
            self.friendliness = max(-100.0, min(100.0, friendliness))
        if dominance is not None:
            self.dominance = max(-100.0, min(100.0, dominance))

    def tick(self, dt):
        num_days = dt / 86400.0
        self.friendliness *= math.exp(-num_days / 90.0)
        self.dominance *= math.exp(-num_days / 120.0)

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

    def appraisal(self, messages, memories, beliefs):
        return []

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
    trust_delta: float = Field(default=0.0, ge=-0.08, le=0.08)
    affection_delta: float = Field(default=0.0, ge=-0.08, le=0.08)
    familiarity_delta: float = Field(default=0.0, ge=-0.04, le=0.04)
    rationale: str = ""


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
        chosen_mood = thought_emotion.mood or self.pending_signal.mood or self.baseline.mood
        pending_stronger = (
            self.pending_signal.intensity >= max(0.12, thought_emotion.intensity * 0.9)
            and abs(self.pending_signal.valence) >= abs(thought_emotion.valence) + 0.05
        )
        if pending_stronger and str(self.pending_signal.mood or "").strip():
            chosen_mood = self.pending_signal.mood
        self.current_state = EmotionState(
            mood=chosen_mood,
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

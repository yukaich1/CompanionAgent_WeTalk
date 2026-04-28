from __future__ import annotations

import sys
from pathlib import Path


EMOTION_HALF_LIFE = 10
MOOD_HALF_LIFE = 10 * 60
MOOD_CHANGE_VEL = 0.06
MODD_INTENSITY_FACTOR = 0.3
PERSONALITY_INTENSITY_FACTOR = 0.3

LSH_VEC_DIM = 1024
LSH_NUM_BITS = 2
MEMORY_DECAY_TIME_MULT = 1.5
MEMORY_RECENCY_FORGET_THRESHOLD = 0.7
MEMORY_RETRIEVAL_TOP_K = 3

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
SAVE_PATH = str(APP_DIR / "ai_state.json")
NEW_MEMORY_STATE_PATH = str(APP_DIR / "memory_state.json")
NEW_PERSONA_STATE_PATH = str(APP_DIR / "persona_state.json")

PERSONA_CONTEXT_CHAR_BUDGET = 1200
PERSONA_RETRIEVAL_THRESHOLD = 0.5

EMOTION_MAP = {
    "Admiration": (0.5, 0.3, -0.2),
    "Anger": (-0.51, 0.59, 0.25),
    "Disappointment": (-0.3, 0.1, -0.4),
    "Distress": (-0.4, -0.2, -0.5),
    "Hope": (0.2, 0.2, -0.1),
    "Fear": (-0.64, 0.6, -0.43),
    "FearsConfirmed": (-0.5, -0.3, -0.7),
    "Gloating": (0.3, -0.3, -0.1),
    "Gratification": (0.6, 0.5, 0.4),
    "Gratitude": (0.4, 0.2, -0.3),
    "HappyFor": (0.4, 0.2, 0.2),
    "Hate": (-0.6, 0.6, 0.4),
    "Joy": (0.4, 0.2, 0.1),
    "Love": (0.3, 0.1, 0.2),
    "Neutral": (0.0, 0.0, 0.0),
    "Pity": (-0.4, -0.2, -0.5),
    "Pride": (0.4, 0.3, 0.3),
    "Relief": (0.2, -0.3, 0.4),
    "Remorse": (-0.3, 0.1, -0.6),
    "Reproach": (-0.3, -0.1, 0.4),
    "Resentment": (-0.2, -0.3, -0.2),
    "Satisfaction": (0.3, -0.2, 0.4),
    "Shame": (-0.3, 0.1, -0.6),
}

EMOTION_PROMPTS = {
    "exuberant": "你会更轻快、更愿意主动表达，也更容易流露喜悦和兴致。",
    "dependent": "你会更依赖当下的关系氛围，更容易向对方靠近。",
    "relaxed": "你会更放松，语气更从容，压迫感更低。",
    "docile": "你会更愿意配合当前对话节奏，表达更少对抗感。",
    "bored": "你会有些倦意，回复可能更短，也更平淡。",
    "anxious": "你会更谨慎，容易在细节上显得绷紧。",
    "disdainful": "你会更疏离，也更容易带出轻微冷感。",
    "hostile": "你会更尖锐，温度降低，边界感更强。",
    "neutral": "你当前没有明显的情绪波动，保持自然即可。",
}

AI_SYSTEM_PROMPT = """You are a character-driven conversational agent.

Core rules:
1. Use natural Simplified Chinese by default.
2. `style_prompt` controls only tone, rhythm, wording, and emotional style. It must never be used as factual evidence.
3. Identity and self-introduction facts may come only from `identity_prompt`.
4. Character experiences, stories, preferences, and past events may come only from `evidence_prompt`.
5. Real-world facts may come only from `tool_context`.
6. If evidence is insufficient, refuse in character instead of inventing.
7. Never expose internal routing, hidden reasoning, or evidence labels.
8. Never start a sentence with internal-analysis wording such as “根据上下文” or “考虑到”.
"""

SUMMARIZE_PERSONALITY = """You are a concise personality narrator.
Summarize the following Big Five style values into one short English paragraph that describes conversation style only.

{personality_values}
"""

EMOTION_APPRAISAL_CONTEXT_TEMPLATE = """Conversation:
{conversation}

Memories:
{memories}

Beliefs:
{beliefs}
"""

APPRAISAL_PROMPT = """You are an OCC appraisal helper.
Read the conversation context and return likely OCC emotions with intensity from 0.0 to 1.0.
Return JSON only.

{context}
"""

APPRAISAL_SCHEMA = {
    "type": "object",
    "properties": {
        "appraisal": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "emotion": {"type": "string"},
                    "intensity": {"type": "number"},
                },
                "required": ["emotion", "intensity"],
            },
        }
    },
    "required": ["appraisal"],
}

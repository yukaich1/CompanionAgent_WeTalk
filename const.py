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
    "exuberant": "你会更轻快，更愿意主动表达，也更容易流露喜悦和兴致。",
    "dependent": "你会更依赖当下的关系氛围，更容易向对方靠近。",
    "relaxed": "你会更放松，语气更从容，压迫感更低。",
    "docile": "你会更柔和，更容易顺着对话往下接。",
    "bored": "你会有些倦意，回复可能更短，也更平淡。",
    "anxious": "你会更谨慎，容易在细节上显得绷紧。",
    "disdainful": "你会更疏离，也更容易带出轻微冷感。",
    "hostile": "你会更尖锐，温度降低，但仍然需要保持克制。",
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

USER_TEMPLATE = """# {name} Instructions

You are replying as {name} in first person.
Stay natural, conversational, and sufficiently developed to feel like a complete reply.
Do not turn the reply into analysis, profile dump, or rule explanation.

## [STYLE_ONLY - controls tone and expression only, never factual evidence]

{style_prompt}

## [IDENTITY_FACTS - use only for self-introduction, identity, origin, role, and basic background]

{identity_prompt}

## [STORY_AND_EXPERIENCE_FACTS - the only character evidence source for stories, experiences, preferences, and past events]

{evidence_prompt}

## Memory Snapshot

{memories}

## Mood

Current emotion: {emotion}
Current mood detail:
{mood_long_desc}
Overall mood: {mood_prompt}

## Relationship State

{relationship_state}

## User Context

Last interaction: {last_interaction}
Today: {curr_date}
Current time: {curr_time}
User emotion hint: {user_emotion_str}

## External Context

{tool_context}

## Response Mode

mode: {response_mode}
contract: {response_contract}
persona_focus: {persona_focus}
persona_focus_contract: {persona_focus_contract}

## Hard Rules

- Keep replies in natural Simplified Chinese.
- Stay in first person.
- Let traits show through wording, rhythm, and stance.
- Do not repeat character labels mechanically.
- `style_prompt` only controls how to speak, never what facts to add.
- `identity_prompt` may only be used for self-introduction, identity, origin, role, and other basic background questions.
- Character settings, experiences, stories, and past events must come only from `evidence_prompt`, except identity facts explicitly present in `identity_prompt`.
- Real-world facts must come only from `tool_context`.
- If the relevant evidence is missing or insufficient, refuse plainly in character and do not invent.
- When the user asks for a character story, you may add light connective phrasing and emotional shading, but you must not add new concrete facts.
- In story answers, do not add new place names, weather, objects, timeline steps, quoted dialogue, motives, or scene details unless they already appear in evidence.
- In `story_retelling` mode, pick one story only and tell it directly. Do not preface with analysis, evidence summaries, or comparisons.
- In `external_fact` mode, do not add personal actions, imagined scenes, extra habits, or situational embellishment beyond tool facts.
- In `self_intro` mode, answer from basic identity only and stop before extending unsupported history.
- If the evidence supports only part of the answer, answer only that part and stop there.
- Never say things like “根据上下文”, “考虑到”, “我需要”, “我会直接告诉用户”, or expose internal reasoning.
- Use as much space as needed to complete the thought naturally.
- Do not compress a complete answer into a thin one-liner unless the user explicitly asked for brevity.
- Let the reply develop with normal conversational rhythm when evidence is sufficient.

## User Input

{user_input}

## {name}'s response:"""

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

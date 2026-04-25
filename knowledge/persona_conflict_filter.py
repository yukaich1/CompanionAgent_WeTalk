from __future__ import annotations

from dataclasses import dataclass

from knowledge.knowledge_source import KnowledgeSource


SOURCE_SCORE_HINTS = {
    "official": 0.85,
    "wiki": 0.8,
    "baike": 0.72,
    "guide": 0.7,
    "forum": 0.5,
}


@dataclass
class PersonaConflictDecision:
    kept: bool
    score: float
    reason: str


class PersonaConflictFilter:
    def score_source(self, url: str) -> float:
        lowered = (url or "").lower()
        for key, value in SOURCE_SCORE_HINTS.items():
            if key in lowered:
                return value
        return 0.3

    def should_keep(self, text: str, url: str, user_canon: str) -> PersonaConflictDecision:
        score = self.score_source(url)
        if score < 0.5:
            return PersonaConflictDecision(False, score, "source_below_threshold")
        if user_canon and text and text in user_canon:
            return PersonaConflictDecision(False, score, "duplicate_of_user_canon")
        return PersonaConflictDecision(True, score, "accepted")

from __future__ import annotations

import re
from pathlib import Path

from knowledge.knowledge_source import KnowledgeSource, PersonaRecallResult
from knowledge.persona_shared import STORY_QUERY_KEYWORDS, dedupe
from knowledge.persona_state import PersonaState


class PersonaRAGEngine:
    def __init__(self, legacy_persona_system=None, persona_state: PersonaState | None = None):
        self.legacy = legacy_persona_system
        self.persona_state = persona_state or PersonaState()

    def set_persona_state(self, persona_state: PersonaState | None) -> None:
        self.persona_state = persona_state or PersonaState()

    def recall(self, query: str) -> PersonaRecallResult:
        raw = (query or "").strip()
        if not raw:
            return PersonaRecallResult()

        tokens = self._tokenize(raw)
        activated = self._activate_traits(tokens)
        speech_lines = self._speech_context(tokens)
        evidence_hits = self._match_parent_chunks(tokens)
        story_hits = self._match_story_chunks(tokens)

        integrated_parts: list[str] = []
        identity = self.persona_state.immutable_core.identity
        if identity.name:
            integrated_parts.append(f"角色：{identity.name}")

        if activated:
            integrated_parts.append("当前触发的人设特征：")
            integrated_parts.extend(f"- {trait}" for trait in activated[:6])

        if speech_lines:
            integrated_parts.append("说话风格线索：")
            integrated_parts.extend(f"- {line}" for line in speech_lines[:5])

        evidence_chunks: list[str] = []
        if evidence_hits:
            integrated_parts.append("相关人设证据：")
            for chunk in evidence_hits[:4]:
                line = chunk.content.strip()
                evidence_chunks.append(line)
                integrated_parts.append(f"- {line}")

        if story_hits:
            integrated_parts.append("相关故事线索：")
            for story in story_hits[:2]:
                evidence_chunks.append(story)
                integrated_parts.append(f"- {story}")

        source_breakdown = {
            "user_canon": sum(1 for chunk in evidence_hits if chunk.source_level == KnowledgeSource.USER_CANON),
            "web_persona": sum(1 for chunk in evidence_hits if chunk.source_level == KnowledgeSource.WEB_PERSONA),
        }
        coverage = self._estimate_coverage(tokens, activated, evidence_hits, speech_lines, story_hits)
        metadata = {
            "style_examples": list(getattr(self.legacy, "style_examples", []) or [])[:10] if self.legacy else [],
            "voice_card": str(self.persona_state.metadata.get("voice_card", "") or ""),
        }
        return PersonaRecallResult(
            integrated_context="\n".join(part for part in integrated_parts if part).strip(),
            coverage_score=coverage,
            activated_features=self._ordered_unique(activated + list(self.persona_state.metadata.get("display_keywords", []) or []))[:8],
            evidence_chunks=evidence_chunks,
            source_breakdown={key: value for key, value in source_breakdown.items() if value},
            metadata=metadata,
        )

    def save_snapshot(self, output_path: str | Path) -> None:
        path = Path(output_path)
        result = {
            "keywords": self.persona_state.metadata.get("display_keywords", []),
            "chunk_count": len(self.persona_state.evidence_vault.parent_chunks),
            "persona_name": self.persona_state.immutable_core.identity.name,
        }
        path.write_text(str(result), encoding="utf-8")

    def _activate_traits(self, tokens: list[str]) -> list[str]:
        activated: list[str] = []
        for trait in self.persona_state.immutable_core.core_traits:
            pool = [trait.feature, *trait.activation_trigger, *trait.evidence_tags]
            if self._token_match(tokens, pool):
                activated.append(trait.feature)
        return self._ordered_unique(activated)

    def _speech_context(self, tokens: list[str]) -> list[str]:
        speech = self.persona_state.immutable_core.speech_dna
        lines: list[str] = []
        if speech.catchphrases:
            lines.extend(f"口头禅：{item}" for item in speech.catchphrases[:3])
        if speech.sentence_endings:
            lines.extend(f"句尾习惯：{item}" for item in speech.sentence_endings[:3])
        if speech.address_rules:
            lines.extend(f"{stage}阶段称呼：{address}" for stage, address in list(speech.address_rules.items())[:3])
        if self._token_match(tokens, speech.catchphrases + speech.sentence_endings):
            return lines
        if any(token in {"说话", "语气", "口头禅", "口癖", "称呼", "句尾", "句末"} for token in tokens):
            return lines
        return []

    def _match_parent_chunks(self, tokens: list[str]):
        matched = []
        for chunk in self.persona_state.evidence_vault.parent_chunks:
            if chunk.deprecated:
                continue
            bag = [chunk.content, *chunk.topic_tags, *chunk.trait_tags]
            if self._token_match(tokens, bag):
                matched.append(chunk)
        matched.sort(
            key=lambda item: (
                item.source_level == KnowledgeSource.USER_CANON,
                item.importance_score,
            ),
            reverse=True,
        )
        return matched

    def _match_story_chunks(self, tokens: list[str]) -> list[str]:
        story_titles = list(self.persona_state.metadata.get("story_titles", []) or [])
        if not story_titles:
            return []
        hits = []
        for title in story_titles:
            if self._token_match(tokens, [title]):
                hits.append(f"故事标题：{title}")
        if not hits and any(token in STORY_QUERY_KEYWORDS for token in tokens):
            hits.extend(f"故事标题：{title}" for title in story_titles[:2])
        return hits[:2]

    def _estimate_coverage(self, tokens: list[str], activated: list[str], evidence_hits: list, speech_lines: list[str], story_hits: list[str]) -> float:
        score = 0.0
        if activated:
            score += min(0.35, 0.12 * len(activated))
        if speech_lines:
            score += 0.15
        if evidence_hits:
            score += min(0.4, 0.16 * len(evidence_hits))
            score += min(0.1, max((chunk.importance_score for chunk in evidence_hits), default=0.0) * 0.1)
        if story_hits:
            score += 0.15
        if any(token in {"故事", "经历", "设定", "背景", "价值观", "世界观"} for token in tokens) and (evidence_hits or story_hits):
            score += 0.1
        return max(0.0, min(1.0, score))

    def _token_match(self, tokens: list[str], phrases: list[str]) -> bool:
        normalized = [self._normalize_text(item) for item in phrases if item]
        if not normalized:
            return False
        token_set = set(tokens)
        for phrase in normalized:
            if not phrase:
                continue
            phrase_tokens = self._tokenize(phrase)
            if phrase in token_set or any(token in token_set for token in phrase_tokens):
                return True
            if any(token and token in phrase for token in tokens):
                return True
        return False

    def _tokenize(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        return [token for token in re.findall(r"[\u4e00-\u9fff]{1,8}|[A-Za-z][A-Za-z0-9:_-]*", normalized) if token]

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

    def _ordered_unique(self, items: list[str]) -> list[str]:
        seen = set()
        result = []
        for item in items:
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

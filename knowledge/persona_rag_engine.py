from __future__ import annotations

import re
from pathlib import Path

from knowledge.knowledge_source import KnowledgeSource, PersonaRecallResult
from knowledge.persona_system import PersonaState


class PersonaRAGEngine:
    """人设召回入口。

    优先使用新的 PersonaState 结构化状态进行特征激活和证据块召回，
    同时保留旧 persona_system 的上下文构建能力作为兼容兜底。
    """

    def __init__(self, legacy_persona_system=None, persona_state: PersonaState | None = None):
        if legacy_persona_system is None:
            from knowledge.persona_system import PersonaSystem as LegacyPersonaSystem

            legacy_persona_system = LegacyPersonaSystem()
        self.legacy = legacy_persona_system
        self.persona_state = persona_state or PersonaState()

    def set_persona_state(self, persona_state: PersonaState | None) -> None:
        self.persona_state = persona_state or PersonaState()

    def recall(self, query: str) -> PersonaRecallResult:
        structured = self._recall_from_state(query)
        if structured.coverage_score >= 0.35 or structured.integrated_context:
            return structured

        context = self.legacy.build_context(query)
        keywords = self.legacy.get_display_keywords()
        evidence = []
        story = self.legacy.build_story_context(query)
        if story:
            evidence.append(story)
        coverage = 0.75 if context else 0.0
        return PersonaRecallResult(
            integrated_context=context,
            coverage_score=coverage,
            activated_features=keywords,
            evidence_chunks=evidence,
            source_breakdown={"legacy_persona": 1 if context else 0},
        )

    def save_snapshot(self, output_path: str | Path) -> None:
        path = Path(output_path)
        result = {
            "keywords": self.persona_state.metadata.get("display_keywords", []) or self.legacy.get_display_keywords(),
            "chunk_count": len(self.persona_state.evidence_vault.parent_chunks) or self.legacy.chunk_count,
            "persona_name": self.persona_state.immutable_core.identity.name or self.legacy.persona_name,
        }
        path.write_text(str(result), encoding="utf-8")

    def _recall_from_state(self, query: str) -> PersonaRecallResult:
        raw = (query or "").strip()
        if not raw:
            return PersonaRecallResult()

        tokens = self._tokenize(raw)
        activated = self._activate_traits(tokens)
        evidence_hits = self._match_parent_chunks(tokens)
        speech_lines = self._speech_context(tokens)

        integrated_parts: list[str] = []
        if self.persona_state.immutable_core.identity.name:
            integrated_parts.append(f"角色：{self.persona_state.immutable_core.identity.name}")

        if activated:
            integrated_parts.append("本轮激活的人设特征：")
            integrated_parts.extend(f"- {trait}" for trait in activated[:6])

        if speech_lines:
            integrated_parts.append("语言风格线索：")
            integrated_parts.extend(f"- {line}" for line in speech_lines[:5])

        evidence_chunks: list[str] = []
        if evidence_hits:
            integrated_parts.append("相关人设证据：")
            for chunk in evidence_hits[:4]:
                line = chunk.content.strip()
                evidence_chunks.append(line)
                integrated_parts.append(f"- {line}")

        source_breakdown = {
            "user_canon": sum(1 for chunk in evidence_hits if chunk.source_level == KnowledgeSource.USER_CANON),
            "web_persona": sum(1 for chunk in evidence_hits if chunk.source_level == KnowledgeSource.WEB_PERSONA),
        }
        coverage = self._estimate_coverage(tokens, activated, evidence_hits, speech_lines)
        return PersonaRecallResult(
            integrated_context="\n".join(part for part in integrated_parts if part).strip(),
            coverage_score=coverage,
            activated_features=self._ordered_unique(activated + self.persona_state.metadata.get("display_keywords", []))[:8],
            evidence_chunks=evidence_chunks,
            source_breakdown={key: value for key, value in source_breakdown.items() if value},
        )

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
            lines.extend(f"句末习惯：{item}" for item in speech.sentence_endings[:3])
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

    def _estimate_coverage(self, tokens: list[str], activated: list[str], evidence_hits: list, speech_lines: list[str]) -> float:
        score = 0.0
        if activated:
            score += min(0.35, 0.12 * len(activated))
        if speech_lines:
            score += 0.15
        if evidence_hits:
            score += min(0.45, 0.16 * len(evidence_hits))
            score += min(0.1, max((chunk.importance_score for chunk in evidence_hits), default=0.0) * 0.1)
        if any(token in {"故事", "经历", "设定", "背景", "价值观", "世界观"} for token in tokens) and evidence_hits:
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
        return [token for token in re.findall(r"[\u4e00-\u9fff]{1,6}|[A-Za-z][A-Za-z0-9:_-]*", normalized) if token]

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

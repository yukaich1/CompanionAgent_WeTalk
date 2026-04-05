from __future__ import annotations

import random

import numpy as np

from const import PERSONA_CONTEXT_CHAR_BUDGET, PERSONA_RETRIEVAL_THRESHOLD
from knowledge.persona_shared import DIMENSION_QUERY_KEYWORDS, DIMENSION_TITLES_ZH, STORY_QUERY_KEYWORDS, dedupe
from llm import MistralEmbeddingCapacityError, mistral_embed_texts


class PersonaContextService:
    def __init__(self, system):
        self.system = system
        self.random = random.Random(7)

    def extract_query_terms(self, query: str) -> list[str]:
        text = self.system._normalize_fact(query, max_chars=120)
        stopwords = {"角色", "设定", "故事", "经历", "背景", "一个", "什么", "怎么", "这个", "那个", "有关", "关于"}
        terms = []
        for token in self.system.dynamic_keyword_re.findall(text):
            token = str(token).strip()
            if len(token) >= 2 and token not in stopwords:
                terms.append(token)
        return dedupe(terms, limit=12)

    def detect_relevant_dimensions(self, query: str) -> list[str]:
        query_norm = self.system._normalize_fact(query, max_chars=160)
        relevant = []
        for dim, keywords in DIMENSION_QUERY_KEYWORDS.items():
            if any(keyword in query_norm for keyword in keywords):
                relevant.append(dim)
        if self.is_story_query(query_norm):
            relevant.extend(["00_BACKGROUND_PROFILE", "16_NARRATIVE_STYLE"])
        return dedupe(relevant)

    def is_story_query(self, query: str) -> bool:
        text = self.system._normalize_fact(query, max_chars=120)
        return any(keyword in text for keyword in STORY_QUERY_KEYWORDS)

    def keyword_search_entries(self, query: str, top_k: int = 6, kinds: set[str] | None = None) -> list[dict]:
        if not query or not self.system.entries:
            return []
        terms = self.extract_query_terms(query)
        if not terms:
            return []

        scored = []
        for entry in self.system.entries:
            text = entry.get("text", "")
            kind = entry.get("kind", "")
            if kinds and kind not in kinds:
                continue
            if not text or self.system._is_meta_commentary(text):
                continue
            entry_keywords = entry.get("keywords", []) or []
            text_hits = sum(1 for term in terms if term in text)
            keyword_hits = sum(2 for term in terms if term in entry_keywords)
            if not text_hits and not keyword_hits:
                continue

            score = float(text_hits + keyword_hits)
            if kind == "story_chunk":
                score += 3.2
            elif kind == "base_template":
                score += 2.4
            elif kind == "core_summary":
                score += 1.6
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def search_entries(self, query: str, top_k: int = 6, kinds: set[str] | None = None) -> list[dict]:
        if not query or self.system.index is None or not self.system.entries:
            return []
        try:
            query_vector = self.system.normalize_vector(mistral_embed_texts(query))
        except MistralEmbeddingCapacityError:
            return []

        scores, indices = self.system.index.search(np.asarray([query_vector], dtype=np.float32), top_k * 3)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.system.entries) or score < PERSONA_RETRIEVAL_THRESHOLD:
                continue
            entry = self.system.entries[idx]
            if kinds and entry.get("kind", "") not in kinds:
                continue
            results.append((float(score) * float(entry.get("priority", 1.0)), entry))
        results.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in results[:top_k]]

    def _format_dimension_block(self, dim: str, dim_data: dict) -> list[str]:
        title = DIMENSION_TITLES_ZH.get(dim, dim)
        lines: list[str] = []
        if dim == "00_BACKGROUND_PROFILE":
            profile = dim_data.get("profile", {})
            key_experiences = dim_data.get("key_experiences", [])
            if profile:
                lines.append(f"【{title}·基础档案】")
                for key, value in list(profile.items())[:6]:
                    lines.append(f"- {key}: {value}")
            if key_experiences:
                lines.append(f"【{title}·关键经历】")
                lines.extend(f"- {item}" for item in key_experiences[:4])
            return lines
        if dim in {"17_LIKES_AND_PREFERENCES", "18_DISLIKES_AND_TABOOS"}:
            items = dim_data.get("items", [])
            if items:
                lines.append(f"【{title}】")
                for item in items[:3]:
                    if isinstance(item, dict):
                        label = item.get("item", "")
                        behavior = item.get("behavior", "")
                        level = item.get("level", "")
                        prefix = f"{label}（{level}）" if level else label
                        lines.append(f"- {prefix}" + (f": {behavior}" if behavior else ""))
                    else:
                        lines.append(f"- {item}")
            return lines
        if dim == "19_AVOID_PATTERNS":
            patterns = dim_data.get("patterns", [])
            if patterns:
                lines.append(f"【{title}】")
                for pattern in patterns[:4]:
                    text = pattern.get("pattern", "") if isinstance(pattern, dict) else str(pattern)
                    if text:
                        lines.append(f"- 不会说：{text}")
            return lines
        rules = dim_data.get("rules", [])
        if rules:
            lines.append(f"【{title}】")
            lines.extend(f"- {rule}" for rule in rules[:3])
        return lines

    def _score_story_chunk(self, query_terms: list[str], chunk: dict) -> float:
        score = 0.0
        text = str(chunk.get("content", "") or "")
        title = str(chunk.get("title", "") or "")
        keywords = list(chunk.get("keywords", []) or [])
        triggers = list(chunk.get("trigger_topics", []) or [])
        impact = str(chunk.get("character_impact", "") or "")
        score += 0.9 * sum(1 for term in query_terms if term in text)
        score += 1.8 * sum(1 for term in query_terms if term in keywords)
        score += 1.2 * sum(1 for term in query_terms if term in triggers)
        score += 0.7 * sum(1 for term in query_terms if term in title)
        score += 0.5 * sum(1 for term in query_terms if term in impact)
        if impact:
            score += 0.15
        return score

    def _collect_story_candidates(self, query: str, top_k: int = 6) -> list[dict]:
        query_terms = self.extract_query_terms(query)
        candidates = []
        seen = set()
        for chunk in getattr(self.system, "story_chunks", []) or []:
            content = str(chunk.get("content", "") or "").strip()
            key = (str(chunk.get("title", "") or "").strip(), content)
            if not content or key in seen or self.system._is_meta_commentary(content):
                continue
            seen.add(key)
            candidates.append((self._score_story_chunk(query_terms, chunk), chunk))
        entry_pool = self.keyword_search_entries(query, top_k=top_k * 3, kinds={"story_chunk"}) + self.search_entries(query, top_k=top_k * 3, kinds={"story_chunk"})
        for entry in entry_pool:
            text = str(entry.get("text", "") or "").strip()
            if not text or text in seen or self.system._is_meta_commentary(text):
                continue
            seen.add(text)
            pseudo_chunk = {
                "title": entry.get("title", entry.get("source_label", "角色经历")),
                "content": text,
                "keywords": list(entry.get("keywords", []) or []),
                "trigger_topics": list(entry.get("trigger_topics", []) or []),
                "character_impact": str(entry.get("character_impact", "") or ""),
                "emotional_weight": str(entry.get("emotional_weight", "") or "medium"),
            }
            candidates.append((self._score_story_chunk(query_terms, pseudo_chunk) + 0.4, pseudo_chunk))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in candidates[:top_k]]

    def _select_story_chunk(self, query: str) -> dict | None:
        candidates = self._collect_story_candidates(query, top_k=5)
        if not candidates:
            return None
        query_terms = self.extract_query_terms(query)
        scored = [(self._score_story_chunk(query_terms, chunk), chunk) for chunk in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        close = [chunk for score, chunk in scored if score >= top_score - 0.6]
        return self.random.choice(close[:3]) if close else scored[0][1]

    def _story_chunk_to_prompt_text(self, chunk: dict) -> str:
        lines = [f"【故事标题】{chunk.get('title', '角色经历')}"]
        content = str(chunk.get("content", "") or "").strip()
        if content:
            lines.append(f"【故事内容】{content}")
        keywords = list(chunk.get("keywords", []) or [])
        if keywords:
            lines.append("【故事关键词】" + "、".join(keywords[:6]))
        triggers = list(chunk.get("trigger_topics", []) or [])
        if triggers:
            lines.append("【自然触发话题】" + "、".join(triggers[:4]))
        impact = str(chunk.get("character_impact", "") or "").strip()
        if impact:
            lines.append(f"【对角色的影响】{impact}")
        weight = str(chunk.get("emotional_weight", "") or "").strip()
        if weight:
            lines.append(f"【情感权重】{weight}")
        return "\n".join(lines)

    def build_context(self, query: str) -> str:
        query = self.system._normalize_fact(query, max_chars=180)
        if not query:
            return ""
        if self.is_story_query(query):
            story_context = self.build_story_context(query)
            if story_context:
                return story_context
        return self.build_precise_query_context(query, top_k=5, char_budget=PERSONA_CONTEXT_CHAR_BUDGET)

    def build_precise_query_context(self, query: str, top_k: int = 5, char_budget: int = 700) -> str:
        if not query:
            return ""
        relevant_dims = self.detect_relevant_dimensions(query)
        local_lines = []
        for dim in relevant_dims:
            local_lines.extend(self._format_dimension_block(dim, self.system.base_template.get(dim, {})))
        query_terms = self.extract_query_terms(query)
        search_pool = self.keyword_search_entries(query, top_k=top_k * 3) + self.search_entries(query, top_k=top_k * 3)
        seen = set()
        scored = []
        for entry in search_pool:
            text = entry.get("text", "")
            if not text or text in seen or self.system._is_meta_commentary(text):
                continue
            seen.add(text)
            kind = entry.get("kind", "")
            score = 0.0
            if kind == "base_template":
                score += 3.0
            elif kind == "story_chunk":
                score += 2.2
            elif kind == "core_summary":
                score += 1.4
            score += 0.35 * sum(1 for term in query_terms if term in text)
            score += 0.8 * sum(1 for term in query_terms if term in entry.get("keywords", []))
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        evidence_lines = [f"[{entry.get('source_label', 'source')}] {entry.get('text', '')}" for _, entry in scored[:top_k]]
        blocks = []
        if local_lines:
            blocks.append("精准角色规则：\n" + "\n".join(local_lines))
        if evidence_lines:
            blocks.append("精准角色证据：\n" + "\n".join(evidence_lines))
        context = "\n\n".join(blocks).strip()
        if len(context) > char_budget:
            context = context[: char_budget - 3].rstrip() + "..."
        return context

    def build_story_context(self, query: str, top_k: int = 1, char_budget: int = 900) -> str:
        selected = self._select_story_chunk(query)
        if not selected:
            return ""
        context = self._story_chunk_to_prompt_text(selected).strip()
        if len(context) > char_budget:
            context = context[: char_budget - 3].rstrip() + "..."
        return context

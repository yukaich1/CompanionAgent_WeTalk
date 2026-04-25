from __future__ import annotations


class ContextCompactor:
    def compact_topics(self, topics: list[str] | None, *, limit: int = 6) -> list[str]:
        seen: set[str] = set()
        compacted: list[str] = []
        for item in list(topics or []):
            value = str(item or "").strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            compacted.append(value)
            if len(compacted) >= limit:
                break
        return compacted

    def compact_thread_summaries(self, summaries: list[str] | None, *, limit: int = 4) -> list[str]:
        compacted: list[str] = []
        for item in list(summaries or []):
            value = str(item or "").strip()
            if not value:
                continue
            compacted.append(value[:220].rstrip())
            if len(compacted) >= limit:
                break
        return compacted

    def archive_thread_summaries(
        self,
        active: list[str] | None,
        archived: list[str] | None,
        *,
        active_limit: int = 4,
        archive_limit: int = 8,
    ) -> tuple[list[str], list[str]]:
        active_values = [str(item or "").strip() for item in list(active or []) if str(item or "").strip()]
        archived_values = [str(item or "").strip() for item in list(archived or []) if str(item or "").strip()]
        kept_active = self.compact_thread_summaries(active_values, limit=active_limit)
        overflow = [
            value[:220].rstrip()
            for value in active_values[active_limit:]
            if value[:220].rstrip()
        ]
        merged_archive = self.compact_thread_summaries([*overflow, *archived_values], limit=archive_limit)
        return kept_active, merged_archive

    def compact_pinned_facts(self, facts: list[str] | None, *, limit: int = 8) -> list[str]:
        seen: set[str] = set()
        compacted: list[str] = []
        for item in list(facts or []):
            value = str(item or "").strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            compacted.append(value[:120].rstrip())
            if len(compacted) >= limit:
                break
        return compacted

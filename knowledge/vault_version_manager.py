from __future__ import annotations

from difflib import SequenceMatcher

from knowledge.knowledge_source import KnowledgeSource
from knowledge.persona_system import ChildChunk, EvidenceVault, ParentChunk


class VaultVersionManager:
    def overlap(self, left: str, right: str) -> float:
        return SequenceMatcher(None, left or "", right or "").ratio()

    def update_parent_chunks(self, vault: EvidenceVault, new_chunks: list[ParentChunk]) -> EvidenceVault:
        for incoming in new_chunks:
            matched = None
            for existing in vault.parent_chunks:
                if self.overlap(existing.content, incoming.content) > 0.9:
                    matched = existing
                    break
            if matched is None:
                vault.parent_chunks.append(incoming)
                continue
            if matched.source_level == KnowledgeSource.USER_CANON and incoming.source_level == KnowledgeSource.WEB_PERSONA:
                continue
            matched.deprecated = True
            incoming.version = matched.version + 1
            vault.parent_chunks.append(incoming)
        return vault

    def rebuild_child_index(self, vault: EvidenceVault, new_children: list[ChildChunk]) -> EvidenceVault:
        vault.child_chunks = new_children
        return vault

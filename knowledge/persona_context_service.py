from __future__ import annotations

from rag.tool import RAGTool


class PersonaContextService:
    """
    仅用于预览或调试，不参与正式聊天主链。
    正式 persona 召回统一通过 PersonaRAGEngine.recall() 完成。
    """

    def __init__(self, system):
        self.system = system
        self.rag = RAGTool(llm=getattr(system, "model", None))

    def extract_query_terms(self, query: str) -> list[str]:
        return self.rag.plan_query(query).metadata.get("keywords", [])

    def is_story_query(self, query: str) -> bool:
        return self.rag.plan_query(query).query_type == "story"

    def build_precise_query_context(self, query: str, top_k: int = 5, char_budget: int = 700) -> str:
        result = self.rag.search(query, self._entries(), top_k=top_k, query_type=self.rag.plan_query(query).query_type or "persona")
        return self.rag.build_context_block("persona evidence", result, char_budget=char_budget)

    def _entries(self) -> list[dict]:
        return [
            entry
            for entry in list(self.system.entries or [])
            if str(entry.get("kind") or entry.get("metadata", {}).get("kind") or "") == "source_chunk"
        ]

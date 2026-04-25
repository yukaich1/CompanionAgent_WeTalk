from __future__ import annotations

from rag.tool import RAGTool


class PersonaContextService:
    """仅用于预览或调试，不参与正式聊天主链。"""

    def __init__(self, system):
        self.system = system
        self.rag = RAGTool(llm=getattr(system, "model", None))

    def build_precise_query_context(self, query: str, top_k: int = 5, char_budget: int = 700) -> str:
        plan = self.rag.plan_query(query)
        kinds = {"story_chunk"} if plan.query_type == "story" else {"source_chunk"}
        entries = [
            entry
            for entry in list(self.system.entries or [])
            if str(entry.get("kind") or entry.get("metadata", {}).get("kind") or "") in kinds
        ]
        result = self.rag.search(
            query,
            entries,
            top_k=top_k,
            query_type=plan.query_type or "persona",
            namespace="persona",
        )
        return self.rag.build_context_block("persona evidence", result, char_budget=char_budget)

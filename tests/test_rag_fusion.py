import unittest

from rag.tool import RAGTool


class RAGFusionWeightTests(unittest.TestCase):
    def test_persona_queries_bias_lexical_more_than_story(self):
        rag = RAGTool(llm=None)
        persona = rag._fusion_weights("persona")
        story = rag._fusion_weights("story")

        self.assertGreaterEqual(persona["lexical"], persona["vector"])
        self.assertGreater(story["vector"], story["lexical"])
        self.assertGreater(persona["keyword_cap"], story["keyword_cap"])

    def test_memory_queries_keep_balanced_fusion(self):
        rag = RAGTool(llm=None)
        memory = rag._fusion_weights("memory")

        self.assertGreater(memory["vector"], memory["lexical"])
        self.assertLess(memory["vector"] - memory["lexical"], 0.25)
        self.assertGreater(memory["rrf"], 5.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from context.context_assembler import ContextAssembler
from context.evidence_adapter import EvidenceAdapter
from knowledge.knowledge_source import (
    KnowledgeSource,
    MemoryRecallResult,
    MemoryRecordView,
    PersonaRecallResult,
    RouteType,
)
from main import AISystem
from memory.memory_writer import MemoryWriter
from memory.state_models import RelationState
from memory.state_models import EpisodicRecord
from tools.intent_extractor import IntentExtractionResult


class StubModel:
    def __init__(self):
        self.taxonomy_calls = 0

    def generate(self, prompt, return_json=False, schema=None, **kwargs):
        if return_json:
            if isinstance(schema, dict) and "memory_type" in schema.get("properties", {}):
                self.taxonomy_calls += 1
                return {
                    "memory_type": "relationship",
                    "topic_room": "daily_chat",
                }
            if isinstance(schema, dict) and "mood" in schema.get("properties", {}):
                return {
                    "mood": "平静",
                    "intensity": 0.1,
                    "valence": 0.0,
                    "trust_delta": 0.0,
                    "affection_delta": 0.0,
                    "familiarity_delta": 0.01,
                    "rationale": "stub",
                }
            if isinstance(schema, dict) and "content" in schema.get("properties", {}):
                return {
                    "content": "用户最近状态偏疲惫。",
                    "domain": "relationship",
                    "confidence": 0.4,
                }
            return {}

        text = str(prompt if not isinstance(prompt, list) else prompt[-1].get("content", ""))
        if "当前关键证据不足" in text:
            return "我现在能确认的资料还不够多，所以不想把没有根据的话说得太满。"
        if "回答角色设定相关问题" in text:
            return "就现有证据来看，我更偏爱金钱和书籍。"
        if "自然回应普通闲聊" in text:
            return "我记得你之前提过最近很累，所以这会儿我更想先让你放松一点。"
        if "优先接住用户情绪" in text:
            return "我在这里。你可以慢慢说，我会认真听着。"
        if "根据外部证据回答现实信息" in text:
            return "根据目前查到的信息，答案就是这些，我不会再往外乱补。"
        if "回答身份、自我介绍" in text:
            return "我是伊蕾娜，一名旅行中的魔女。"
        return "这是一个自然但受控的模拟回复。"


class MemoryArchitectureTests(unittest.TestCase):
    def test_episodic_record_uses_summary_field(self):
        record = EpisodicRecord(record_id="r1", summary="新摘要")
        self.assertEqual(record.summary, "新摘要")
        self.assertEqual(record.recall_text(), "新摘要")

    def test_evidence_adapter_unifies_persona_and_memory(self):
        adapter = EvidenceAdapter()
        persona = PersonaRecallResult(
            integrated_context="我是伊蕾娜。",
            evidence_chunks=["喜欢金钱和书籍。"],
            coverage_score=0.8,
        )
        memory = MemoryRecallResult(
            episode_records=[
                MemoryRecordView(
                    record_id="m1",
                    content="User: 你说过你最近很累。",
                    source=KnowledgeSource.DIALOGUE_MEMORY,
                    metadata={"memory_type": "relationship", "topic_room": "daily_chat"},
                )
            ],
            stable_records=[
                MemoryRecordView(
                    record_id="m2",
                    content="用户最近状态偏疲惫。",
                    source=KnowledgeSource.DIALOGUE_MEMORY,
                    metadata={"domain": "relationship", "confidence": 0.7},
                )
            ],
            relation_state={"stage": "familiar"},
        )

        persona_items = adapter.adapt_persona(persona)
        memory_items = adapter.adapt_memory(memory)

        self.assertEqual(persona_items[0].source_kind, "persona_integrated")
        self.assertEqual(persona_items[1].source_kind, "persona_chunk")
        self.assertTrue(any(item.source_kind == "memory_episode" for item in memory_items))
        self.assertTrue(any(item.source_kind == "memory_stable" for item in memory_items))
        self.assertTrue(any(item.source_kind == "relation_state" for item in memory_items))

    def test_context_assembler_produces_layered_slots(self):
        assembler = ContextAssembler()
        assembled = assembler.assemble(
            route_type=RouteType.E1,
            deduped=type(
                "Deduped",
                (),
                {
                    "persona": PersonaRecallResult(
                        integrated_context="我是伊蕾娜。",
                        evidence_chunks=["喜欢金钱和书籍。"],
                        coverage_score=0.75,
                    ),
                    "memory": MemoryRecallResult(
                        episode_records=[
                            MemoryRecordView(
                                record_id="m1",
                                content="User: 你说过你最近很累。",
                                source=KnowledgeSource.DIALOGUE_MEMORY,
                                metadata={"memory_type": "relationship", "topic_room": "daily_chat", "retrieval_score": 0.9},
                            )
                        ],
                        stable_records=[
                            MemoryRecordView(
                                record_id="m2",
                                content="用户最近状态偏疲惫。",
                                source=KnowledgeSource.DIALOGUE_MEMORY,
                                metadata={"domain": "relationship", "confidence": 0.7},
                            )
                        ],
                        relation_state={"stage": "familiar"},
                    ),
                },
            )(),
        )
        self.assertIn("layer0_identity", assembled.slots)
        self.assertIn("layer1_stable_memory", assembled.slots)
        self.assertIn("layer2_topic_memory", assembled.slots)
        self.assertIn("layer3_deep_memory", assembled.slots)
        self.assertEqual(assembled.metadata["memory_layers"]["l0"], True)
        self.assertGreaterEqual(assembled.metadata["evidence_total_count"], 1)

    def test_memory_layers_deduplicate_overlapping_content(self):
        assembler = ContextAssembler()
        assembled = assembler.assemble(
            route_type=RouteType.E1,
            deduped=type(
                "Deduped",
                (),
                {
                    "persona": PersonaRecallResult(
                        integrated_context="我是伊蕾娜。",
                        evidence_chunks=["喜欢金钱和书籍。"],
                        coverage_score=0.75,
                    ),
                    "memory": MemoryRecallResult(
                        episode_records=[
                            MemoryRecordView(
                                record_id="m1",
                                content="用户最近状态偏疲惫。",
                                source=KnowledgeSource.DIALOGUE_MEMORY,
                                metadata={"memory_type": "relationship", "topic_room": "daily_chat", "retrieval_score": 0.9},
                            ),
                            MemoryRecordView(
                                record_id="m2",
                                content="用户最近状态偏疲惫。",
                                source=KnowledgeSource.DIALOGUE_MEMORY,
                                metadata={"memory_type": "relationship", "topic_room": "daily_chat", "retrieval_score": 0.8},
                            ),
                            MemoryRecordView(
                                record_id="m3",
                                content="用户上次提到想去旅行。",
                                source=KnowledgeSource.DIALOGUE_MEMORY,
                                metadata={"memory_type": "interest", "topic_room": "travel", "retrieval_score": 0.7},
                            ),
                        ],
                        stable_records=[
                            MemoryRecordView(
                                record_id="m4",
                                content="用户最近状态偏疲惫。",
                                source=KnowledgeSource.DIALOGUE_MEMORY,
                                metadata={"domain": "relationship", "confidence": 0.9},
                            )
                        ],
                        relation_state={"stage": "familiar"},
                    ),
                },
            )(),
        )
        self.assertIn("用户最近状态偏疲惫。", assembled.slots["layer1_stable_memory"])
        self.assertNotIn("用户最近状态偏疲惫。", assembled.slots["layer2_topic_memory"])
        self.assertIn("用户上次提到想去旅行。", assembled.slots["layer2_topic_memory"])

    def test_relation_stage_moves_to_acquaintance_after_real_initial_progress(self):
        writer = MemoryWriter()
        stage = writer._stage_from_state(
            RelationState(trust=0.1, affection=0.13, familiarity=0.1)
        )
        self.assertEqual(stage, "acquaintance")

    def test_memory_capture_uses_mode_safe_summary_not_raw_roleplay(self):
        system = AISystem()
        capture = system._memory_capture_for_turn(
            user_input="神通广大的伊蕾娜小姐，可以告诉我现在东京都的天气如何嘛？",
            response="我刚刚借用风的声音问了问东京的天气，还顺便想起了樱花季。",
            intent_result=IntentExtractionResult(response_mode="external"),
            persona_recall=PersonaRecallResult(),
        )
        self.assertEqual(capture["assistant_text"], "角色基于外部证据回答了现实信息。")
        self.assertIn("基于外部证据给出了简洁回答", capture["summary"])
        self.assertNotIn("借用风的声音", capture["verbatim_excerpt"])

    def test_sync_persona_state_deduplicates_parent_chunks_by_chunk_id(self):
        system = AISystem()
        system.persona_system.entries = [
            {"chunk_id": "dup-1", "content": "第一段资料", "keywords": ["资料"], "kind": "source_chunk"},
            {"chunk_id": "dup-1", "content": "第一段资料的重复版本", "keywords": ["资料"], "kind": "source_chunk"},
        ]
        system._sync_persona_state()
        self.assertEqual(len(system.new_persona_state.evidence_vault.parent_chunks), 1)

    def _build_system(self) -> AISystem:
        system = AISystem()
        system.model = StubModel()
        system.memory_writer.llm = system.model
        system.tool_runtime.execute = lambda **kwargs: type(
            "ToolReport",
            (),
            {"follow_up_message": "", "context_text": "", "metadata": {}},
        )()
        system.persona_system.entries = []
        system.new_persona_state.evidence_vault.parent_chunks = []
        system.new_memory_state.episode_records = []
        system.new_memory_state.stable_records = []
        return system

    def test_persona_fact_missing_is_gated(self):
        system = self._build_system()
        system.query_router.extractor.extract = lambda **kwargs: IntentExtractionResult(
            intent="character_related",
            response_mode="persona_fact",
            recall_mode="persona",
            persona_focus="likes",
            extracted_topic="喜好",
        )
        system.persona_rag_engine.recall = lambda *args, **kwargs: PersonaRecallResult()
        system.memory_rag_engine.recall = lambda *args, **kwargs: MemoryRecallResult()

        reply = system.send_message("你喜欢什么？")

        self.assertIn("资料还不够多", reply)
        self.assertEqual(system.last_debug_info["evidenceGate"]["reason"], "persona_missing")
        self.assertFalse(system.last_debug_info["evidenceGate"]["ready"])

    def test_memory_sensitive_request_uses_memory_evidence(self):
        system = self._build_system()
        system.query_router.extractor.extract = lambda **kwargs: IntentExtractionResult(
            intent="casual_chat",
            response_mode="casual",
            recall_mode="none",
            extracted_topic="之前的事情",
        )
        system.persona_rag_engine.recall = lambda *args, **kwargs: PersonaRecallResult(
            integrated_context="我是伊蕾娜。"
        )
        system.memory_rag_engine.recall = lambda *args, **kwargs: MemoryRecallResult(
            episode_records=[
                MemoryRecordView(
                    record_id="m1",
                    content="User: 你说过你最近很累。",
                    source=KnowledgeSource.DIALOGUE_MEMORY,
                    metadata={"memory_type": "relationship", "topic_room": "daily_chat", "retrieval_score": 0.9},
                )
            ],
            stable_records=[
                MemoryRecordView(
                    record_id="m2",
                    content="用户最近状态偏疲惫。",
                    source=KnowledgeSource.DIALOGUE_MEMORY,
                    metadata={"domain": "relationship", "confidence": 0.7},
                )
            ],
            relation_state={"stage": "familiar"},
        )

        reply = system.send_message("你还记得我之前说过什么吗？")

        self.assertIn("你之前提过最近很累", reply)
        self.assertEqual(system.last_debug_info["evidenceGate"]["reason"], "memory")
        self.assertTrue(system.last_debug_info["evidenceGate"]["ready"])

    def test_memory_writer_uses_single_taxonomy_call(self):
        system = self._build_system()
        system.memory_writer.remember(
            system.new_memory_state,
            summary="用户提到最近很累，我回应会陪着她。",
            user_text="我最近很累。",
            assistant_text="我会陪着你。",
        )
        self.assertEqual(system.model.taxonomy_calls, 1)


if __name__ == "__main__":
    unittest.main()

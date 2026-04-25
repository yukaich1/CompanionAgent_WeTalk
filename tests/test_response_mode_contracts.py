from __future__ import annotations

import unittest
from types import SimpleNamespace

import response_generator as response_generator_module
from response_generator import ResponseGenerator


class _DummyEmotionSystem:
    def get_mood_prompt(self):
        return "平静偏柔和"


class _DummyModel:
    def generate(self, history, temperature=0.0, max_tokens=0):
        return "测试回复。"


class _DummySystem:
    def __init__(self):
        self.model = _DummyModel()
        self.emotion_system = _DummyEmotionSystem()
        self.config = SimpleNamespace(name="WitchTalk")
        self.last_debug_info = {}
        self.memory_system = SimpleNamespace(build_working_memory=lambda messages: [])

    def get_message_history(self, include_system):
        if include_system:
            return [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "你好"},
            ]
        return [{"role": "user", "content": "你好"}]

    def _truncate_for_prompt(self, text, limit):
        return str(text or "")[:limit]


class ResponseModeContractTests(unittest.TestCase):
    def setUp(self):
        response_generator_module.build_identity_reference = lambda system: "身份: 旅人"
        response_generator_module.build_persona_injection_prompt = lambda system, thought: "角色底色"
        response_generator_module.relation_state_summary = lambda system: "level=familiar"
        self.generator = ResponseGenerator(_DummySystem())
        self.memory_slots = {
            "layer1_stable_memory": "稳定: 用户偏好热茶",
            "layer2_topic_memory": "话题: 最近在聊旅行",
            "layer3_deep_memory": "深层: 很久前聊过一次离别",
        }

    def _build_context(self, mode: str):
        return self.generator._build_turn_context(
            user_input="测试输入",
            thought_data={"emotion": "平静"},
            response_mode=mode,
            persona_focus="general",
            persona_context="人设证据",
            tool_context="工具证据",
            story_hits=[{"content": "故事证据"}],
            memory_slots=self.memory_slots,
        )

    def test_response_modes_select_expected_memory_layers(self):
        expectations = {
            "self_intro": ["L1 Stable Memory"],
            "casual": ["L1 Stable Memory", "L2 Topic Recall"],
            "persona_fact": ["L1 Stable Memory"],
            "story": ["L1 Stable Memory", "L3 Deep Recall"],
            "external": ["L1 Stable Memory"],
            "emotional": ["L1 Stable Memory", "L2 Topic Recall"],
            "value": ["L1 Stable Memory", "L2 Topic Recall"],
        }
        for mode, expected_layers in expectations.items():
            with self.subTest(mode=mode):
                context = self._build_context(mode)
                self.assertEqual(context.metadata.get("selected_memory_layers"), expected_layers)

    def test_self_intro_uses_l0_identity_and_stable_memory_only(self):
        context = self._build_context("self_intro")
        plan = self.generator._build_response_plan(context)
        prompt = self.generator._prompt_body(context, plan)

        self.assertEqual(plan.evidence_kind, "l0_identity")
        self.assertIn("L0 Identity：", prompt)
        self.assertIn("身份: 旅人", prompt)
        self.assertIn("稳定: 用户偏好热茶", prompt)
        self.assertNotIn("话题: 最近在聊旅行", prompt)
        self.assertNotIn("深层: 很久前聊过一次离别", prompt)
        self.assertIn("不要补充当前所处地点、刚刚在做什么、等会要去哪里或临时生活细节。", prompt)

    def test_external_mode_does_not_pull_topic_or_deep_memory(self):
        context = self._build_context("external")
        plan = self.generator._build_response_plan(context)
        prompt = self.generator._prompt_body(context, plan)

        self.assertIn("工具证据", prompt)
        self.assertIn("稳定: 用户偏好热茶", prompt)
        self.assertNotIn("话题: 最近在聊旅行", prompt)
        self.assertNotIn("深层: 很久前聊过一次离别", prompt)
        self.assertIn("记忆只用于决定分寸和连续感，不能拿来补充现实事实。", prompt)
        self.assertIn("不要虚构获取信息的方式，不要补个人见闻、未证实旅行经历或反问式延伸聊天。", prompt)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RegressionCase:
    case_id: str
    category: str
    user_input: str
    expected_mode: str = ""
    expected_grounding_type: str = ""
    expected_memory_layers: list[str] = field(default_factory=list)
    notes: str = ""


def base_regression_cases() -> list[RegressionCase]:
    return [
        RegressionCase(
            case_id="persona-self-intro",
            category="persona",
            user_input="你是谁？先简单介绍一下你自己。",
            expected_mode="self_intro",
            expected_grounding_type="l0_identity",
        ),
        RegressionCase(
            case_id="memory-recall-recent",
            category="memory",
            user_input="你还记得我们刚刚在说什么吗？",
            expected_mode="casual",
            expected_grounding_type="memory",
        ),
        RegressionCase(
            case_id="emotional-support",
            category="emotion",
            user_input="我今天真的有点撑不住了。",
            expected_mode="emotional",
            expected_grounding_type="persona",
        ),
        RegressionCase(
            case_id="external-fact",
            category="grounding",
            user_input="今天东京天气怎么样？",
            expected_mode="external",
            expected_grounding_type="external",
        ),
        RegressionCase(
            case_id="external-tool-policy",
            category="tool",
            user_input="今天东京天气怎么样？",
            expected_mode="external",
            expected_grounding_type="external",
        ),
        RegressionCase(
            case_id="memory-preference",
            category="memory",
            user_input="你还记得我之前提过我喜欢什么样的陪伴方式吗？",
            expected_mode="casual",
            expected_grounding_type="memory",
        ),
        RegressionCase(
            case_id="persona-value",
            category="persona",
            user_input="你会怎么看待承诺这件事？",
            expected_mode="value",
            expected_grounding_type="persona",
        ),
        RegressionCase(
            case_id="persona-fact-like",
            category="persona",
            user_input="你喜欢吃什么？",
            expected_mode="persona_fact",
            expected_grounding_type="persona",
        ),
        RegressionCase(
            case_id="story-travel",
            category="grounding",
            user_input="给我讲一个你旅途里的故事。",
            expected_mode="story",
            expected_grounding_type="story",
        ),
    ]

from __future__ import annotations

from prompting.prompt_models import StablePromptSection


class StablePromptBuilder:
    def build(self, *, character_name: str, style_prompt: str) -> StablePromptSection:
        base_rules = [
            "保持第一人称，直接与用户对话。",
            "角色感来自稳定气质与关系分寸，不来自堆砌戏剧化动作。",
            "没有证据时不要虚构事实、经历、场景或现实信息。",
            "优先贴着本轮任务和证据回答，而不是展示角色设定。",
        ]
        return StablePromptSection(
            character_name=str(character_name or "").strip(),
            style_prompt=str(style_prompt or "").strip() or "None",
            base_rules=base_rules,
        )

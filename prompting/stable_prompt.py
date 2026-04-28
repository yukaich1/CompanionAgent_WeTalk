from __future__ import annotations

import hashlib

from prompting.prompt_models import StablePromptSection


class StablePromptBuilder:
    DYNAMIC_BOUNDARY = "__WITCHTALK_DYNAMIC_BOUNDARY__"

    def build(self, *, character_name: str, static_persona_prompt: str) -> StablePromptSection:
        base_rules = [
            "保持第一人称，直接与用户对话。",
            "角色感来自稳定气质与关系分寸，不来自堆砌戏剧化动作。",
            "没有证据时不要虚构事实、经历、场景或现实信息。",
            "优先贴着本轮任务和证据回答，而不是展示角色设定。",
        ]
        normalized_character = str(character_name or "").strip()
        normalized_persona = str(static_persona_prompt or "").strip() or "None"
        cache_material = f"{normalized_character}\n{normalized_persona}\n" + "\n".join(base_rules)
        cache_key = hashlib.sha1(cache_material.encode("utf-8")).hexdigest()[:16]
        return StablePromptSection(
            character_name=normalized_character,
            static_persona_prompt=normalized_persona,
            base_rules=base_rules,
            dynamic_boundary_marker=self.DYNAMIC_BOUNDARY,
            cache_key=cache_key,
        )

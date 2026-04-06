from __future__ import annotations

import re


class PersonaResponsePolicy:
    def __init__(self, persona_system, character_name_getter):
        self.persona_system = persona_system
        self._character_name_getter = character_name_getter

    def infer_grounding_needs(self, user_input: str, route_decision=None, persona_recall=None, tool_report=None) -> dict:
        text = str(user_input or "").strip()
        route_type = str(getattr(route_decision, "type", "") or "").lower()
        metadata = getattr(persona_recall, "metadata", {}) or {}
        story_hits = list(metadata.get("story_hits", []) or [])
        evidence_chunks = list(getattr(persona_recall, "evidence_chunks", []) or [])
        tool_follow_up = str(getattr(tool_report, "follow_up_message", "") or "").strip()

        is_self_intro = self._looks_like_self_intro(text)
        asks_story = self._looks_like_story_request(text)
        asks_external = self._looks_like_external_request(text) or route_type in {"e3", "e4"}
        asks_persona_fact = self._looks_like_persona_request(text)
        asks_persona = is_self_intro or asks_story or asks_persona_fact or route_type in {"e1", "e2", "e2b"} or bool(evidence_chunks)

        return {
            "is_self_intro": is_self_intro,
            "needs_story_grounding": asks_story,
            "needs_persona_grounding": asks_persona,
            "needs_persona_fact_grounding": asks_persona_fact,
            "needs_external_grounding": asks_external,
            "has_story_hits": bool(story_hits),
            "has_persona_evidence": bool(evidence_chunks),
            "has_identity_reference": self.has_identity_reference(),
            "has_tool_followup": bool(tool_follow_up),
        }

    def has_identity_reference(self) -> bool:
        background = self._base_template_section("00_BACKGROUND")
        profile = background.get("profile", {}) if isinstance(background, dict) else {}
        experiences = list(background.get("key_experiences", []) or []) if isinstance(background, dict) else []
        return bool(profile or experiences)

    def persona_query_focus(self, user_input: str) -> str:
        text = str(user_input or "")
        if self._looks_like_self_intro(text):
            return "self_intro"
        if any(token in text for token in ("口头禅", "常说", "会怎么说", "说话习惯")):
            return "catchphrase"
        if any(token in text for token in ("喜欢", "喜好", "爱吃", "偏好")):
            return "likes"
        if any(token in text for token in ("讨厌", "不喜欢", "禁忌", "忌讳", "厌恶")):
            return "dislikes"
        if any(token in text for token in ("性格", "怎么样的人", "什么样", "脾气")):
            return "personality"
        return ""

    def build_self_intro_response(self) -> str:
        if not self.has_identity_reference():
            return ""
        background = self._base_template_section("00_BACKGROUND")
        profile = background.get("profile", {}) if isinstance(background, dict) else {}
        experiences = list(background.get("key_experiences", []) or []) if isinstance(background, dict) else []

        fragments = [self._trim_rule_text(value, 48) for value in profile.values() if self._trim_rule_text(value, 48)]
        lines: list[str] = []
        if fragments:
            lines.append(f"我是{fragments[0]}。")
        elif self._character_name():
            lines.append(f"我是{self._character_name()}。")
        if len(fragments) > 1:
            lines.append("如果只说最基本的身份，大概可以算是" + "，".join(fragments[1:3]) + "。")
        if experiences:
            lead = self._trim_rule_text(experiences[0], 60)
            if lead:
                lines.append("至于来历，能明说的大概就是：" + lead + "。")
        return "\n".join(lines[:3]).strip()

    def generate_in_character_refusal(self, system, mode: str, user_input: str = "") -> str:
        style_prompt = self._style_prompt()
        identity_prompt = self._identity_prompt() or "暂无可用身份资料。"
        reason_map = {
            "self_intro": "基础身份信息不足，不能把不存在的身份背景说成真的。",
            "story": "没有足够的故事证据，不能把零碎信息补成完整经历。",
            "external": "没有可靠的外部工具结果，不能把现实信息说成确定事实。",
            "persona": "没有足够的人设证据，不能把角色设定说满。",
        }
        reason = reason_map.get(mode, reason_map["persona"])
        prompt = (
            "你要以角色本人第一人称，简短拒绝一个无法可靠回答的问题。\n"
            f"角色名：{self._character_name()}\n"
            f"说话底色：\n{style_prompt}\n\n"
            f"身份参考：\n{identity_prompt}\n\n"
            f"用户问题：{user_input}\n"
            f"拒绝原因：{reason}\n\n"
            "要求：\n"
            "1. 只用自然简体中文。\n"
            "2. 保持第一人称。\n"
            "3. 简短，不解释系统，不提证据标签。\n"
            "4. 可以保留角色气质，但不要固定成僵硬的疏离或讨好。\n"
            "5. 不编造任何新事实。"
        )
        try:
            reply = str(system.model.generate(prompt, temperature=0.25, max_tokens=120) or "").strip()
            if reply:
                return reply
        except Exception:
            pass
        if mode == "external":
            return "这件事我现在还不能替你说得太肯定。等查清楚了，再告诉你。"
        if mode == "story":
            return "这类事如果资料里没写清楚，我就不把它说成完整故事。"
        if mode == "self_intro":
            return f"我是{self._character_name()}。再往下那些没写清楚的部分，我就不乱补了。"
        return "这件事我不能随口说满。资料里没有写清楚的部分，我不会擅自补上。"

    def _style_prompt(self) -> str:
        voice_card = str(getattr(self.persona_system, "character_voice_card", "") or "").strip()
        keywords = [self._trim_rule_text(item, 18) for item in list(getattr(self.persona_system, "display_keywords", []) or [])[:8]]
        examples: list[str] = []
        for item in list(getattr(self.persona_system, "style_examples", []) or [])[:2]:
            text = item.get("text", "") if isinstance(item, dict) else item
            clean = self._trim_rule_text(text, 80)
            if clean:
                examples.append(clean)
        lines: list[str] = []
        if voice_card:
            lines.append(voice_card)
        if keywords:
            lines.append("关键词：" + "、".join(keyword for keyword in keywords if keyword))
        if examples:
            lines.append("示例：" + " / ".join(examples))
        return "\n".join(lines).strip() or "保持角色本人的说话方式。"

    def _identity_prompt(self) -> str:
        background = self._base_template_section("00_BACKGROUND")
        profile = background.get("profile", {}) if isinstance(background, dict) else {}
        experiences = list(background.get("key_experiences", []) or []) if isinstance(background, dict) else []
        lines: list[str] = []
        for key, value in profile.items():
            clean = self._trim_rule_text(value, 50)
            if clean:
                lines.append(f"{key}: {clean}")
        for item in experiences[:3]:
            clean = self._trim_rule_text(item, 60)
            if clean:
                lines.append(f"经历: {clean}")
        return "\n".join(lines).strip()

    def _looks_like_self_intro(self, text: str) -> bool:
        value = str(text or "")
        tokens = ("自我介绍", "介绍一下你自己", "你是谁", "介绍你自己", "做个介绍", "你从哪来", "你是什么身份")
        return any(token in value for token in tokens)

    def _looks_like_story_request(self, text: str) -> bool:
        value = str(text or "")
        tokens = ("故事", "经历", "过去", "以前发生", "旅行见闻", "讲一段", "讲一个", "说说那次", "那件事")
        if any(token in value for token in tokens):
            return True
        return bool(re.search(r"(你|妳).*(经历|过去|以前|旅行|故事)", value))

    def _looks_like_persona_request(self, text: str) -> bool:
        value = str(text or "")
        tokens = ("喜欢", "讨厌", "性格", "口头禅", "价值观", "世界观", "怎么看", "设定", "你自己", "说话方式", "习惯", "偏好")
        if any(token in value for token in tokens):
            return True
        return bool(re.search(r"(你|妳).*(性格|喜好|偏好|口头禅|设定|怎么看)", value))

    def _looks_like_external_request(self, text: str) -> bool:
        value = str(text or "")
        tokens = ("天气", "气温", "温度", "下雨", "新闻", "最新", "最近", "比赛", "票房", "搜索", "是什么", "为什么", "原理", "介绍", "解释", "汇率", "价格")
        return any(token in value for token in tokens)

    def _base_template_section(self, key: str) -> dict:
        base_template = getattr(self.persona_system, "base_template", {}) or {}
        value = base_template.get(key, {})
        return value if isinstance(value, dict) else {}

    def _trim_rule_text(self, text: str, max_chars: int = 70) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n-:：；，。！？")
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "…"

    def _character_name(self) -> str:
        return str(self._character_name_getter() or getattr(self.persona_system, "persona_name", "") or "角色").strip()

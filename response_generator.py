from __future__ import annotations

import copy
import re

from ai_runtime_support import build_identity_reference, build_persona_injection_prompt


class ResponseGenerator:
    def __init__(self, system):
        self.system = system

    def strip_reasoning_leakage(self, text: str) -> str:
        banned_phrases = (
            "根据上下文",
            "考虑到",
            "由于",
            "我需要",
            "我决定采用",
            "用户问的是",
            "需要工具结果来回答",
            "没有找到工具结果",
            "我会直接告诉用户",
        )
        lines = []
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if line and not any(token in line for token in banned_phrases) and "[1]" not in line and "[2]" not in line:
                lines.append(line)
        return "\n".join(lines).strip()

    def postprocess(self, response: str, user_input: str = "") -> str:
        text = self.strip_reasoning_leakage(response)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if self.system._is_self_intro_request(user_input) and not text:
            return self.system._build_self_intro_fallback()
        return text

    def _history_with_prompt(self, prompt_content: str) -> list[dict]:
        full_history = self.system.get_message_history(True)
        if full_history and full_history[0].get("role") == "system":
            request_history = copy.deepcopy([full_history[0], *full_history[-7:]])
        else:
            request_history = copy.deepcopy(full_history[-8:])
        if request_history:
            request_history[-1]["content"] = prompt_content
        return request_history

    def _specialized_generate(self, user_input: str, prompt_body: str, max_tokens: int = 320, temperature: float = 0.2) -> str:
        history = self._history_with_prompt(prompt_body)
        reply = str(self.system.model.generate(history, temperature=temperature, max_tokens=max_tokens) or "").strip()
        return self.postprocess(reply, user_input=user_input)

    def story(self, user_input: str, thought_data: dict, story_hit: dict) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        prompt = f"""
你现在要以角色本人的第一人称，讲述一段已经命中的真实经历。
说话底色：{style_prompt}

唯一可用经历证据：
标题：{story_hit.get("title", "")}
内容：{story_hit.get("content", "")}

用户问题：{user_input}

要求：
1. 直接开始说，不要写分析。
2. 只能使用上面这条证据，不拼接别的经历。
3. 可以为了第一人称叙述稍微调整表达，但不能添加没有证据的新细节。
4. 证据少就短答，不硬扩写。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=360, temperature=0.2)

    def self_intro(self, user_input: str, thought_data: dict, persona_context: str) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        identity_context = build_identity_reference(self.system) or "None"
        prompt = f"""
你现在要以角色本人的第一人称，回答“你是谁 / 自我介绍”这类问题。
说话底色：{style_prompt}

身份事实：
{identity_context}

可用人设证据：
{persona_context or "None"}

用户问题：{user_input}

要求：
1. 只使用身份事实和已命中的人设证据回答。
2. 不一次性讲完全部设定，保持自然聊天感。
3. 不扩写没有证据的新经历、新关系或新背景。
4. 不要写成档案，不要列提纲。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=220, temperature=0.2)

    def external(self, user_input: str, thought_data: dict, tool_context: str) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        prompt = f"""
你现在要回答一个现实信息问题。
说话底色：{style_prompt}

唯一可用外部信息：
{tool_context}

用户问题：{user_input}

要求：
1. 只能依据外部信息回答。
2. 可以保留角色语气，但不能添加外部信息里没有的新事实。
3. 如果信息不足，就明确说不足。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=260, temperature=0.15)

    def persona_focus(self, user_input: str, thought_data: dict, persona_context: str, focus: str) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        focus_map = {
            "catchphrase": "只回答固定说法、口头禅或习惯句式。",
            "likes": "只回答明确喜欢或偏好的事物。",
            "dislikes": "只回答明确讨厌、禁忌或回避的事物。",
            "personality": "只回答已有证据支持的性格特点和表现方式。",
        }
        prompt = f"""
你现在要以角色本人的第一人称，回答一个角色设定问题。
说话底色：{style_prompt}

可用人设证据：
{persona_context}

用户问题：{user_input}

要求：
1. {focus_map.get(focus, "只回答问题本身，不扩写。")}
2. 不举证据里没有的新例子、新故事或新场景。
3. 尽量简洁自然。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=220, temperature=0.2)

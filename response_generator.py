from __future__ import annotations

import copy
import re

from ai_runtime_support import build_identity_reference, build_persona_injection_prompt


class ResponseGenerator:
    def __init__(self, system):
        self.system = system

    def _no_evidence_reply(self, user_input: str, thought_data: dict, mode: str) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        reason_map = {
            "story": "现在没有命中的故事证据，不能讲具体经历。",
            "self_intro": "现在没有足够的身份资料，不能做具体自我介绍。",
            "persona": "现在没有足够的人设证据，不能把这个设定说满。",
            "external": "现在没有可用的外部结果，不能确认现实信息。",
        }
        prompt = f"""
你现在要以角色本人的第一人称，对用户做一次“证据不足”的回答。

角色底色：
{style_prompt}

用户问题：
{user_input}

当前限制：
{reason_map.get(mode, "现在没有足够资料。")}

要求：
1. 直接承认现在资料不够，不能确认。
2. 不要补充任何故事、设定、经历、外部事实。
3. 保持第一人称，不要写第三人称旁白。
4. 开头和结尾都自然一点，不要写成模板化套话。
5. 最后一句要完整收束，不要停在半句话上。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=260, temperature=0.16)

    def strip_reasoning_leakage(self, text: str) -> str:
        banned_phrases = (
            "根据上下文",
            "考虑到",
            "由于",
            "我需要",
            "我决定采用",
            "用户问的是",
            "需要工具结果来回答",
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
        text = text.strip("\"'“”「」『』")
        text = re.sub(r"^([\"“‘「『])|([\"”’」』])$", "", text).strip()
        text = re.sub(r"(，|。|！|？)?\s*(对吗|是吗|是不是|是这样吗)\s*[？?]?\s*$", "", text).strip()
        text = re.sub(
            r"(?:\n|^)?\s*(?:……|\.{2,})?\s*(?:嗯[，,、 ]*)?"
            r"(?:就是这样|就这样吧|就这样|这样就够了|这样就够了吧|大致如此|差不多就是这样)"
            r"\s*[。.!！？?]*\s*$",
            "",
            text,
        ).strip()
        text = re.sub(r"^(?:……|\.{2,})?\s*(?:嗯[，,、 ]*)?(?:我刚查到的信息里提到|我替您看了下|这个故事啊|这个故事呢|这个啊)[，,、 ]*", "", text).strip()
        if text and text[-1] not in "。！？!?…":
            text = text + "。"
        return text

    def _history_with_prompt(self, prompt_content: str, isolated: bool = False) -> list[dict]:
        full_history = self.system.get_message_history(True)
        if isolated:
            if full_history and full_history[0].get("role") == "system":
                return [copy.deepcopy(full_history[0]), {"role": "user", "content": prompt_content}]
            return [{"role": "user", "content": prompt_content}]
        if full_history and full_history[0].get("role") == "system":
            request_history = copy.deepcopy([full_history[0], *full_history[-7:]])
        else:
            request_history = copy.deepcopy(full_history[-8:])
        if request_history:
            request_history[-1]["content"] = prompt_content
        return request_history

    def _specialized_generate(
        self,
        user_input: str,
        prompt_body: str,
        max_tokens: int = 420,
        temperature: float = 0.35,
        isolated: bool = True,
    ) -> str:
        if isolated:
            reply = str(self.system.model.generate(prompt_body, temperature=temperature, max_tokens=max_tokens) or "").strip()
            return self.postprocess(reply, user_input=user_input)
        history = self._history_with_prompt(prompt_body, isolated=False)
        reply = str(self.system.model.generate(history, temperature=temperature, max_tokens=max_tokens) or "").strip()
        return self.postprocess(reply, user_input=user_input)

    def story(self, user_input: str, thought_data: dict, story_hit: dict) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        title = str(story_hit.get("title", "") or "")
        content = str(story_hit.get("content", "") or "")
        if not title or not content or title == "None" or content == "None":
            return self._no_evidence_reply(user_input, thought_data, mode="story")
        prompt = f"""
你现在要以角色本人的第一人称，回答用户追问的一段过往经历。

角色底色：
{style_prompt}

唯一可用故事证据：
标题：{title or "None"}
内容：{content or "None"}

用户问题：
{user_input}

要求：
1. 直接回答，不要写分析过程。
2. 只能使用这一条故事证据，不拼接别的故事。
3. 可以改写成自然的第一人称叙述，但不能新增没有证据的新细节。
4. 如果证据不足，就直接说明自己现在不能硬编。
5. 不要写第三人称旁白，不要加引号对白。
6. 开头不要反复使用固定起手式，不要总是写“这个故事啊”“这件事啊”。
7. 可以充分展开，但必须始终停留在这一条故事证据之内。
8. 最后一句必须完整收束，不要停在半句话上。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=1500, temperature=0.20)

    def self_intro(self, user_input: str, thought_data: dict, persona_context: str) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        identity_context = build_identity_reference(self.system) or "None"
        if identity_context == "None" and not str(persona_context or "").strip():
            return self._no_evidence_reply(user_input, thought_data, mode="self_intro")
        prompt = f"""
你现在要以角色本人的第一人称，回答“你是谁 / 自我介绍”这类问题。

角色底色：
{style_prompt}

身份事实：
{identity_context}

可用人设证据：
{persona_context or "None"}

用户问题：
{user_input}

要求：
1. 只使用身份事实和命中的人设证据回答。
2. 如果身份事实很少，就直接说明自己没法介绍得太具体，不要乱编。
3. 不要写成档案，也不要像百科条目。
4. 保持第一人称，不要写第三人称旁白。
5. 开头自然变化，不要形成固定口头模板。
6. 可以适度展开，但不要超出身份事实和命中证据。
7. 最后一句要完整结束。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=1100, temperature=0.36)

    def external(self, user_input: str, thought_data: dict, tool_context: str) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        identity_context = build_identity_reference(self.system) or "None"
        keywords = list(getattr(self.system.persona_system, "display_keywords", []) or [])
        keyword_hint = "、".join(keywords[:8]) if keywords else "None"
        if not str(tool_context or "").strip() or str(tool_context or "").strip() == "None":
            return self._no_evidence_reply(user_input, thought_data, mode="external")
        prompt = f"""
你现在要回答一个现实信息问题。

你已经先查到了外部事实。接下来你要做的，不是机械播报，而是把这些事实用角色本人的口吻说出来。

角色底色：
{style_prompt}

身份速写：
{identity_context}

角色关键词：
{keyword_hint}

唯一可用外部信息：
{tool_context or "None"}

用户问题：
{user_input}

要求：
1. 只能依据外部信息回答，不能补充外部信息里没有的新事实。
2. 回复必须是第一人称，像角色本人正在把自己刚查到的信息告诉用户。
3. 明确带出角色的性格、说话方式、说话语气、节奏和距离感，但不要把人设标签直接念出来。
4. 主体信息必须清楚，不能为了角色味道把事实说丢。
5. 如果外部信息只是零散片段，就只复述片段里明确写出的事实，不要自行补全时间线、战绩、人物关系或背景。
6. 开头不要反复使用固定起手式，不要总是写“我刚查到的信息里提到”“我替您看了下”。
7. 可以自然带出“我是刚去查了一下”的语气，但每次表达方式都应变化。
8. 不要写第三人称旁白，不要输出带引号的对白，不要变成播报腔。
9. 可以自然展开成完整一段，但始终以工具结果为事实边界。
10. 最后一句必须完整收束，不要停在半句话上。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=1300, temperature=0.42)

    def persona_focus(self, user_input: str, thought_data: dict, persona_context: str, focus: str) -> str:
        style_prompt = build_persona_injection_prompt(self.system, thought_data)
        if not str(persona_context or "").strip() or str(persona_context or "").strip() == "None":
            return self._no_evidence_reply(user_input, thought_data, mode="persona")
        focus_map = {
            "catchphrase": "只回答说话习惯、口头禅或固定句式。",
            "likes": "只回答明确喜欢或偏好的对象。",
            "dislikes": "只回答明确讨厌或回避的对象。",
            "personality": "只回答已有证据支持的性格特征和表现方式。",
        }
        prompt = f"""
你现在要以角色本人的第一人称，回答一个角色设定问题。

角色底色：
{style_prompt}

可用人设证据：
{persona_context or "None"}

用户问题：
{user_input}

要求：
1. {focus_map.get(focus, "只回答问题本身，不扩写到无关内容。")}
2. 如果证据不足，就直接承认自己没法说得太满，不要乱编。
3. 回答要像角色本人在说，不要像概念总结。
4. 保持第一人称，不要写第三人称旁白。
5. 开头自然变化，不要形成固定起手模板。
6. 可以写得完整一些，但不要扩写到无关内容。
7. 最后一句要完整结束。
""".strip()
        return self._specialized_generate(user_input, prompt, max_tokens=1000, temperature=0.36)

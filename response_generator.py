from __future__ import annotations

import copy
import re

from ai_runtime_support import build_identity_reference, build_persona_injection_prompt


class ResponseGenerator:
    def __init__(self, system):
        self.system = system

    def _recent_dialogue_block(self, max_messages: int = 6) -> str:
        messages = [
            message
            for message in self.system.get_message_history(False)
            if message.get("role") in {"user", "assistant"}
        ]
        recent = messages[-max_messages:]
        if not recent:
            return "None"

        lines: list[str] = []
        for message in recent:
            role = "用户" if message.get("role") == "user" else "你上一轮的回答"
            content = str(message.get("content", "") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "None"

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

    def _sanitize_response(self, response: str) -> str:
        text = str(response or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return ""

        cleaned_lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if re.fullmatch(r"[（(][^()\n]{4,}[)）]", line):
                continue
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^[\"“”‘’「」『』]+|[\"“”‘’「」『』]+$", "", text).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in "。！？?!~":
            text += "。"
        return text

    def _looks_truncated(self, response: str) -> bool:
        text = str(response or "").strip()
        if not text or len(text) < 24:
            return False
        if text[-1] not in "。！？?!":
            return True
        if re.search(r"[，、：；（\[]$", text):
            return True
        if text.count("（") > text.count("）") or text.count("(") > text.count(")"):
            return True
        if text.count("“") > text.count("”") or text.count('"') % 2 == 1:
            return True
        return False

    def _generate(self, prompt_body: str, max_tokens: int, temperature: float, isolated: bool = True) -> str:
        if isolated:
            reply = self.system.model.generate(prompt_body, temperature=temperature, max_tokens=max_tokens)
        else:
            history = self._history_with_prompt(prompt_body, isolated=False)
            reply = self.system.model.generate(history, temperature=temperature, max_tokens=max_tokens)
        return self._sanitize_response(str(reply or ""))

    def _generate_with_continuation(
        self,
        prompt_body: str,
        max_tokens: int,
        temperature: float,
        isolated: bool = True,
        continuation_max_tokens: int = 420,
    ) -> str:
        if isolated:
            first_pass_raw = str(self.system.model.generate(prompt_body, temperature=temperature, max_tokens=max_tokens) or "")
        else:
            history = self._history_with_prompt(prompt_body, isolated=False)
            first_pass_raw = str(self.system.model.generate(history, temperature=temperature, max_tokens=max_tokens) or "")

        if not self._looks_truncated(first_pass_raw):
            return self._sanitize_response(first_pass_raw)

        continuation_prompt = f"""
下面是一段还没有完整收束的角色回答。请继续往下写，只补上自然收尾，不要重复前面的内容。

必须遵守：
1. 延续同一段回答的语气、视角和事实边界。
2. 不要引入新的事实、故事细节或新的结论。
3. 只把当前这段话自然说完，让最后一句完整落地。
4. 不要写“补充说明”“继续”“总之就是这样”之类的提示语或模板收尾。

原始任务：
{prompt_body}

当前已生成内容：
{first_pass_raw}
""".strip()
        continuation = str(
            self.system.model.generate(
                continuation_prompt,
                temperature=max(0.12, temperature - 0.05),
                max_tokens=continuation_max_tokens,
            )
            or ""
        )
        combined = f"{first_pass_raw.rstrip()} {continuation.lstrip()}".strip()
        return self._sanitize_response(combined)

    def _style_block(self, thought_data: dict | None = None) -> str:
        return str(build_persona_injection_prompt(self.system, thought_data or {}) or "").strip() or "None"

    def _identity_block(self) -> str:
        return str(build_identity_reference(self.system) or "").strip() or "None"

    def _display_keywords_block(self) -> str:
        keywords = list(getattr(self.system, "display_keywords", []) or [])
        return "、".join(str(keyword).strip() for keyword in keywords if str(keyword).strip()) or "None"

    def _build_prompt(
        self,
        *,
        task_label: str,
        user_input: str,
        thought_data: dict | None = None,
        evidence_title: str = "",
        evidence_text: str = "",
        constraints: list[str] | None = None,
    ) -> str:
        style_block = self._style_block(thought_data)
        identity_block = self._identity_block()
        display_keywords = self._display_keywords_block()
        recent_dialogue = self._recent_dialogue_block()
        evidence_title = str(evidence_title or "").strip()
        evidence_text = str(evidence_text or "").strip() or "None"
        constraints = [str(item).strip() for item in list(constraints or []) if str(item).strip()]

        evidence_header = f"证据标题：{evidence_title}\n" if evidence_title else ""
        constraints_block = "\n".join(f"- {item}" for item in constraints) if constraints else "- 按证据回答。"

        return f"""
你现在要完成的任务：{task_label}

你要像角色本人一样直接对用户说话。
角色底色只决定“怎么说”，不能凭空制造“说什么”。

角色底色：
{style_block}

身份参考：
{identity_block}

展示标签：
{display_keywords}

这些标签只用于帮助你理解人物，不要把标签本身机械复述给用户。

最近对话：
{recent_dialogue}

{evidence_header}本轮证据：
{evidence_text}

用户这句话：
{user_input}

必须遵守：
{constraints_block}
- 保持第一人称，直接对用户说话。
- 不要写成第三人称旁白、舞台说明、括号动作或小剧场。
- 不要机械重复固定开场、固定退场句、固定身份句或固定口头禅。
- 模板、标签、示例句都只是帮助你理解角色，不是现成台词库。
- 优先直接回应用户当前意图，不要把回答写成谜语、猜谜、展示文案或刻意表演的人设开场，除非用户明确要求这种效果。
- 如果证据不足，只承认证据不足，不要编造。
""".strip()

    def _no_evidence_reply(self, user_input: str, thought_data: dict, mode: str = "general") -> str:
        mode_guidance = {
            "story": "用户在问故事或经历，但当前没有足够证据支持具体叙述。",
            "self_intro": "用户在问自我介绍，但当前身份资料不足，只能承认信息有限。",
            "external": "用户在问现实信息，但当前没有可靠工具结果。",
            "persona": "用户在问角色设定，但当前没有足够命中证据。",
            "general": "当前证据不足，请自然承认边界。",
        }
        extra_constraints = []
        if mode == "self_intro":
            extra_constraints = [
                "不要写成失忆、混乱、自我否定或“连自己是谁都不知道”这类戏剧化表达。",
                "更适合平静地说现在能确认的资料不多，或者请用户换个更具体的角度问。",
            ]
        prompt = self._build_prompt(
            task_label="在证据不足时给出自然、诚实、角色化的回应",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text=mode_guidance.get(mode, mode_guidance["general"]),
            constraints=[
                "明确承认当前资料或证据不足。",
                "不要暗示其实知道但不愿意说，也不要假装忘记某段并不存在的经历。",
                "可以轻微保持角色底色，但不要靠固定句式敷衍。",
                "如果合适，可以引导用户换个更具体的问法。",
                *extra_constraints,
            ],
        )
        response = self._generate(prompt, max_tokens=420, temperature=0.45, isolated=False)
        if mode == "self_intro" and all(token not in response for token in ("我是谁都", "说不太清楚", "困惑", "失忆", "连自己是谁")):
            return response
        if mode not in {"story", "self_intro"}:
            return response
        if all(token not in response for token in ("没发生", "还没发生", "没发生过", "忘在脑后")):
            if mode == "story":
                return response

        if mode == "story":
            retry_prompt = self._build_prompt(
                task_label="重新完成一次无证据的故事回应",
                user_input=user_input,
                thought_data=thought_data,
                evidence_text="当前并没有可用的故事证据，只能诚实承认不足。",
                constraints=[
                    "不要说某段故事没发生过，也不要说自己忘了。",
                    "只说现在资料不够，或者请用户给出更具体范围。",
                    "保持第一人称和角色底色，但不要戏剧化。",
                ],
            )
            return self._generate(retry_prompt, max_tokens=380, temperature=0.38, isolated=False)

        retry_prompt = self._build_prompt(
            task_label="重新完成一次无证据的自我介绍回应",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text="当前缺少足够身份资料，只能平静承认资料有限。",
            constraints=[
                "不要写成失忆、混乱、自我否定或戏剧化失控。",
                "直接说明现在能确认的自我资料不多，如果对方愿意，可以换个具体方向问。",
                "保持第一人称和角色底色，但语气要稳定。",
            ],
        )
        return self._generate(retry_prompt, max_tokens=360, temperature=0.36, isolated=False)

    def casual(
        self,
        user_input: str,
        thought_data: dict,
        persona_context: str = "",
        tool_context: str = "",
    ) -> str:
        evidence_parts = [
            part.strip()
            for part in [str(persona_context or ""), str(tool_context or "")]
            if str(part or "").strip()
        ]
        prompt = self._build_prompt(
            task_label="自然接住一条普通闲聊或问候，让对话顺滑往下继续",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text="\n\n".join(evidence_parts) if evidence_parts else "最近对话与角色底色。",
            constraints=[
                "先自然回应这句问候或闲聊本身，不要急着表演角色设定。",
                "如果合适，可以顺着时间感、场景感或关系感轻轻展开一点，但不要生硬加戏。",
                "不要突然切去讲设定、故事、自我介绍或外部知识。",
                "普通问候优先像真实对话，不要写成猜谜、展示句或固定模板寒暄。",
            ],
        )
        return self._generate(prompt, max_tokens=820, temperature=0.6, isolated=False)

    def story(self, user_input: str, thought_data: dict, story_hit: dict | None) -> str:
        story_hit = dict(story_hit or {})
        content = str(story_hit.get("content", "") or story_hit.get("text", "") or "").strip()
        title = str(story_hit.get("title", "") or story_hit.get("metadata", {}).get("title", "") or "").strip()
        if not content:
            return self._no_evidence_reply(user_input, thought_data, mode="story")

        prompt = self._build_prompt(
            task_label="只讲述一个已经命中的角色故事，并用第一人称自然叙述",
            user_input=user_input,
            thought_data=thought_data,
            evidence_title=title,
            evidence_text=content,
            constraints=[
                "只能依据这一段故事证据回答，只讲这一段，不拼接别的故事。",
                "允许做轻微的第一人称改写，让叙述更自然，但不能新增证据中没有的细节。",
                "不要补地点、天气、动机、对话、时间步骤或新人关系。",
                "不要先做分析、总结证据或比较别的故事，再开始讲。",
                "最后一句必须完整收束，不要停在半句话上。",
            ],
        )
        return self._generate_with_continuation(
            prompt,
            max_tokens=1800,
            temperature=0.35,
            isolated=False,
            continuation_max_tokens=520,
        )

    def self_intro(self, user_input: str, thought_data: dict, persona_context: str) -> str:
        identity_context = self._identity_block()
        persona_context = str(persona_context or "").strip()
        if identity_context == "None" and not persona_context:
            return self._no_evidence_reply(user_input, thought_data, mode="self_intro")

        evidence = "\n\n".join(part for part in [identity_context, persona_context] if part and part != "None")
        prompt = self._build_prompt(
            task_label="回答自我介绍或“你是谁”这类问题",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text=evidence,
            constraints=[
                "身份事实优先使用基础身份背景，补充证据只能辅助，不要扩写成完整百科。",
                "如果资料不够，就只说能确定的部分，不要硬补过去经历。",
                "要像角色本人在介绍自己，不要写成档案、简历或第三人称概述。",
                "先直接回答“我是谁”，再自然补 1 到 2 个有助于认识这个角色的细节。",
                "不要写成谜语、反问游戏、舞台开场白、角色展示文案或刻意吊着对方的句式。",
                "不要为了显得有角色味，就把示例里的固定句型硬搬出来。",
                "可以自然展开，但不要被固定开场或固定收尾绑住。",
                "说完就自然落地，不要故意留成半截句或等待对方来猜。",
                "不要在说完后追加一句否定自己前文的回退句。",
            ],
        )
        return self._generate(prompt, max_tokens=1100, temperature=0.48, isolated=False)

    def external(self, user_input: str, thought_data: dict, tool_context: str) -> str:
        tool_context = str(tool_context or "").strip()
        if not tool_context or tool_context == "None":
            return self._no_evidence_reply(user_input, thought_data, mode="external")

        prompt = self._build_prompt(
            task_label="根据外部工具结果回答现实信息问题，并用角色底色自然表达",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text=tool_context,
            constraints=[
                "现实事实只能来自外部工具结果，不能补出新的事实。",
                "先判断哪些结果和用户问题真正相关，只保留相关部分。",
                "如果外部证据只提到某个对象，就不要顺手扩展到同领域的别人或别的作品。",
                "用角色本人的第一人称表达这些信息，不要写成机械播报。",
                "角色底色只负责语气、节奏、态度和距离感，不负责新增事实。",
                "不要用“我记得”“印象里”“大概”这类把无证据推断包装成事实的措辞。",
                "不要写括号动作、第三人称旁白、固定模板开头或分析痕迹。",
                "不要在答完后追加“我不太明白您的意思”“就这些”之类的反向收尾。",
            ],
        )
        response = self._generate(prompt, max_tokens=1300, temperature=0.52, isolated=False)
        if all(token not in response for token in ("我记得", "印象里", "没印象", "不清楚这个人")):
            return response

        retry_prompt = self._build_prompt(
            task_label="重新完成一次基于外部证据的回答，这次必须完全贴着证据说",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text=tool_context,
            constraints=[
                "只根据眼前这份外部证据回答，不要像在凭记忆回想。",
                "如果工具结果里已经有答案，就直接按结果说，不要再说自己没印象或不清楚。",
                "如果证据不足，就明确说暂时只查到这些，不要自己补细节。",
                "保持第一人称和角色底色，但事实边界必须收紧。",
            ],
        )
        return self._generate(retry_prompt, max_tokens=1250, temperature=0.46, isolated=False)

    def emotional(self, user_input: str, thought_data: dict) -> str:
        prompt = self._build_prompt(
            task_label="优先接住用户情绪，保持角色底色自然回应",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text="用户当前表达了情绪、压力、委屈、疲惫或低落，需要先被接住，再继续对话。",
            constraints=[
                "先回应情绪本身，再决定是否追问，不要急着转去讲设定、故事或外部知识。",
                "至少完成两步：先让用户感觉被听见，再给一句陪伴、安抚或温和推进。",
                "不要只回一句空洞的“是吗”“这样啊”“嗯”。",
                "保持第一人称和角色底色，但不要演成舞台旁白、括号动作或模板台词。",
                "不要自顾自讲自己的具体经历，也不要无端联网搜索。",
                "如果想继续推进对话，可以自然接一句温和追问或陪伴式回应，但不要机械反问。",
            ],
        )
        response = self._generate(prompt, max_tokens=900, temperature=0.58, isolated=False)
        stripped = re.sub(r"[。！？?!\s]", "", response)
        has_dialogue_address = any(token in response for token in ("你", "您", "我"))
        quotes_balanced = response.count('"') % 2 == 0 and response.count("“") == response.count("”")
        sentence_count = len(re.findall(r"[。！？?!]", response))
        question_only = sentence_count <= 1 and response.endswith(("吗？", "吗?", "呢？", "呢?"))
        if (
            len(stripped) >= 12
            and has_dialogue_address
            and quotes_balanced
            and sentence_count >= 1
            and not question_only
        ):
            return response

        retry_prompt = self._build_prompt(
            task_label="重新完成一次情绪回应，这次必须真正接住用户情绪",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text="上一版回答太薄或太像单纯追问，需要重新回答。",
            constraints=[
                "至少写成两句完整的话。",
                "第一句明确接住用户此刻的情绪，不要只用“嗯”“这样啊”“是吗”之类的空回应。",
                "第二句给出陪伴、安抚、理解或温和推进，让用户知道你愿意继续听。",
                "不要编造你自己的过去经历或具体事件来当安慰材料。",
                "仍然保持角色底色和直接对话口吻，不要突然分析、讲设定、讲故事或联网搜索。",
                "不要写成第三人称旁白、舞台说明、动作描写或带引号的小剧场。",
            ],
        )
        return self._generate(retry_prompt, max_tokens=950, temperature=0.62, isolated=False)

    def persona_focus(self, user_input: str, thought_data: dict, persona_context: str, focus: str) -> str:
        persona_context = str(persona_context or "").strip()
        if not persona_context or persona_context == "None":
            return self._no_evidence_reply(user_input, thought_data, mode="persona")

        focus_map = {
            "catchphrase": "只回答说话习惯、口头禅或固定表达方式。",
            "likes": "只回答明确喜欢或偏好的对象。",
            "dislikes": "只回答明确讨厌或回避的对象。",
            "personality": "只回答已有证据支持的性格特征与表现方式。",
            "self_intro": "只回答基本身份。",
        }
        prompt = self._build_prompt(
            task_label="回答一个角色设定相关的问题",
            user_input=user_input,
            thought_data=thought_data,
            evidence_text=persona_context,
            constraints=[
                focus_map.get(focus, "只回答当前问题真正涉及的那部分设定。"),
                "只依据当前命中的人设证据回答，不要补证据之外的新设定。",
                "如果证据只能支持高层概括，就保持高层，不要细化成新情节。",
                "保持第一人称和角色口吻，但不要演成括号旁白或舞台剧说明。",
                "避免用固定的自我定义短句做开场或收尾。",
            ],
        )
        return self._generate(prompt, max_tokens=1000, temperature=0.45, isolated=False)

    def postprocess(self, response: str, user_input: str = "") -> str:
        del user_input
        return self._sanitize_response(response)

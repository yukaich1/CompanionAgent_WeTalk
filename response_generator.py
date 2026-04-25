from __future__ import annotations

import copy
import re
from collections.abc import Mapping

from ai_runtime_support import build_persona_injection_prompt, relation_state_summary
from context.context_selector import ContextSelector
from prompting.evidence_pack_builder import EvidencePackBuilder
from prompting.prompt_composer import PromptComposer
from prompting.prompt_models import PromptSections
from prompting.response_planner import ResponsePlanner
from prompting.stable_prompt import StablePromptBuilder
from turn_runtime import DynamicStateView, EvidenceBundle, ResponsePlan, TurnRuntimeContext
from utils import format_memories_to_string


class ResponseGenerator:
    def __init__(self, system):
        self.system = system
        self.context_selector = ContextSelector()
        self.evidence_pack_builder = EvidencePackBuilder(system)
        self.response_planner = ResponsePlanner()
        self.prompt_composer = PromptComposer()
        self.stable_prompt_builder = StablePromptBuilder()

    def _is_memory_recall_prompt(self, user_input: str, response_mode: str) -> bool:
        mode = str(response_mode or "").strip()
        if mode not in {"casual", "value"}:
            return False
        text = str(user_input or "").strip()
        markers = (
            "还记得",
            "记得吗",
            "之前说过",
            "上次说过",
            "我刚刚说过",
            "我刚才说过",
            "我们聊过",
            "你说过",
            "我说过",
        )
        return any(marker in text for marker in markers)

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

    def _generate(self, prompt_body: str, max_tokens: int, temperature: float, isolated: bool = False) -> str:
        history = self._history_with_prompt(prompt_body, isolated=isolated)
        reply = self.system.model.generate(history, temperature=temperature, max_tokens=max_tokens)
        return self._sanitize_response(str(reply or ""))

    def _generate_with_continuation(
        self,
        prompt_body: str,
        max_tokens: int,
        temperature: float,
        continuation_max_tokens: int,
    ) -> str:
        history = self._history_with_prompt(prompt_body, isolated=False)
        first_pass_raw = str(self.system.model.generate(history, temperature=temperature, max_tokens=max_tokens) or "")
        if not self._looks_truncated(first_pass_raw):
            return self._sanitize_response(first_pass_raw)

        continuation_prompt = f"""
下面是一段还没完整收束的角色回答。请继续往下写，只补上自然收尾。

必须遵守：
1. 延续同一段回答的语气、视角和事实边界。
2. 不要重复前文，不要新增事实。
3. 只把这一段自然说完，不要写“继续”“补充说明”之类提示语。

原始任务：
{prompt_body}

当前已生成内容：
{first_pass_raw}
""".strip()
        continuation = str(
            self.system.model.generate(
                self._history_with_prompt(continuation_prompt, isolated=False),
                temperature=max(0.12, temperature - 0.05),
                max_tokens=continuation_max_tokens,
            )
            or ""
        )
        return self._sanitize_response(f"{first_pass_raw.rstrip()} {continuation.lstrip()}".strip())

    def _fallback_memory_snapshot(self) -> str:
        memory_views = self.system.memory_system.build_working_memory(self.system.get_message_history(False))
        rendered = format_memories_to_string(memory_views, "None")
        return self.system._truncate_for_prompt(rendered, 500)

    def _selected_memory_layers(self, response_mode: str, memory_slots: Mapping[str, object] | None = None) -> list[tuple[str, str]]:
        slots = memory_slots if isinstance(memory_slots, Mapping) else {}
        layer_values = {
            "L1 Stable Memory": str(slots.get("layer1_stable_memory", "") or "").strip(),
            "L2 Topic Recall": str(slots.get("layer2_topic_memory", "") or "").strip(),
            "L3 Deep Recall": str(slots.get("layer3_deep_memory", "") or "").strip(),
        }
        selected_by_mode = {
            "self_intro": ("L1 Stable Memory",),
            "casual": ("L1 Stable Memory", "L2 Topic Recall"),
            "persona_fact": ("L1 Stable Memory",),
            "story": ("L1 Stable Memory", "L3 Deep Recall"),
            "external": ("L1 Stable Memory",),
            "emotional": ("L1 Stable Memory", "L2 Topic Recall"),
            "value": ("L1 Stable Memory", "L2 Topic Recall"),
        }
        layer_order = selected_by_mode.get(str(response_mode or "casual").strip() or "casual", ("L1 Stable Memory",))
        selected = [(label, layer_values[label]) for label in layer_order if layer_values.get(label)]
        return selected

    def _memory_snapshot(self, response_mode: str, memory_slots: Mapping[str, object] | None = None) -> str:
        selected = self._selected_memory_layers(response_mode, memory_slots)
        if selected:
            rendered = "\n\n".join(f"=== {label} ===\n{body}" for label, body in selected if body)
            return self.system._truncate_for_prompt(rendered, 800)
        return self._fallback_memory_snapshot()

    def _user_emotion_hint(self, thought_data: dict) -> str:
        emotions = list((thought_data or {}).get("possible_user_emotions", []) or [])
        if emotions:
            return "用户可能正在感受：" + "、".join(str(item) for item in emotions if str(item).strip())
        return "用户没有表现出特别强的显性情绪。"

    def _dynamic_state(
        self,
        thought_data: dict | None = None,
        response_mode: str = "casual",
        memory_slots: Mapping[str, object] | None = None,
    ) -> DynamicStateView:
        thought_data = thought_data or {}
        return DynamicStateView(
            mood=str(thought_data.get("emotion", "平静") or "平静"),
            mood_detail=self.system.emotion_system.get_mood_prompt(),
            relation_state=relation_state_summary(self.system),
            user_emotion_hint=self._user_emotion_hint(thought_data),
            recent_dialogue=self._recent_dialogue_block(),
            memory_snapshot=self._memory_snapshot(response_mode, memory_slots),
            session_context=str(self.system.session_context_manager.render() or "").strip() or "None",
        )

    def _build_turn_context(
        self,
        *,
        user_input: str,
        thought_data: dict | None,
        response_mode: str,
        persona_focus: str,
        persona_context: str = "",
        tool_context: str = "",
        story_hits: list[dict] | None = None,
        response_contract: str = "",
        persona_focus_contract: str = "",
        memory_slots: Mapping[str, object] | None = None,
    ) -> TurnRuntimeContext:
        thought_data = thought_data or {}
        normalized_mode = str(response_mode or "casual").strip() or "casual"
        memory_sensitive = self._is_memory_recall_prompt(user_input, normalized_mode)
        style_prompt = str(build_persona_injection_prompt(self.system, thought_data) or "").strip() or "None"
        evidence, selected_memory_layers = self.evidence_pack_builder.build(
            response_mode=normalized_mode,
            user_input=user_input,
            persona_context=persona_context,
            tool_context=tool_context,
            story_hits=story_hits,
            memory_slots=memory_slots,
        )
        return TurnRuntimeContext(
            user_input=str(user_input or "").strip(),
            response_mode=normalized_mode,
            persona_focus=str(persona_focus or "general").strip() or "general",
            character_name=self.system.config.name,
            style_prompt=style_prompt,
            dynamic_state=self._dynamic_state(thought_data, normalized_mode, memory_slots),
            evidence=evidence,
            response_contract=str(response_contract or "").strip(),
            persona_focus_contract=str(persona_focus_contract or "").strip(),
            metadata={
                "thought_data": thought_data,
                "selected_memory_layers": selected_memory_layers,
                "memory_sensitive": memory_sensitive,
                "working_memory_preview": self._recent_dialogue_block(),
                "session_context_state": self.system.session_context_manager.build_session_context(),
                "persistence_boundary": {
                    "full_history_turns": len(self.system.get_message_history(False)),
                    "working_memory_available": self._memory_snapshot(normalized_mode, memory_slots).strip() not in {"", "None"},
                    "session_context_available": str(self.system.session_context_manager.render() or "").strip() not in {"", "None"},
                },
            },
        )

    def _fallback_response(self, context: TurnRuntimeContext, plan: ResponsePlan) -> str:
        prompt = self.prompt_composer.compose(self._build_prompt_sections(context, plan), fallback=True)
        return self._generate(prompt, max_tokens=min(plan.max_tokens, 420), temperature=0.38, isolated=False)

    def _build_prompt_sections(self, context: TurnRuntimeContext, plan: ResponsePlan) -> PromptSections:
        stable = self.stable_prompt_builder.build(
            character_name=context.character_name,
            style_prompt=context.style_prompt,
        )
        return PromptSections(
            stable=stable,
            turn_plan=plan,
            turn_context=context,
        )

    def missing_evidence_reply(self, *, response_mode: str, reason: str = "") -> str:
        mode = str(response_mode or "casual").strip() or "casual"
        reason = str(reason or "").strip()
        canned_replies = {
            "story": "这一段我现在还不能说得太确定。手边没有足够稳的故事依据，我不想顺手编成另一个版本。",
            "persona_fact": "我现在能确认的资料还不够多，所以不想把没有根据的话说得太满。",
            "self_intro": "我现在只能先说最确定的部分。再往外展开的话，就容易说得太满了。",
            "external": "这件事我现在还没拿到足够可靠的外部信息，所以不想把答案说得太死。",
            "casual": "这件事我现在没有确切记住，所以不想装作自己已经接住了。",
            "emotional": "我在听，但这部分我现在不想拿不确定的内容来敷衍你。",
            "value": "我可以有自己的态度，但在依据不够的时候，我不想把话说成定论。",
        }
        if reason == "memory_missing":
            return "这件事我现在没有确切记住，所以不想装作自己已经接住了。"
        return canned_replies.get(mode, "这件事我现在还不能说得太确定，所以不想把没有根据的话说满。")

    def reply(
        self,
        *,
        user_input: str,
        thought_data: dict | None,
        response_mode: str,
        persona_focus: str = "general",
        persona_context: str = "",
        tool_context: str = "",
        story_hits: list[dict] | None = None,
        response_contract: str = "",
        persona_focus_contract: str = "",
        memory_slots: Mapping[str, object] | None = None,
    ) -> str:
        context = self._build_turn_context(
            user_input=user_input,
            thought_data=thought_data,
            response_mode=response_mode,
            persona_focus=persona_focus,
            persona_context=persona_context,
            tool_context=tool_context,
            story_hits=story_hits,
            response_contract=response_contract,
            persona_focus_contract=persona_focus_contract,
            memory_slots=memory_slots,
        )
        context = self.context_selector.select(context)
        plan = self.response_planner.build(context)
        self.system.last_debug_info = {
            **(self.system.last_debug_info or {}),
            "responsePlan": {
                "mode": context.response_mode,
                "personaFocus": context.persona_focus,
                "task": plan.task_label,
                "evidenceKind": plan.evidence_kind,
                "evidenceRequired": plan.evidence_required,
                "evidenceReady": plan.evidence_ready,
                "evidenceSources": context.evidence.active_sources(),
                "selectedEvidenceSources": [
                    *([plan.evidence_kind] if plan.evidence_kind else []),
                    *(
                        ["working_memory", "session_context"]
                        if bool(context.metadata.get("memory_sensitive"))
                        else []
                    ),
                ],
                "evidencePreview": self.system._truncate_for_prompt(plan.evidence_text, 220),
                "memoryLayers": list(context.metadata.get("selected_memory_layers", []) or []),
            },
            "selectedContextView": dict((context.metadata or {}).get("selected_context_view", {}) or {}),
            "persistenceBoundary": dict((context.metadata or {}).get("persistence_boundary", {}) or {}),
        }

        if plan.evidence_required and not plan.evidence_ready:
            return self._fallback_response(context, plan)

        prompt = self.prompt_composer.compose(self._build_prompt_sections(context, plan), fallback=False)
        if plan.allow_continuation:
            return self._generate_with_continuation(
                prompt,
                max_tokens=plan.max_tokens,
                temperature=plan.temperature,
                continuation_max_tokens=plan.continuation_max_tokens,
            )
        return self._generate(prompt, max_tokens=plan.max_tokens, temperature=plan.temperature, isolated=False)

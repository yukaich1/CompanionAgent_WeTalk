"""AI 主编排模块。"""

from __future__ import annotations

import json
import re
from collections import deque
from datetime import datetime
from uuid import uuid4

import requests
from pydantic import BaseModel, Field

from ai_runtime_support import (
    build_format_data,
    derive_relation_impact,
    derive_topic_tags,
    estimate_pending_signal,
    record_debug_info,
    split_tool_context_by_mode,
    thought_signal,
)
from ai_system_store import AISystemStore
from config import DEFAULT_CONFIG
from const import AI_SYSTEM_PROMPT, NEW_MEMORY_STATE_PATH, NEW_PERSONA_STATE_PATH, USER_TEMPLATE
from context.context_assembler import ContextAssembler
from context.recall_deduplicator import RecallDeduplicator
from diagnostics.conflict_log import ConflictLog
from diagnostics.health_monitor import HealthMonitor
from diagnostics.self_check import SelfCheck
from knowledge.knowledge_source import PersonaRecallResult
from knowledge.persona_rag_engine import PersonaRAGEngine
from knowledge.persona_response_policy import PersonaResponsePolicy
from knowledge.persona_system import CoreTrait, IdentityProfile, ParentChunk, PersonaState, PersonaSystem, PersonaSystemStore
from llm import FallbackMistralLLM, MistralLLM, get_active_llm_label, get_llm_settings, has_llm_api_key
from memory.memory_rag_engine import MemoryRAGEngine
from memory.memory_system import MemorySystem
from memory.memory_writer import MemoryWriter
from memory.state_models import MemorySystemState, MemorySystemStore
from reasoning.emotion_state_machine import EmotionStateMachine, EmotionSystem, PersonalitySystem, RelationshipSystem
from reasoning.thought_system import ThoughtSystem
from response_generator import ResponseGenerator
from routing.query_router import QueryRouter
from tools import DEFAULT_TOOL_REGISTRY, ToolRuntime
from utils import conversation_to_string


class MessageBuffer:
    def __init__(self, max_messages: int):
        self.messages = deque(maxlen=max_messages)
        self.system_prompt = ""

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = str(prompt or "").strip()

    def add_message(self, role: str, content) -> None:
        self.messages.append({"role": role, "content": content})

    def flush(self) -> None:
        self.messages.clear()

    def to_list(self, include_system_prompt: bool = True) -> list[dict]:
        history = []
        if include_system_prompt and self.system_prompt:
            history.append({"role": "system", "content": self.system_prompt})
        history.extend(message.copy() for message in self.messages)
        return history


class PersonalityConfig(BaseModel):
    open: float = Field(ge=-1.0, le=1.0)
    conscientious: float = Field(ge=-1.0, le=1.0)
    agreeable: float = Field(ge=-1.0, le=1.0)
    extrovert: float = Field(ge=-1.0, le=1.0)
    neurotic: float = Field(ge=-1.0, le=1.0)


class AIConfig(BaseModel):
    name: str = Field(default="Ireina")
    system_prompt: str = Field(default=AI_SYSTEM_PROMPT)
    personality: PersonalityConfig = Field(
        default_factory=lambda: PersonalityConfig(
            open=0.35,
            conscientious=0.22,
            extrovert=0.18,
            agreeable=0.93,
            neurotic=-0.1,
        )
    )


class AISystem:
    def __init__(self, config: AIConfig | None = None):
        self.config = config or AIConfig()
        self.runtime_config = DEFAULT_CONFIG
        self.buffer = MessageBuffer(20)
        self.buffer.set_system_prompt(self.config.system_prompt)

        self.personality_system = PersonalitySystem(
            openness=self.config.personality.open,
            conscientious=self.config.personality.conscientious,
            extrovert=self.config.personality.extrovert,
            agreeable=self.config.personality.agreeable,
            neurotic=self.config.personality.neurotic,
        )
        self.relation_system = RelationshipSystem()
        self.emotion_system = EmotionSystem(self.personality_system, self.relation_system, self.config)
        self.persona_system = PersonaSystem(persona_name=self.config.name)
        self.persona_policy = PersonaResponsePolicy(self.persona_system, lambda: self.config.name)
        self.tool_runtime = ToolRuntime(DEFAULT_TOOL_REGISTRY)
        self.model = FallbackMistralLLM()

        self.conflict_log = ConflictLog()
        self.health_monitor = HealthMonitor()
        self.self_check = SelfCheck()
        self.new_memory_store = MemorySystemStore(NEW_MEMORY_STATE_PATH)
        self.new_persona_store = PersonaSystemStore(NEW_PERSONA_STATE_PATH)
        self.new_memory_state = MemorySystemState()
        self.new_persona_state = PersonaState()
        self.memory_writer = MemoryWriter(self.conflict_log, self.model)
        self.memory_rag_engine = MemoryRAGEngine(self.model)
        self.memory_system = MemorySystem(self.config, self.new_memory_state, self.memory_writer, self.memory_rag_engine)
        self.thought_system = ThoughtSystem(self.config, self.emotion_system, self.memory_system, self.relation_system, self.personality_system)
        self.persona_rag_engine = PersonaRAGEngine(self.persona_system, self.new_persona_state)
        self.query_router = QueryRouter()
        self.context_assembler = ContextAssembler()
        self.recall_deduplicator = RecallDeduplicator()
        self.emotion_state_machine = EmotionStateMachine()
        self.response_generator = ResponseGenerator(self)
        self.store = AISystemStore(self)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]

        self.num_messages = 0
        self.last_message = None
        self.last_tick = datetime.now()
        self.last_debug_info = {}
        self.recent_story_chunk_ids = deque(maxlen=6)

        self._load_architecture_state()
        self._sync_persona_state()

    def set_config(self, config: AIConfig) -> None:
        self.config = config
        self.buffer.set_system_prompt(config.system_prompt)
        self.persona_system.persona_name = config.name
        self._sync_persona_state()

    def on_startup(self) -> None:
        self.self_check.run()

    def _sync_persona_state(self) -> None:
        persona = self.persona_system
        state = self.new_persona_state or PersonaState()
        base_template = getattr(persona, "base_template", {}) or {}
        profile = (base_template.get("00_BACKGROUND", {}) or {}).get("profile", {}) or {}
        state.immutable_core.identity = IdentityProfile(
            name=str(profile.get("full_name") or profile.get("name") or self.config.name),
            aliases=list(profile.get("aliases", []) or []),
            species=str(profile.get("species") or ""),
            archetype=str(profile.get("archetype") or ""),
        )
        state.immutable_core.core_traits = [
            CoreTrait(feature=item, activation_trigger=[item], mandatory_behavior="", evidence_tags=[item])
            for item in list(getattr(persona, "display_keywords", []) or [])[:12]
        ]

        def _bounded_importance(raw_value: object) -> float:
            try:
                score = float(raw_value if raw_value is not None else 0.5)
            except (TypeError, ValueError):
                score = 0.5
            return max(0.0, min(1.0, score))

        state.evidence_vault.parent_chunks = [
            ParentChunk(
                chunk_id=str(entry.get("chunk_id") or ""),
                content=str(entry.get("content", "") or "").strip(),
                topic_tags=list(entry.get("keywords", []) or []),
                trait_tags=list(entry.get("keywords", []) or []),
                importance_score=_bounded_importance(entry.get("priority", 0.5)),
                kind=str(entry.get("kind") or "source_chunk"),
                title=str(entry.get("title", "") or ""),
                metadata=dict(entry.get("metadata", {}) or {}),
            )
            for entry in list(getattr(persona, "entries", []) or [])
            if str(entry.get("content", "") or "").strip()
        ]
        state.metadata = {
            "voice_card": getattr(persona, "character_voice_card", "") or "",
            "display_keywords": list(getattr(persona, "display_keywords", []) or []),
            "style_examples": list(getattr(persona, "style_examples", []) or []),
            "base_template": base_template,
        }
        self.new_persona_state = state
        self.persona_rag_engine.set_persona_state(state)

    def _load_architecture_state(self) -> None:
        try:
            self.new_memory_state = self.new_memory_store.load()
        except Exception:
            self.new_memory_state = MemorySystemState()
        try:
            self.new_persona_state = self.new_persona_store.load()
        except Exception:
            self.new_persona_state = PersonaState()

        self.memory_system.set_state(self.new_memory_state)
        metadata = self.new_persona_state.metadata or {}
        if metadata:
            self.persona_system.display_keywords = list(metadata.get("display_keywords", []) or [])
            self.persona_system.style_examples = list(metadata.get("style_examples", []) or [])
            self.persona_system.character_voice_card = str(metadata.get("voice_card", "") or "")
            if isinstance(metadata.get("base_template"), dict):
                self.persona_system.base_template = metadata["base_template"]
        self.persona_rag_engine.set_persona_state(self.new_persona_state)

    def _save_architecture_state(self) -> None:
        self._sync_persona_state()
        self.new_memory_store.save(self.new_memory_state)
        self.new_persona_store.save(self.new_persona_state)

    def get_message_history(self, include_system_prompt: bool = True) -> list[dict]:
        return self.buffer.to_list(include_system_prompt=include_system_prompt)

    def _recent_assistant_messages(self, limit: int = 3) -> list[str]:
        messages = [message.get("content", "") for message in list(self.buffer.messages) if message.get("role") == "assistant"]
        return [str(item or "") for item in messages[-limit:]]

    def _truncate_for_prompt(self, text, limit: int = 400) -> str:
        value = str(text or "").strip()
        return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "..."

    def _has_tool_evidence(self, tool_context: str) -> bool:
        value = str(tool_context or "").strip()
        return bool(value and value != "None")

    def _persona_foundation_available(self) -> bool:
        persona = self.persona_system
        entries = list(getattr(persona, "entries", []) or [])
        return bool(
            entries
            or getattr(persona, "character_voice_card", "").strip()
            or getattr(persona, "display_keywords", [])
            or any(payload for payload in (getattr(persona, "base_template", {}) or {}).values())
        )

    def _run_pipeline(self, user_input: str):
        normalized_query = re.sub(r"\s+", " ", str(user_input or "").strip())
        recent_conversation = conversation_to_string(self.get_message_history(False)[-6:])
        intent_result = self.query_router.extractor.extract(
            user_input=normalized_query,
            recent_conversation=recent_conversation,
            character_name=self.config.name,
        )
        recall_query_parts = [
            str(intent_result.extracted_topic or "").strip(),
            *[str(item or "").strip() for item in list(intent_result.extracted_keywords or [])],
        ]
        recall_query = " ".join(part for part in recall_query_parts if part).strip() or normalized_query

        if str(getattr(intent_result, "recall_mode", "none") or "none") == "none":
            persona_recall = PersonaRecallResult(
                metadata={"query_plan": {"direct_query": "", "query_type": "external", "multi_queries": []}, "hits": [], "story_hits": []}
            )
        else:
            recall_mode = str(getattr(intent_result, "recall_mode", "none") or "none").strip()
            preferred_query_type = "general"
            if recall_mode == "story":
                preferred_query_type = "story"
            elif recall_mode in {"identity", "persona"}:
                preferred_query_type = "persona"
            exclude_chunk_ids = list(self.recent_story_chunk_ids) if str(getattr(intent_result, "response_mode", "") or "") == "story" else []
            persona_recall = self.persona_rag_engine.recall(
                recall_query,
                exclude_chunk_ids=exclude_chunk_ids,
                preferred_query_type=preferred_query_type,
            )

        route_decision = self.query_router.route(
            normalized_query,
            persona_recall,
            is_public=True,
            recent_conversation=recent_conversation,
            character_name=self.config.name,
            intent_result=intent_result,
        )
        memory_result = self.memory_rag_engine.recall(normalized_query, self.new_memory_state)
        tool_report = self.tool_runtime.execute(intent_result=intent_result, persona_name=self.config.name)
        deduped = self.recall_deduplicator.dedup(persona_recall, memory_result)
        web_persona_context, web_reality_context = split_tool_context_by_mode(tool_report, route_decision)
        assembled = self.context_assembler.assemble(
            route_type=route_decision.type,
            deduped=deduped,
            web_persona_context=web_persona_context,
            web_reality_context=web_reality_context,
        )
        return route_decision, intent_result, tool_report, assembled, persona_recall

    def _build_grounding_contract(self, intent_result, persona_recall) -> dict:
        response_mode = str(getattr(intent_result, "response_mode", "casual") or "casual")
        persona_focus = str(getattr(intent_result, "persona_focus", "general") or "general")
        return {
            "response_mode": response_mode,
            "persona_focus": persona_focus,
            "has_identity_reference": self.persona_policy.has_identity_reference(),
            "story_hits": list((getattr(persona_recall, "metadata", {}) or {}).get("story_hits", []) or []),
            "response_contract": {
                "self_intro": "只使用身份背景和命中的身份证据回答，不扩写没有证据的新经历。",
                "story": "只使用一个命中的完整故事块回答，不合并多个故事，不补写新细节。",
                "persona_fact": "只依据命中的角色资料回答喜好、性格、习惯或价值判断。",
                "external": "只依据工具返回的现实信息回答，并用角色底色自然表达。",
                "emotional": "优先接住用户情绪，保持角色说话方式，不强行填设定。",
                "value": "允许表达角色态度，但不要虚构具体角色经历作为论据。",
                "casual": "自然聊天，延续角色底色，不补写没有根据的事实。",
            }.get(response_mode, "自然回答，不虚构没有根据的事实。"),
            "persona_focus_contract": {
                "likes": "只回答明确喜欢或偏好的对象。",
                "dislikes": "只回答明确讨厌或回避的对象。",
                "catchphrase": "只回答说话习惯、口头禅或固定句式。",
                "personality": "只回答已有证据支持的性格特征。",
                "self_intro": "只回答基本身份。",
            }.get(persona_focus, ""),
        }

    def _memory_sensitive_request(self, user_input: str, intent_result) -> bool:
        response_mode = str(getattr(intent_result, "response_mode", "casual") or "casual").strip()
        if response_mode in {"self_intro", "story", "persona_fact", "external", "emotional"}:
            return False

        text = str(user_input or "").strip()
        if not text:
            return False

        memory_markers = (
            "还记得",
            "记得吗",
            "之前说过",
            "上次说过",
            "你说过",
            "我说过",
            "以前",
            "之前",
            "上次",
            "我们聊过",
            "你还知道",
            "我的喜好",
            "我喜欢什么",
            "我讨厌什么",
            "我是不是",
        )
        return any(marker in text for marker in memory_markers)

    def _memory_evidence_available(self, assembled_context) -> bool:
        if not assembled_context:
            return False
        slots = getattr(assembled_context, "slots", {}) or {}
        return bool(
            str(slots.get("layer1_stable_memory", "") or "").strip()
            or str(slots.get("layer2_topic_memory", "") or "").strip()
            or str(slots.get("layer3_deep_memory", "") or "").strip()
        )

    def _evaluate_evidence_status(
        self,
        user_input: str,
        intent_result,
        grounding: dict,
        tool_context: str,
        persona_context: str,
        assembled_context,
        story_hits: list | None = None,
    ) -> dict:
        response_mode = str(getattr(intent_result, "response_mode", "casual") or "casual").strip()
        story_hits = list(story_hits or [])
        persona_context_ok = bool(str(persona_context or "").strip())
        identity_ok = bool(grounding.get("has_identity_reference"))
        memory_ok = self._memory_evidence_available(assembled_context)
        tool_ok = bool(str(tool_context or "").strip())

        if response_mode == "external":
            return {"required": True, "ready": tool_ok, "reason": "tool" if tool_ok else "tool_missing"}
        if response_mode == "story":
            return {
                "required": True,
                "ready": bool(story_hits),
                "reason": "story_hit" if story_hits else "story_missing",
            }
        if response_mode == "self_intro":
            return {
                "required": True,
                "ready": identity_ok or persona_context_ok,
                "reason": "identity" if (identity_ok or persona_context_ok) else "identity_missing",
            }
        if response_mode == "persona_fact":
            return {
                "required": True,
                "ready": persona_context_ok,
                "reason": "persona" if persona_context_ok else "persona_missing",
            }
        if self._memory_sensitive_request(user_input, intent_result):
            return {
                "required": True,
                "ready": memory_ok,
                "reason": "memory" if memory_ok else "memory_missing",
            }
        return {"required": False, "ready": True, "reason": "not_required"}

    def _choose_response(
        self,
        user_input: str,
        thought_data: dict,
        grounding: dict,
        tool_context: str,
        persona_context: str,
        persona_recall,
        intent_result,
        assembled_context=None,
    ) -> str:
        response_mode = str(getattr(intent_result, "response_mode", "casual") or "casual").strip()
        persona_focus = str(getattr(intent_result, "persona_focus", "general") or "general").strip()
        story_hits = list((getattr(persona_recall, "metadata", {}) or {}).get("story_hits", []) or [])
        evidence_status = self._evaluate_evidence_status(
            user_input=user_input,
            intent_result=intent_result,
            grounding=grounding,
            tool_context=tool_context,
            persona_context=persona_context,
            assembled_context=assembled_context,
            story_hits=story_hits,
        )
        self.last_debug_info = {
            **(self.last_debug_info or {}),
            "evidenceGate": evidence_status,
        }

        if response_mode == "self_intro":
            if not evidence_status["ready"]:
                return self.response_generator._no_evidence_reply(str(user_input or ""), thought_data, mode="self_intro")
            return self.response_generator.self_intro(str(user_input or ""), thought_data, persona_context)

        if response_mode == "story":
            if not evidence_status["ready"]:
                return self.response_generator._no_evidence_reply(str(user_input or ""), thought_data, mode="story")
            return self.response_generator.story(str(user_input or ""), thought_data, story_hits[0] if story_hits else {})

        if response_mode == "external":
            if not evidence_status["ready"]:
                return self.response_generator._no_evidence_reply(str(user_input or ""), thought_data, mode="external")
            return self.response_generator.external(str(user_input or ""), thought_data, tool_context)

        if response_mode == "emotional":
            return self.response_generator.emotional(str(user_input or ""), thought_data)

        if response_mode == "persona_fact":
            if not evidence_status["ready"]:
                return self.response_generator._no_evidence_reply(str(user_input or ""), thought_data, mode="persona")
            return self.response_generator.persona_focus(str(user_input or ""), thought_data, persona_context, persona_focus)
        if response_mode == "value" and evidence_status["required"] and not evidence_status["ready"]:
            return self.response_generator._no_evidence_reply(str(user_input or ""), thought_data, mode="general")
        if response_mode == "casual":
            if evidence_status["required"] and not evidence_status["ready"]:
                return self.response_generator._no_evidence_reply(str(user_input or ""), thought_data, mode="general")
            return self.response_generator.casual(
                str(user_input or ""),
                thought_data,
                persona_context=persona_context,
                tool_context=tool_context,
            )
        prompt_content = USER_TEMPLATE.format(
            **build_format_data(
                self,
                user_input,
                thought_data,
                self.memory_system.build_working_memory(self.get_message_history(False)),
                persona_context,
                tool_context,
                grounding=grounding,
            )
        )
        request_history = self.response_generator._history_with_prompt(prompt_content)
        self.last_debug_info = {
            **(self.last_debug_info or {}),
            "requestMetrics": {
                "userInputChars": len(str(user_input or "")),
                "personaContextChars": len(str(persona_context or "")),
                "toolContextChars": len(str(tool_context or "")),
                "promptChars": len(prompt_content),
                "historyMessages": len(request_history),
            },
        }
        response = self.model.generate(request_history, temperature=0.3 if response_mode == "story" else 0.8, max_tokens=2000)
        return self.response_generator.postprocess(response, user_input=str(user_input or ""))

    def send_message(self, user_input, return_json: bool = False, attached_image=None):
        self.num_messages += 1
        self.buffer.add_message("user", user_input)
        recent_conversation = conversation_to_string(self.get_message_history(False)[-6:])

        route_decision, intent_result, tool_report, assembled_context, persona_recall = self._run_pipeline(str(user_input or ""))
        if tool_report.follow_up_message:
            self.buffer.add_message("assistant", tool_report.follow_up_message)
            return tool_report.follow_up_message

        tool_context_for_turn = assembled_context.slots.get("web_reality_context") or assembled_context.slots.get("web_persona_context") or ""
        persona_context = assembled_context.slots.get("evidence_chunks", "")
        thought_persona_context = self.context_assembler.build_prompt_context(assembled_context)
        grounding = self._build_grounding_contract(intent_result, persona_recall)

        pending_signal = estimate_pending_signal(
            self,
            str(user_input or ""),
            recent_conversation=recent_conversation,
        )
        self.emotion_state_machine.queue_signal(pending_signal)
        memories, recalled_memories = self.memory_system.recall_memories(self.get_message_history(False))
        thought_data = self.thought_system.think(
            messages=self.get_message_history(False),
            memories=memories,
            recalled_memories=recalled_memories,
            last_message=self.last_message,
            persona_context=thought_persona_context,
        )
        self.emotion_state_machine.update_from_thought(thought_signal(thought_data))
        record_debug_info(
            self,
            route_decision,
            assembled_context,
            thought_data=thought_data,
            intent_result=intent_result,
            persona_recall=persona_recall,
        )

        response = self._choose_response(
            user_input,
            thought_data,
            grounding,
            tool_context_for_turn,
            persona_context,
            persona_recall,
            intent_result,
            assembled_context=assembled_context,
        )
        if not response:
            response = "我现在没法把这件事说得太确定。"
        if str(getattr(intent_result, "response_mode", "") or "") == "story":
            story_hits = list((getattr(persona_recall, "metadata", {}) or {}).get("story_hits", []) or [])
            if story_hits:
                chunk_id = str(story_hits[0].get("chunk_id", "") or "").strip()
                if chunk_id:
                    self.recent_story_chunk_ids.append(chunk_id)

        self.memory_writer.remember(
            self.new_memory_state,
            summary=self._truncate_for_prompt(
                f"用户提到：{user_input}；{self.config.name} 回应：{response}",
                280,
            ),
            user_text=str(user_input or ""),
            assistant_text=str(response or ""),
            topic_tags=derive_topic_tags(str(user_input or "")),
            relation_impact=derive_relation_impact(pending_signal),
            importance=0.55,
            character_emotion=str(thought_data.get("emotion", "平静")),
            scope="",
            source_session_id=self.session_id,
            source_turn_index=self.num_messages,
        )
        self.memory_system.set_state(self.new_memory_state)
        self._sync_persona_state()

        self.last_message = datetime.now()
        self.tick()
        evidence_gate = (self.last_debug_info or {}).get("evidenceGate", {})
        self.health_monitor.record_turn_metrics(
            coverage_score=assembled_context.metadata.get("coverage_score", 0.0),
            evidence_backed=bool(evidence_gate.get("ready", False)),
        )
        self.buffer.add_message("assistant", response)
        return json.dumps(response, ensure_ascii=False, indent=2) if return_json else response

    def set_thought_visibility(self, shown: bool) -> None:
        self.thought_system.show_thoughts = shown

    def get_mood(self):
        return self.emotion_system.mood

    def get_beliefs(self):
        stable_notes = [
            record.content
            for record in getattr(self.new_memory_state, "stable_records", [])
            if getattr(record, "scope", "") == "USER_SPECIFIC" and getattr(record, "content", "")
        ]
        return stable_notes or self.memory_system.get_beliefs()

    def set_mood(self, pleasure=None, arousal=None, dominance=None) -> None:
        if pleasure is None and arousal is None and dominance is None:
            self.emotion_system.reset_mood()
        else:
            self.emotion_system.set_emotion(pleasure=pleasure, arousal=arousal, dominance=dominance)

    def set_relation(self, friendliness=None, dominance=None) -> None:
        self.relation_system.set_relation(friendliness=friendliness, dominance=dominance)

    def get_memories(self):
        return self.memory_system.build_working_memory(self.get_message_history(False))

    def consolidate_memories(self) -> None:
        self.memory_system.consolidate_memories()

    def tick(self) -> None:
        now = datetime.now()
        delta = (now - self.last_tick).total_seconds()
        self.emotion_system.tick(delta)
        if self.thought_system.can_reflect():
            self.thought_system.reflect()
        self.memory_system.tick(delta)
        self.last_tick = now

    def _config_payload(self):
        return self.config.model_dump() if hasattr(self.config, "model_dump") else self.config.dict()

    @staticmethod
    def _config_from_payload(payload):
        return AIConfig.model_validate(payload or {}) if hasattr(AIConfig, "model_validate") else AIConfig.parse_obj(payload or {})

    def _state_snapshot(self):
        return self.store.snapshot()

    def _restore_snapshot(self, payload):
        self.store.restore(payload)

    def save(self, path):
        self.store.save(path)

    @staticmethod
    def load(path):
        store_payload = AISystemStore.__new__(AISystemStore)
        payload = AISystemStore.load_payload(store_payload, path)
        if payload is None:
            return None
        ai_system = AISystem(config=AISystem._config_from_payload(payload.get("config", {})))
        ai_system._restore_snapshot(payload)
        ai_system._load_architecture_state()
        return ai_system

    @classmethod
    def load_or_create(cls, path):
        ai_system = cls.load(path)
        if ai_system is None:
            ai_system = AISystem()
        ai_system.on_startup()
        return ai_system


def check_has_valid_key():
    settings = get_llm_settings()
    if not has_llm_api_key():
        print(f"使用 Ireina 需要提供 {get_active_llm_label()} API 密钥。")
        print("请在项目同目录创建 .env 文件，并写入 LLM_PROVIDER、LLM_API_KEY、LLM_CHAT_MODEL 等配置。")
        return False
    print(f"正在验证 {get_active_llm_label()} API 密钥...")
    model = MistralLLM(settings.chat_model or None)
    try:
        model.generate("Hello", max_tokens=1, timeout=20, max_tries=2)
    except requests.HTTPError as exc:
        if exc.response.status_code == 401:
            print("提供的 API 密钥无效，请检查 .env 文件中的配置后重新运行。")
            return False
        print("暂时无法完成在线验证，将先启动程序，真正发送消息时再重试。")
        return True
    except requests.RequestException:
        print("网络连接出现问题，将先启动程序，真正发送消息时再重试。")
        return True
    except RuntimeError as exc:
        print(f"当前 LLM 配置不可用：{exc}")
        return False
    print("验证成功。")
    return True

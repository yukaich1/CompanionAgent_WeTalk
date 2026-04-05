"""AI 主编排模块。"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from datetime import datetime

import requests
from pydantic import BaseModel, Field

from ai_runtime_support import build_format_data, derive_relation_impact, derive_topic_tags, estimate_pending_signal, record_debug_info, split_tool_context_by_mode, thought_signal
from config import DEFAULT_CONFIG
from const import AI_SYSTEM_PROMPT, NEW_MEMORY_STATE_PATH, NEW_PERSONA_STATE_PATH, USER_TEMPLATE
from context.context_assembler import ContextAssembler
from context.recall_deduplicator import RecallDeduplicator
from diagnostics.conflict_log import ConflictLog
from diagnostics.health_monitor import HealthMonitor
from diagnostics.self_check import SelfCheck
from knowledge.knowledge_source import KnowledgeSource
from knowledge.persona_rag_engine import PersonaRAGEngine
from knowledge.persona_system import CoreTrait, IdentityProfile, ParentChunk, PersonaState, PersonaSystem, PersonaSystemStore
from llm import FallbackMistralLLM, MistralLLM, get_active_llm_label, get_llm_settings, has_llm_api_key
from memory.memory_rag_engine import MemoryRAGEngine
from memory.memory_system import MemorySystem
from memory.memory_writer import MemoryWriter
from memory.state_models import MemorySystemState, MemorySystemStore
from reasoning.emotion_state_machine import EmotionStateMachine, EmotionSystem, PersonalitySystem, RelationshipSystem
from reasoning.thought_system import ThoughtSystem
from routing.query_rewriter import QueryRewriter
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
    personality: PersonalityConfig = Field(default_factory=lambda: PersonalityConfig(open=0.35, conscientious=0.22, extrovert=0.18, agreeable=0.93, neurotic=-0.1))


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
        self.memory_system = MemorySystem(self.config)
        self.emotion_system = EmotionSystem(self.personality_system, self.relation_system, self.config)
        self.thought_system = ThoughtSystem(self.config, self.emotion_system, self.memory_system, self.relation_system, self.personality_system)
        self.persona_system = PersonaSystem(persona_name=self.config.name)
        self.tool_runtime = ToolRuntime(DEFAULT_TOOL_REGISTRY)
        self.model = FallbackMistralLLM()

        self.conflict_log = ConflictLog()
        self.health_monitor = HealthMonitor()
        self.self_check = SelfCheck()
        self.new_memory_store = MemorySystemStore(NEW_MEMORY_STATE_PATH)
        self.new_persona_store = PersonaSystemStore(NEW_PERSONA_STATE_PATH)
        self.new_memory_state = MemorySystemState()
        self.new_persona_state = PersonaState()
        self.memory_rag_engine = MemoryRAGEngine()
        self.memory_writer = MemoryWriter(self.conflict_log)
        self.persona_rag_engine = PersonaRAGEngine(self.persona_system, self.new_persona_state)
        self.query_router = QueryRouter()
        self.query_rewriter = QueryRewriter()
        self.context_assembler = ContextAssembler()
        self.recall_deduplicator = RecallDeduplicator()
        self.emotion_state_machine = EmotionStateMachine()

        self.num_messages = 0
        self.last_message = None
        self.last_tick = datetime.now()
        self.last_debug_info = {}

        self._load_new_architecture_state()
        self._sync_new_persona_state()

    def set_config(self, config: AIConfig) -> None:
        self.config = config
        self.buffer.set_system_prompt(config.system_prompt)
        self.persona_system.persona_name = config.name
        self._sync_new_persona_state()

    def on_startup(self) -> None:
        self.self_check.run()

    def _sync_new_persona_state(self) -> None:
        persona = self.persona_system
        state = self.new_persona_state or PersonaState()
        base_template = getattr(persona, "base_template", {}) or {}
        profile = (base_template.get("00_BACKGROUND_PROFILE", {}) or {}).get("profile", {}) or {}
        state.immutable_core.identity = IdentityProfile(
            name=str(profile.get("full_name") or profile.get("name") or self.config.name),
            aliases=list(profile.get("aliases", []) or []),
            species=str(profile.get("species") or ""),
            archetype=str(profile.get("archetype") or ""),
        )
        state.immutable_core.core_traits = [CoreTrait(feature=item, activation_trigger=[item], mandatory_behavior="", evidence_tags=[item]) for item in list(getattr(persona, "selected_keywords", []) or getattr(persona, "display_keywords", []) or [])[:12]]
        parent_chunks = []
        for index, entry in enumerate(getattr(persona, "entries", []) or []):
            content = str(entry.get("text", "") or "").strip()
            if not content:
                continue
            try:
                importance_score = float(entry.get("priority", 0.5) or 0.5)
            except Exception:
                importance_score = 0.5
            importance_score = max(0.0, min(1.0, importance_score))
            parent_chunks.append(
                ParentChunk(
                    chunk_id=str(entry.get("id") or f"entry-{index}"),
                    content=content,
                    source_level=KnowledgeSource.USER_CANON,
                    topic_tags=list(entry.get("keywords", []) or []),
                    trait_tags=list(entry.get("keywords", []) or []),
                    importance_score=importance_score,
                )
            )
        state.evidence_vault.parent_chunks = parent_chunks
        state.metadata = {
            "voice_card": getattr(persona, "character_voice_card", "") or "",
            "display_keywords": list(getattr(persona, "display_keywords", []) or []),
            "selected_keywords": list(getattr(persona, "selected_keywords", []) or []),
            "style_examples": list(getattr(persona, "style_examples", []) or []),
            "story_titles": [str(item.get("title", "") or "").strip() for item in list(getattr(persona, "story_chunks", []) or []) if str(item.get("title", "") or "").strip()],
            "story_chunks": list(getattr(persona, "story_chunks", []) or []),
            "base_template": base_template,
        }
        self.new_persona_state = state
        self.persona_rag_engine.set_persona_state(state)

    def _load_new_architecture_state(self) -> None:
        try:
            self.new_memory_state = self.new_memory_store.load()
        except Exception:
            self.new_memory_state = MemorySystemState()
        try:
            self.new_persona_state = self.new_persona_store.load()
        except Exception:
            self.new_persona_state = PersonaState()
        if self.new_persona_state.metadata and not self.persona_system.display_keywords:
            metadata = self.new_persona_state.metadata
            self.persona_system.display_keywords = list(metadata.get("display_keywords", []) or [])
            self.persona_system.selected_keywords = list(metadata.get("selected_keywords", []) or [])
            self.persona_system.style_examples = list(metadata.get("style_examples", []) or [])
            self.persona_system.story_chunks = list(metadata.get("story_chunks", []) or [])
            self.persona_system.character_voice_card = str(metadata.get("voice_card", "") or "")
            if isinstance(metadata.get("base_template"), dict):
                self.persona_system.base_template = metadata["base_template"]
        self.persona_rag_engine.set_persona_state(self.new_persona_state)

    def _save_new_architecture_state(self) -> None:
        self._sync_new_persona_state()
        self.new_memory_store.save(self.new_memory_state)
        self.new_persona_store.save(self.new_persona_state)

    def get_message_history(self, include_system_prompt: bool = True) -> list[dict]:
        return self.buffer.to_list(include_system_prompt=include_system_prompt)

    def _recent_assistant_messages(self, limit: int = 3) -> list[str]:
        messages = [message.get("content", "") for message in list(self.buffer.messages) if message.get("role") == "assistant"]
        return [str(item or "") for item in messages[-limit:]]

    def _truncate_for_prompt(self, text, limit: int = 400) -> str:
        value = str(text or "").strip()
        return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"

    def _is_self_intro_request(self, text: str) -> bool:
        value = str(text or "")
        return any(token in value for token in ("自我介绍", "介绍一下你自己", "你是谁", "介绍你自己", "做个介绍"))

    def _requires_story_grounding(self, text: str) -> bool:
        value = str(text or "")
        return any(token in value for token in ("故事", "经历", "过去", "往事", "旅行见闻", "以前发生"))

    def _requires_persona_grounding(self, text: str) -> bool:
        value = str(text or "")
        if self._is_self_intro_request(value) or self._requires_story_grounding(value):
            return True
        return any(token in value for token in ("喜欢", "讨厌", "性格", "口头禅", "价值观", "世界观", "怎么看", "设定", "你自己"))

    def _requires_external_grounding(self, text: str) -> bool:
        value = str(text or "")
        return any(token in value for token in ("天气", "气温", "温度", "下雨", "新闻", "最新", "最近", "比赛", "票房", "搜索"))

    def _has_tool_evidence(self, tool_context: str) -> bool:
        value = str(tool_context or "").strip()
        return bool(value and value != "None")

    def _persona_foundation_available(self) -> bool:
        persona = self.persona_system
        return bool(getattr(persona, "character_voice_card", "").strip() or getattr(persona, "selected_keywords", []) or getattr(persona, "display_keywords", []) or any(payload for payload in (getattr(persona, "base_template", {}) or {}).values()))

    def _build_background_profile_text(self) -> str:
        base_template = getattr(self.persona_system, "base_template", {}) or {}
        background = base_template.get("00_BACKGROUND_PROFILE", {})
        profile = background.get("profile", {}) if isinstance(background, dict) else {}
        experiences = list(background.get("key_experiences", []) or []) if isinstance(background, dict) else []
        pieces = []
        for label, key in (("姓名", "full_name"), ("称号", "title"), ("出身", "origin"), ("外观", "appearance")):
            value = str(profile.get(key) or profile.get("name") if key == "full_name" else profile.get(key) or "").strip()
            if value:
                pieces.append(f"{label}：{self._truncate_for_prompt(value, 80)}")
        if experiences:
            pieces.append(f"关键经历：{self._truncate_for_prompt(experiences[0], 120)}")
        return "\n".join(pieces)

    def _build_self_intro_grounding_block(self) -> str:
        parts = []
        background = self._build_background_profile_text()
        if background:
            parts.append("【背景档案】\n" + background)
        if getattr(self.persona_system, "character_voice_card", "").strip():
            parts.append("【角色声音底稿】\n" + str(self.persona_system.character_voice_card).strip())
        keywords = list(getattr(self.persona_system, "selected_keywords", []) or getattr(self.persona_system, "display_keywords", []) or [])
        if keywords:
            parts.append("【高权重特征】\n" + "、".join(keywords[:8]))
        return "\n\n".join(parts)

    def _build_self_intro_fallback(self) -> str:
        background = self._build_background_profile_text()
        if background:
            lines = [f"我是{self.config.name}。"]
            for line in background.splitlines()[:2]:
                _, _, value = line.partition("：")
                if value.strip():
                    lines.append(value.strip() + "。")
            return "\n\n".join(lines[:3])
        voice = str(getattr(self.persona_system, "character_voice_card", "") or "").strip()
        if voice:
            parts = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", voice) if part.strip()]
            return "\n\n".join(parts[:2]) if parts else f"我是{self.config.name}。"
        return f"我是{self.config.name}。"

    def _grounded_persona_fallback(self) -> str:
        return "这个问题如果随口回答，反而容易把设定说偏。你要是愿意，我可以只说资料里明确提到的部分。"

    def _grounded_story_fallback(self) -> str:
        return "这件事我不想随口编一个故事给你听。要是资料里没有写清楚，我宁愿只说我真正知道的那部分。"

    def _grounded_external_fallback(self) -> str:
        return "这个我现在没法确定。等我查到可靠的信息，再告诉你会更合适一些。"

    def _strip_reasoning_leakage(self, text: str) -> str:
        banned = ("根据上下文", "考虑到", "由于", "我需要", "我决定采用", "用户问的是", "这超出了我个人经历的范围", "需要工具结果来回答", "没有找到工具结果", "我会直接告诉用户")
        lines = []
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if line and not any(line.startswith(prefix) for prefix in banned):
                lines.append(line)
        return "\n".join(lines).strip()

    def _postprocess_assistant_response(self, response: str, user_input: str = "") -> str:
        text = self._strip_reasoning_leakage(response)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if self._is_self_intro_request(user_input) and (not text or any(token in text for token in ("小秘密", "随口胡说", "资料里没有明确写", "不想告诉你"))):
            text = self._build_self_intro_fallback()
        return text.strip()

    def _run_harness_pipeline(self, user_input: str):
        normalized_query = re.sub(r"\s+", " ", str(user_input or "").strip())
        persona_recall = self.persona_rag_engine.recall(normalized_query)
        recent_conversation = conversation_to_string(self.get_message_history(False)[-6:])
        route_decision = self.query_router.route(normalized_query, persona_recall, is_public=True, recent_conversation=recent_conversation, character_name=self.config.name)
        intent_result = self.query_router.last_intent_result
        rewrite = self.query_rewriter.rewrite(normalized_query, route_decision, self.config.name, intent_result)
        if intent_result.tool_name == "web_search":
            params = dict(intent_result.tool_params or {})
            if rewrite.reality_query:
                params["search_query"] = rewrite.reality_query
            intent_result.tool_params = params
        elif intent_result.tool_name == "weather":
            params = dict(intent_result.tool_params or {})
            if not params.get("location") and rewrite.reality_query:
                location = rewrite.reality_query.replace("天气", "").strip()
                if location:
                    params["location"] = location
                    params.setdefault("location_confidence", "low")
                    params.setdefault("location_source", "context")
            intent_result.tool_params = params
        memory_result = self.memory_rag_engine.recall(normalized_query, self.new_memory_state)
        tool_report = self.tool_runtime.execute(intent_result=intent_result, persona_name=self.config.name)
        deduped = self.recall_deduplicator.dedup(persona_recall, memory_result)
        web_persona_context, web_reality_context = split_tool_context_by_mode(tool_report, route_decision)
        assembled = self.context_assembler.assemble(route_type=route_decision.type, deduped=deduped, web_persona_context=web_persona_context, web_reality_context=web_reality_context)
        return route_decision, intent_result, tool_report, assembled

    def send_message(self, user_input, return_json: bool = False, attached_image=None):
        self.num_messages += 1
        self.buffer.add_message("user", user_input)
        self.relation_system.on_user_message(str(user_input or ""))
        self.emotion_system.apply_user_signal(str(user_input or ""))

        memories, recalled_memories = self.memory_system.recall_memories(self.get_message_history(False))
        route_decision, intent_result, tool_report, assembled_context = self._run_harness_pipeline(str(user_input or ""))
        if tool_report.follow_up_message:
            self.buffer.add_message("assistant", tool_report.follow_up_message)
            return tool_report.follow_up_message

        local_precise_context = self.persona_system.build_precise_query_context(str(user_input or ""))
        local_story_context = self.persona_system.build_story_context(str(user_input or "")) if self._requires_story_grounding(str(user_input or "")) else ""
        if self._is_self_intro_request(str(user_input or "")):
            local_precise_context = "\n\n".join(part for part in (self._build_self_intro_grounding_block(), local_precise_context) if part)
        if local_precise_context or local_story_context:
            assembled_context.slots["evidence_chunks"] = "\n\n".join(part for part in (assembled_context.slots.get("evidence_chunks", ""), local_precise_context, local_story_context) if part)

        tool_context_for_turn = assembled_context.slots.get("web_reality_context") or assembled_context.slots.get("web_persona_context") or ""
        persona_context = self.context_assembler.build_prompt_context(assembled_context)
        record_debug_info(self, route_decision, assembled_context, tool_context_for_turn, local_precise_context=local_precise_context, local_story_context=local_story_context)

        if self._requires_story_grounding(str(user_input or "")) and not local_story_context:
            response = self._grounded_story_fallback()
            self.buffer.add_message("assistant", response)
            return response
        if self._requires_persona_grounding(str(user_input or "")) and not (local_precise_context or assembled_context.slots.get("immutable_core", "") or assembled_context.slots.get("evidence_chunks", "")) and not self._has_tool_evidence(tool_context_for_turn):
            response = self._build_self_intro_fallback() if self._is_self_intro_request(str(user_input or "")) else self._grounded_persona_fallback()
            self.buffer.add_message("assistant", response)
            return response
        if self._requires_external_grounding(str(user_input or "")) and not self._has_tool_evidence(tool_context_for_turn):
            response = self._grounded_external_fallback()
            self.buffer.add_message("assistant", response)
            return response

        pending_signal = estimate_pending_signal(str(user_input or ""))
        self.emotion_state_machine.queue_signal(pending_signal)
        thought_data = self.thought_system.think(messages=self.get_message_history(False), memories=memories, recalled_memories=recalled_memories, last_message=self.last_message, persona_context=persona_context)
        self.emotion_state_machine.update_from_thought(thought_signal(thought_data))

        prompt_content = USER_TEMPLATE.format(**build_format_data(self, user_input, thought_data, memories, persona_context, tool_context_for_turn))
        request_history = self.get_message_history(True)
        request_history[-1]["content"] = prompt_content
        response = self.model.generate(request_history, temperature=0.8, max_tokens=2048, return_json=return_json)
        if not return_json:
            response = self._postprocess_assistant_response(response, user_input=str(user_input or ""))

        if not (self._requires_external_grounding(str(user_input or "")) and not self._has_tool_evidence(tool_context_for_turn)):
            self.memory_system.remember(f"用户说：{user_input}\n{self.config.name} 回答：{response}", emotion=thought_data.get("emotion_obj"))
            self.memory_writer.remember(self.new_memory_state, event_summary=self._truncate_for_prompt(f"用户提到：{user_input}；{self.config.name} 回应：{response}", 280), topic_tags=derive_topic_tags(str(user_input or "")), relation_impact=derive_relation_impact(pending_signal), importance=0.55, character_emotion=thought_data.get("emotion", "平静"))
            self._sync_new_persona_state()

        self.last_message = datetime.now()
        self.tick()
        self.health_monitor.record_turn_metrics(coverage_score=assembled_context.metadata.get("coverage_score", 0.0))
        self.buffer.add_message("assistant", response)
        return json.dumps(response, ensure_ascii=False, indent=2) if return_json else response

    def set_thought_visibility(self, shown: bool) -> None:
        self.thought_system.show_thoughts = shown

    def get_mood(self):
        return self.emotion_system.mood

    def get_beliefs(self):
        semantic_notes = [record.content for record in getattr(self.new_memory_state, "semantic_records", []) if getattr(record, "scope", "") == "USER_SPECIFIC" and getattr(record, "content", "")]
        return semantic_notes or self.memory_system.get_beliefs()

    def set_mood(self, pleasure=None, arousal=None, dominance=None) -> None:
        if pleasure is None and arousal is None and dominance is None:
            self.emotion_system.reset_mood()
        else:
            self.emotion_system.set_emotion(pleasure=pleasure, arousal=arousal, dominance=dominance)

    def set_relation(self, friendliness=None, dominance=None) -> None:
        self.relation_system.set_relation(friendliness=friendliness, dominance=dominance)

    def get_memories(self):
        return self.memory_system.get_short_term_memories()

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

    def _json_safe(self, value):
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "model_dump"):
            return self._json_safe(value.model_dump())
        if hasattr(value, "dict"):
            return self._json_safe(value.dict())
        return str(value)

    @staticmethod
    def _config_from_payload(payload):
        return AIConfig.model_validate(payload or {}) if hasattr(AIConfig, "model_validate") else AIConfig.parse_obj(payload or {})

    def _state_snapshot(self):
        self._sync_new_persona_state()
        return self._json_safe({
            "config": self._config_payload(),
            "buffer": {"system_prompt": self.buffer.system_prompt, "messages": list(self.buffer.messages)},
            "runtime": {"num_messages": self.num_messages, "last_message": self.last_message, "thought_visible": bool(getattr(self.thought_system, "show_thoughts", False))},
            "persona_runtime": {
                "base_template": getattr(self.persona_system, "base_template", {}),
                "entries": getattr(self.persona_system, "entries", []),
                "source_records": getattr(self.persona_system, "source_records", {}),
                "pending_previews": getattr(self.persona_system, "pending_previews", {}),
                "display_keywords": getattr(self.persona_system, "display_keywords", []),
                "selected_keywords": getattr(self.persona_system, "selected_keywords", []),
                "style_examples": getattr(self.persona_system, "style_examples", []),
                "natural_reference_triggers": getattr(self.persona_system, "natural_reference_triggers", []),
                "character_voice_card": getattr(self.persona_system, "character_voice_card", ""),
                "story_chunks": getattr(self.persona_system, "story_chunks", []),
            },
        })

    def _restore_snapshot(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        buffer_payload = payload.get("buffer", {}) if isinstance(payload.get("buffer", {}), dict) else {}
        self.buffer.set_system_prompt(str(buffer_payload.get("system_prompt", "") or self.config.system_prompt))
        self.buffer.flush()
        for message in buffer_payload.get("messages", []) or []:
            if isinstance(message, dict) and message.get("role"):
                self.buffer.add_message(str(message["role"]), message.get("content", ""))
        runtime_payload = payload.get("runtime", {}) if isinstance(payload.get("runtime", {}), dict) else {}
        self.num_messages = int(runtime_payload.get("num_messages", len(self.buffer.messages)) or 0)
        self.last_message = runtime_payload.get("last_message")
        if isinstance(self.last_message, str):
            try:
                self.last_message = datetime.fromisoformat(self.last_message.strip())
            except ValueError:
                self.last_message = None
        self.thought_system.show_thoughts = bool(runtime_payload.get("thought_visible", getattr(self.thought_system, "show_thoughts", False)))
        persona_payload = payload.get("persona_runtime", {}) if isinstance(payload.get("persona_runtime", {}), dict) else {}
        self.persona_system.base_template = dict(persona_payload.get("base_template", self.persona_system._empty_base_template()))
        self.persona_system.entries = list(persona_payload.get("entries", []) or [])
        self.persona_system.source_records = dict(persona_payload.get("source_records", {}) or {})
        self.persona_system.pending_previews = dict(persona_payload.get("pending_previews", {}) or {})
        self.persona_system.display_keywords = list(persona_payload.get("display_keywords", []) or [])
        self.persona_system.selected_keywords = list(persona_payload.get("selected_keywords", []) or [])
        self.persona_system.style_examples = list(persona_payload.get("style_examples", []) or [])
        self.persona_system.natural_reference_triggers = list(persona_payload.get("natural_reference_triggers", []) or [])
        self.persona_system.character_voice_card = str(persona_payload.get("character_voice_card", "") or "")
        self.persona_system.story_chunks = list(persona_payload.get("story_chunks", []) or [])
        self.persona_system._repair_state()
        self._sync_new_persona_state()

    def save(self, path):
        self._save_new_architecture_state()
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self._state_snapshot(), file, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            return None
        ai_system = AISystem(config=AISystem._config_from_payload(payload.get("config", {})))
        ai_system._restore_snapshot(payload)
        ai_system._load_new_architecture_state()
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
    print("验证成功！")
    return True

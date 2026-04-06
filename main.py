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
from ai_system_store import AISystemStore
from config import DEFAULT_CONFIG
from const import AI_SYSTEM_PROMPT, NEW_MEMORY_STATE_PATH, NEW_PERSONA_STATE_PATH, USER_TEMPLATE
from context.context_assembler import ContextAssembler
from context.recall_deduplicator import RecallDeduplicator
from diagnostics.conflict_log import ConflictLog
from diagnostics.health_monitor import HealthMonitor
from diagnostics.self_check import SelfCheck
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
        self.memory_writer = MemoryWriter(self.conflict_log)
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

        self.num_messages = 0
        self.last_message = None
        self.last_tick = datetime.now()
        self.last_debug_info = {}

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
        self._bootstrap_persona_from_uploads_if_needed()

    def _bootstrap_persona_from_uploads_if_needed(self) -> None:
        if self._persona_foundation_available():
            return
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        if not os.path.isdir(uploads_dir):
            return
        candidate_paths: list[str] = []
        for root, _, files in os.walk(uploads_dir):
            for name in files:
                lower = name.lower()
                if lower.endswith((".txt", ".md", ".markdown", ".json")) or lower.endswith("_txt"):
                    candidate_paths.append(os.path.join(root, name))
        for path in sorted(candidate_paths)[:6]:
            try:
                self.persona_system.load_file(path)
            except Exception:
                continue
        self._sync_persona_state()

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
        return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"

    def _is_self_intro_request(self, text: str) -> bool:
        return bool(self.persona_policy.infer_grounding_needs(text).get("is_self_intro"))

    def _requires_story_grounding(self, text: str) -> bool:
        return bool(self.persona_policy.infer_grounding_needs(text).get("needs_story_grounding"))

    def _requires_persona_grounding(self, text: str) -> bool:
        return bool(self.persona_policy.infer_grounding_needs(text).get("needs_persona_grounding"))

    def _requires_external_grounding(self, text: str) -> bool:
        return bool(self.persona_policy.infer_grounding_needs(text).get("needs_external_grounding"))

    def _has_tool_evidence(self, tool_context: str) -> bool:
        value = str(tool_context or "").strip()
        return bool(value and value != "None")

    def _persona_foundation_available(self) -> bool:
        persona = self.persona_system
        return bool(getattr(persona, "character_voice_card", "").strip() or getattr(persona, "display_keywords", []) or any(payload for payload in (getattr(persona, "base_template", {}) or {}).values()))

    def _build_self_intro_fallback(self) -> str:
        response = self.persona_policy.build_self_intro_response()
        if response:
            return response
        return self.persona_policy.generate_in_character_refusal(self, "self_intro")

    def _grounded_persona_fallback(self) -> str:
        return self.persona_policy.generate_in_character_refusal(self, "persona")

    def _grounded_story_fallback(self) -> str:
        return self.persona_policy.generate_in_character_refusal(self, "story")

    def _grounded_external_fallback(self) -> str:
        return self.persona_policy.generate_in_character_refusal(self, "external")

    def _run_pipeline(self, user_input: str):
        normalized_query = re.sub(r"\s+", " ", str(user_input or "").strip())
        persona_recall = self.persona_rag_engine.recall(normalized_query)
        recent_conversation = conversation_to_string(self.get_message_history(False)[-6:])
        route_decision = self.query_router.route(normalized_query, persona_recall, is_public=True, recent_conversation=recent_conversation, character_name=self.config.name)
        intent_result = self.query_router.last_intent_result
        if intent_result.tool_name == "weather":
            params = dict(intent_result.tool_params or {})
            if not params.get("location"):
                location = self.query_router.extractor._extract_weather_location(normalized_query, recent_conversation)
                if location:
                    params["location"] = location
                    params.setdefault("location_confidence", "low")
                    params.setdefault("location_source", "context")
            intent_result.tool_params = params
        elif intent_result.tool_name == "web_search":
            params = dict(intent_result.tool_params or {})
            params["search_query"] = str(params.get("search_query") or normalized_query).strip()
            intent_result.tool_params = params
        memory_result = self.memory_rag_engine.recall(normalized_query, self.new_memory_state)
        tool_report = self.tool_runtime.execute(intent_result=intent_result, persona_name=self.config.name)
        deduped = self.recall_deduplicator.dedup(persona_recall, memory_result)
        web_persona_context, web_reality_context = split_tool_context_by_mode(tool_report, route_decision)
        assembled = self.context_assembler.assemble(route_type=route_decision.type, deduped=deduped, web_persona_context=web_persona_context, web_reality_context=web_reality_context)
        return route_decision, intent_result, tool_report, assembled, persona_recall

    def _choose_response(self, user_input: str, thought_data: dict, grounding: dict, tool_context: str, persona_context: str, persona_recall) -> str:
        persona_focus = str(self.persona_policy.persona_query_focus(str(user_input or "")) or "").strip()
        story_hits = list((getattr(persona_recall, "metadata", {}) or {}).get("story_hits", []) or [])
        if grounding["is_self_intro"]:
            if not grounding["has_identity_reference"] and not str(persona_context or "").strip():
                return self._build_self_intro_fallback()
            return self.response_generator.self_intro(str(user_input or ""), thought_data, persona_context) or self._build_self_intro_fallback()
        if grounding["needs_story_grounding"]:
            if not story_hits:
                return self._grounded_story_fallback()
            return self.response_generator.story(str(user_input or ""), thought_data, story_hits[0]) or self._grounded_story_fallback()
        if grounding["needs_external_grounding"]:
            return self.response_generator.external(str(user_input or ""), thought_data, tool_context) or self._grounded_external_fallback()
        if persona_focus in {"catchphrase", "likes", "dislikes", "personality"} and str(persona_context or "").strip():
            return self.response_generator.persona_focus(str(user_input or ""), thought_data, persona_context, persona_focus) or self._grounded_persona_fallback()

        prompt_content = USER_TEMPLATE.format(**build_format_data(self, user_input, thought_data, self.memory_system.get_short_term_memories(), persona_context, tool_context, grounding=grounding))
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
        response = self.model.generate(request_history, temperature=0.3 if grounding["needs_story_grounding"] else 0.8, max_tokens=800)
        return self.response_generator.postprocess(response, user_input=str(user_input or ""))

    def send_message(self, user_input, return_json: bool = False, attached_image=None):
        self.num_messages += 1
        self.buffer.add_message("user", user_input)

        route_decision, intent_result, tool_report, assembled_context, persona_recall = self._run_pipeline(str(user_input or ""))
        if tool_report.follow_up_message:
            self.buffer.add_message("assistant", tool_report.follow_up_message)
            return tool_report.follow_up_message

        tool_context_for_turn = assembled_context.slots.get("web_reality_context") or assembled_context.slots.get("web_persona_context") or ""
        persona_context = assembled_context.slots.get("evidence_chunks", "")
        thought_persona_context = self.context_assembler.build_prompt_context(assembled_context)
        grounding = self.persona_policy.infer_grounding_needs(str(user_input or ""), route_decision=route_decision, persona_recall=persona_recall, tool_report=tool_report)

        if grounding["needs_story_grounding"] and not persona_context:
            response = self._grounded_story_fallback()
            self.buffer.add_message("assistant", response)
            return response
        if grounding["needs_external_grounding"] and not self._has_tool_evidence(tool_context_for_turn):
            response = self._grounded_external_fallback()
            self.buffer.add_message("assistant", response)
            return response
        if grounding["needs_persona_grounding"] and not persona_context and not (grounding["is_self_intro"] and grounding["has_identity_reference"]):
            response = self._build_self_intro_fallback() if grounding["is_self_intro"] else self._grounded_persona_fallback()
            self.buffer.add_message("assistant", response)
            return response

        pending_signal = estimate_pending_signal(str(user_input or ""))
        self.emotion_state_machine.queue_signal(pending_signal)
        memories, recalled_memories = self.memory_system.recall_memories(self.get_message_history(False))
        thought_data = self.thought_system.think(messages=self.get_message_history(False), memories=memories, recalled_memories=recalled_memories, last_message=self.last_message, persona_context=thought_persona_context)
        self.emotion_state_machine.update_from_thought(thought_signal(thought_data))
        record_debug_info(self, route_decision, assembled_context, tool_context_for_turn, thought_data=thought_data, local_precise_context="", local_story_context="")

        response = self._choose_response(user_input, thought_data, grounding, tool_context_for_turn, persona_context, persona_recall)
        if not response:
            response = self._grounded_persona_fallback()

        self.memory_writer.remember(
            self.new_memory_state,
            event_summary=self._truncate_for_prompt(f"用户提到：{user_input}；{self.config.name} 回应：{response}", 280),
            topic_tags=derive_topic_tags(str(user_input or "")),
            relation_impact=derive_relation_impact(pending_signal),
            importance=0.55,
            character_emotion=str(thought_data.get("emotion", "平静")),
        )
        self.memory_system.set_state(self.new_memory_state)
        self._sync_persona_state()

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
    print("验证成功！")
    return True

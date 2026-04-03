"""杩愯 AI 鐨勪富妯″潡銆?"""

import json
import os
import pickle
import re
from collections import deque
from datetime import datetime

import requests
from pydantic import BaseModel, Field

from config import DEFAULT_CONFIG
from const import AI_SYSTEM_PROMPT, NEW_MEMORY_STATE_PATH, NEW_PERSONA_STATE_PATH, SAVE_PATH, USER_TEMPLATE
from context.context_assembler import ContextAssembler
from context.recall_deduplicator import RecallDeduplicator
from diagnostics.conflict_log import ConflictLog
from diagnostics.health_monitor import HealthMonitor
from diagnostics.self_check import SelfCheck
from knowledge.knowledge_source import KnowledgeSource, RouteDecision, RouteType, SearchMode
from knowledge.persona_rag_engine import PersonaRAGEngine
from knowledge.persona_system import (
	AttitudeTowardUser,
	CoreTrait,
	IdentityProfile,
	InnateBelief,
	ParentChunk,
	PersonaState,
	PersonaSystemStore,
	SpeechDNA,
)
from llm import FallbackMistralLLM, MistralLLM, get_active_llm_label, get_llm_settings, has_llm_api_key
from memory.memory_system import MemorySystem
from memory.memory_rag_engine import MemoryRAGEngine
from memory.memory_system import MemorySystemState, MemorySystemStore
from memory.memory_writer import MemoryWriter
from knowledge.persona_system import PersonaSystem
from reasoning.emotion_state_machine import (
	EmotionStateMachine,
	EmotionSystem,
	PersonalitySystem,
	RelationshipSystem,
)
from reasoning.emotion_state_machine import EmotionSignal
from reasoning.thought_system import ThoughtSystemAdapter
from reasoning.thought_system import ThoughtSystem
from routing.query_rewriter import QueryRewriter
from routing.query_router import QueryRouter
from tools import DEFAULT_TOOL_REGISTRY, ToolRuntime
from utils import (
	format_date,
	format_memories_to_string,
	format_time,
	time_since_last_message_string,
)


class MessageBuffer:
	"""淇濆瓨鏈€杩戞秷鎭殑缂撳啿鍖猴紝瓒呭嚭涓婇檺鏃朵細鑷姩娣樻卑鏃ф秷鎭€?"""

	def __init__(self, max_messages):
		self.max_messages = max_messages
		self.messages = deque(maxlen=max_messages)
		self.system_prompt = ""

	def set_system_prompt(self, prompt):
		"""璁剧疆绯荤粺鎻愮ず璇嶃€?"""
		self.system_prompt = prompt.strip()

	def add_message(self, role, content):
		"""鍚戠紦鍐插尯娣诲姞涓€鏉℃秷鎭€?"""
		self.messages.append({"role": role, "content": content})

	def pop(self):
		"""绉婚櫎骞惰繑鍥炴渶鍚庝竴鏉℃秷鎭€?"""
		return self.messages.pop()

	def flush(self):
		"""娓呯┖缂撳啿鍖恒€?"""
		self.messages.clear()

	def to_list(self, include_system_prompt=True):
		"""灏嗙紦鍐插尯杞崲涓哄垪琛紝鍙€夋槸鍚﹂檮甯︾郴缁熸彁绀鸿瘝銆?"""
		history = []
		if include_system_prompt and self.system_prompt:
			history.append({"role": "system", "content": self.system_prompt})
		history.extend(message.copy() for message in self.messages)
		return history


class PersonalityConfig(BaseModel):
	"""AI 浜烘牸閰嶇疆銆?"""

	open: float = Field(ge=-1.0, le=1.0)
	conscientious: float = Field(ge=-1.0, le=1.0)
	agreeable: float = Field(ge=-1.0, le=1.0)
	extrovert: float = Field(ge=-1.0, le=1.0)
	neurotic: float = Field(ge=-1.0, le=1.0)


class AIConfig(BaseModel):
	"""AI 鎬婚厤缃€?"""

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
	"""鍖呭惈鍚勪釜瀛愮郴缁熺殑 AI 涓荤郴缁熴€?"""

	PERSONA_EVIDENCE_KEYWORDS = (
		"故事", "经历", "设定", "背景", "过去", "原作", "剧情",
		"性格", "口头禅", "口癖", "说话方式", "语气", "称呼",
		"喜欢", "讨厌", "厌恶", "价值观", "世界观", "身份",
		"外貌", "外观", "发色", "关系", "喜好", "习惯", "经典台词",
		"今天的经历", "最近的经历", "旅行经历", "讲讲你", "自我介绍",
	)
	EXTERNAL_FACT_KEYWORDS = (
		"谁是", "是谁", "如何评价", "怎么看", "介绍一下", "什么是", "新闻",
		"资料", "信息", "比赛", "战队", "选手", "职业选手", "战绩", "排名",
		"冠军", "作品", "游戏", "动漫", "电影", "小说", "公司", "品牌",
	)

	def __init__(self, config=None):
		config = config or AIConfig()
		personality = config.personality

		self.config = config
		self.personality_system = PersonalitySystem(
			openness=personality.open,
			conscientious=personality.conscientious,
			extrovert=personality.extrovert,
			agreeable=personality.agreeable,
			neurotic=personality.neurotic,
		)
		self.memory_system = MemorySystem(config)
		self.memory_system.legacy_belief_enabled = False
		self.relation_system = RelationshipSystem()
		self.emotion_system = EmotionSystem(
			self.personality_system,
			self.relation_system,
			self.config,
		)
		self.thought_system = ThoughtSystem(
			config,
			self.emotion_system,
			self.memory_system,
			self.relation_system,
			self.personality_system,
		)
		self.persona_system = PersonaSystem(persona_name=config.name)
		self.tool_runtime = ToolRuntime(DEFAULT_TOOL_REGISTRY)
		self.model = FallbackMistralLLM()
		self.runtime_config = DEFAULT_CONFIG
		self.conflict_log = ConflictLog()
		self.health_monitor = HealthMonitor()
		self.self_check = SelfCheck()
		self.new_memory_store = MemorySystemStore(NEW_MEMORY_STATE_PATH)
		self.new_persona_store = PersonaSystemStore(NEW_PERSONA_STATE_PATH)
		self.new_memory_state = self.new_memory_store.load()
		self.new_persona_state = self.new_persona_store.load()
		self.memory_rag_engine = MemoryRAGEngine()
		self.memory_writer = MemoryWriter(self.conflict_log)
		self.persona_rag_engine = PersonaRAGEngine(self.persona_system, self.new_persona_state)
		self.query_router = QueryRouter()
		self.query_rewriter = QueryRewriter()
		self.context_assembler = ContextAssembler()
		self.recall_deduplicator = RecallDeduplicator()
		self.emotion_state_machine = EmotionStateMachine()
		self.thought_adapter = ThoughtSystemAdapter(self.thought_system)
		self._sync_new_persona_state()
		self.self_check.run()

		self.num_messages = 0
		self.last_message = None
		self.last_recall_tick = datetime.now()
		self.last_tick = datetime.now()

		self.buffer = MessageBuffer(20)
		self.buffer.set_system_prompt(config.system_prompt)

	def set_config(self, config):
		"""鏇存柊閰嶇疆銆?"""
		self.config = config
		self.memory_system.config = config
		if getattr(self.memory_system, "belief_system", None) is not None:
			self.memory_system.belief_system.config = config
		self.memory_system.legacy_belief_enabled = False
		self.thought_system.config = config
		self.emotion_system.config = config
		self.persona_system.persona_name = config.name
		self._sync_new_persona_state()
		personality = config.personality
		self.personality_system = PersonalitySystem(
			openness=personality.open,
			conscientious=personality.conscientious,
			extrovert=personality.extrovert,
			agreeable=personality.agreeable,
			neurotic=personality.neurotic,
		)
		self.emotion_system.personality_system = self.personality_system
		self.thought_system.personality_system = self.personality_system

	def _repair_runtime_state(self):
		"""涓烘棫瀛樻。琛ラ綈杩愯鏃跺瓧娈点€?"""
		if not hasattr(self, "tool_runtime") or self.tool_runtime is None:
			self.tool_runtime = ToolRuntime(DEFAULT_TOOL_REGISTRY)
		if not hasattr(self, "persona_system") or self.persona_system is None:
			self.persona_system = PersonaSystem(persona_name=self.config.name)
		elif not getattr(self.persona_system, "persona_name", None):
			self.persona_system.persona_name = self.config.name
		if not hasattr(self, "runtime_config"):
			self.runtime_config = DEFAULT_CONFIG
		if not hasattr(self, "conflict_log") or self.conflict_log is None:
			self.conflict_log = ConflictLog()
		if not hasattr(self, "health_monitor") or self.health_monitor is None:
			self.health_monitor = HealthMonitor()
		if not hasattr(self, "self_check") or self.self_check is None:
			self.self_check = SelfCheck()
		if not hasattr(self, "new_memory_store") or self.new_memory_store is None:
			self.new_memory_store = MemorySystemStore(NEW_MEMORY_STATE_PATH)
		if not hasattr(self, "new_persona_store") or self.new_persona_store is None:
			self.new_persona_store = PersonaSystemStore(NEW_PERSONA_STATE_PATH)
		if not hasattr(self, "new_memory_state") or self.new_memory_state is None:
			try:
				self.new_memory_state = self.new_memory_store.load()
			except Exception:
				self.new_memory_state = MemorySystemState()
		if not hasattr(self, "new_persona_state") or self.new_persona_state is None:
			try:
				self.new_persona_state = self.new_persona_store.load()
			except Exception:
				self.new_persona_state = PersonaState()
		if not hasattr(self, "memory_rag_engine") or self.memory_rag_engine is None:
			self.memory_rag_engine = MemoryRAGEngine()
		if not hasattr(self, "memory_writer") or self.memory_writer is None:
			self.memory_writer = MemoryWriter(self.conflict_log)
		if not hasattr(self, "persona_rag_engine") or self.persona_rag_engine is None:
			self.persona_rag_engine = PersonaRAGEngine(self.persona_system, self.new_persona_state)
		if not hasattr(self, "query_router") or self.query_router is None:
			self.query_router = QueryRouter()
		if not hasattr(self, "query_rewriter") or self.query_rewriter is None:
			self.query_rewriter = QueryRewriter()
		if not hasattr(self, "context_assembler") or self.context_assembler is None:
			self.context_assembler = ContextAssembler()
		if not hasattr(self, "recall_deduplicator") or self.recall_deduplicator is None:
			self.recall_deduplicator = RecallDeduplicator()
		if not hasattr(self, "emotion_state_machine") or self.emotion_state_machine is None:
			self.emotion_state_machine = EmotionStateMachine()
		if not hasattr(self, "thought_adapter") or self.thought_adapter is None:
			self.thought_adapter = ThoughtSystemAdapter(self.thought_system)
		if hasattr(self, "memory_system"):
			self.memory_system.legacy_belief_enabled = False
		self._sync_new_persona_state()

	def _load_new_architecture_state(self):
		"""从新的 JSON 存储恢复新架构状态。"""
		try:
			self.new_memory_state = self.new_memory_store.load()
		except Exception:
			self.new_memory_state = MemorySystemState()
		try:
			self.new_persona_state = self.new_persona_store.load()
		except Exception:
			self.new_persona_state = PersonaState()
		if hasattr(self, "persona_rag_engine") and self.persona_rag_engine is not None:
			self.persona_rag_engine.set_persona_state(self.new_persona_state)
		self._sync_new_persona_state()

	def _save_new_architecture_state(self):
		"""显式保存新架构状态，避免仅依赖 pickle。"""
		self._sync_new_persona_state()
		self.new_memory_store.save(self.new_memory_state)
		self.new_persona_store.save(self.new_persona_state)

	def _sync_new_persona_state(self):
		"""将旧 persona_system 的实时内容同步为新的结构化状态。"""
		persona_name = getattr(self.persona_system, "persona_name", None) or self.config.name
		profile = getattr(self.persona_system, "profile", {}) or {}
		display_keywords = list(getattr(self.persona_system, "get_display_keywords", lambda limit=8: [])(8) or [])
		selected_keywords = list(getattr(self.persona_system, "selected_keywords", []) or [])
		core_summaries = list(getattr(self.persona_system, "core_summaries", []) or [])
		entries = list(getattr(self.persona_system, "entries", []) or [])

		identity = IdentityProfile(
			name=persona_name,
			archetype=(selected_keywords[0] if selected_keywords else (display_keywords[0] if display_keywords else "")),
		)
		core_traits = [
			CoreTrait(
				feature=str(keyword),
				strength=0.7,
				activation_trigger=[str(keyword)],
				mandatory_behavior="自然流露，不刻意表演。",
				evidence_tags=[str(keyword)],
			)
			for keyword in (selected_keywords or display_keywords)[:8]
			if str(keyword).strip()
		]
		speech_dna = SpeechDNA(
			catchphrases=list(profile.get("catchphrases", []) or []),
			sentence_endings=list(profile.get("sentence_endings", []) or []),
		)
		innate_beliefs = []
		for item in list(profile.get("values", []) or [])[:6] + list(profile.get("worldview", []) or [])[:6]:
			text = str(item or "").strip()
			if text:
				innate_beliefs.append(InnateBelief(content=text, domain="persona", strength=0.7))

		parent_chunks = []
		for index, entry in enumerate(entries):
			text = str(entry.get("text", "")).strip()
			if not text:
				continue
			source_label = str(entry.get("source_label", "") or "")
			source_level = (
				KnowledgeSource.USER_CANON
				if any(token in source_label.lower() for token in ("user", "local", "manual", "file", "txt", "md", "json", "csv"))
				else KnowledgeSource.WEB_PERSONA
			)
			parent_chunks.append(
				ParentChunk(
					chunk_id=str(entry.get("id", f"legacy-{index}")),
					content=text,
					source_level=source_level,
					topic_tags=list(entry.get("keywords", []) or []),
					trait_tags=list((selected_keywords or display_keywords)[:4]),
					importance_score=max(0.0, min(1.0, float(entry.get("priority", 0.5) or 0.5))),
					language="zh",
					version=1,
					deprecated=False,
				)
			)

		relation_state = getattr(self.new_memory_state, "relation_state", None)
		self.new_persona_state.immutable_core.identity = identity
		self.new_persona_state.immutable_core.core_traits = core_traits
		self.new_persona_state.immutable_core.speech_dna = speech_dna
		self.new_persona_state.immutable_core.innate_beliefs = innate_beliefs
		self.new_persona_state.slow_change_layer.attitude_toward_user = AttitudeTowardUser(
			trust=float(getattr(relation_state, "trust", 0.0) or 0.0),
			affection=float(getattr(relation_state, "affection", 0.0) or 0.0),
			respect=float(getattr(relation_state, "trust", 0.0) or 0.0),
		)
		self.new_persona_state.evidence_vault.parent_chunks = parent_chunks
		self.new_persona_state.metadata = {
			"display_keywords": (selected_keywords or display_keywords)[:8],
			"selected_keywords": selected_keywords[:8],
			"legacy_chunk_count": len(entries),
			"core_summaries": [
				str(item.get("text", "")).strip()
				for item in core_summaries
				if str(item.get("text", "")).strip()
			],
		}
		if hasattr(self, "persona_rag_engine") and self.persona_rag_engine is not None:
			self.persona_rag_engine.set_persona_state(self.new_persona_state)

	def get_message_history(self, include_system_prompt=True):
		"""鑾峰彇褰撳墠瀵硅瘽鍘嗗彶銆?"""
		return self.buffer.to_list(include_system_prompt)

	def on_startup(self):
		"""鍦ㄧ郴缁熷姞杞藉悗鎵ц鍒濆鍖栥€?"""
		self.last_tick = datetime.now()
		self.tick()

	def _image_to_description(self, image_url):
		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image_url", "image_url": image_url},
					{
						"type": "text",
						"text": (
							"Please describe in detail what you see in this image. "
							"Make sure to include specific details, such as style, colors, etc."
						),
					},
				],
			}
		]
		model = MistralLLM("mistral-medium-latest")
		return model.generate(messages, temperature=0.1, max_tokens=1024)

	def _input_to_memory(self, user_input, ai_response, attached_image=None):
		user_msg = ""
		if attached_image:
			description = self._image_to_description(attached_image)
			user_msg += f'<attached_img url="{attached_image}">Description: {description}</attached_img>\n'
		user_msg += user_input
		return f"User: {user_msg}\n\n{self.config.name}: {self._memory_safe_assistant_response(ai_response, user_input)}"

	def _select_prompt_memories(self, user_input, memories):
		"""涓?prompt 閫夋嫨鏈€鐩稿叧鐨勭煭鏈熻蹇嗐€?"""
		relevant_memories = self.memory_system.short_term.retrieve_bm25(user_input, top_k=5)
		if relevant_memories:
			return relevant_memories
		if not memories:
			return []
		return memories[-5:]

	def _truncate_for_prompt(self, value, max_chars):
		"""鎸夊瓧绗﹂绠楄鍓敞鍏?prompt 鐨勬枃鏈€?"""
		if len(value) <= max_chars:
			return value
		return value[: max_chars - 3].rstrip() + "..."

	def _recent_assistant_messages(self, limit=3):
		"""鏀堕泦鏈€杩戝嚑杞?assistant 鍥炲锛岀敤浜庨伩鍏嶉噸澶嶆彁鍙婂悓涓€浜鸿缁嗚妭銆?"""
		recent_messages = []
		for message in reversed(self.buffer.messages):
			if message.get("role") != "assistant":
				continue
			content = message.get("content", "")
			if isinstance(content, list):
				text_parts = []
				for item in content:
					if isinstance(item, dict) and item.get("type") == "text":
						text_parts.append(str(item.get("text", "")).strip())
				content = "\n".join(part for part in text_parts if part)
			text = str(content or "").strip()
			if not text:
				continue
			recent_messages.append(text)
			if len(recent_messages) >= limit:
				break
		recent_messages.reverse()
		return recent_messages

	def _get_format_data(self, content, thought_data, memories, persona_context, tool_context):
		now = datetime.now()
		user_emotions = thought_data["possible_user_emotions"]
		user_emotion_list_str = ", ".join(user_emotions)
		if user_emotions:
			user_emotion_str = "The user appears to be feeling the following emotions: " + user_emotion_list_str
		else:
			user_emotion_str = "The user doesn't appear to show any strong emotion."

		thought_str = "\n".join("- " + thought["content"] for thought in thought_data["thoughts"])
		beliefs = self.get_beliefs()
		if beliefs:
			belief_str = "\n".join(f"- {belief}" for belief in beliefs)
		else:
			belief_str = "None"

		memories_str = format_memories_to_string(memories, "You don't have any memories of this user yet!")
		memories_str = self._truncate_for_prompt(memories_str, 500)

		return {
			"name": self.config.name,
			"personality_summary": self.personality_system.get_summary(),
			"persona_context": persona_context or "",
			"tool_context": tool_context,
			"recent_assistant_context": self._recent_assistant_context(),
			"persona_grounding_required": "yes" if self._requires_persona_grounding(content) else "no",
			"external_grounding_required": "yes" if self._requires_external_grounding(content) else "no",
			"tool_evidence_available": "yes" if self._has_tool_evidence(tool_context) else "no",
			"user_input": content,
			"ai_thoughts": thought_str,
			"emotion": thought_data["emotion"],
			"emotion_reason": thought_data["emotion_reason"],
			"memories": memories_str,
			"curr_date": format_date(now),
			"curr_time": format_time(now),
			"user_emotion_str": user_emotion_str,
			"beliefs": belief_str,
			"mood_long_desc": self.emotion_system.get_mood_long_description(),
			"mood_prompt": self.emotion_system.get_mood_prompt(),
			"last_interaction": time_since_last_message_string(self.last_message),
		}

	def _recent_assistant_context(self):
		recent_messages = self._recent_assistant_messages(limit=3)
		if not recent_messages:
			return "None"
		summaries = []
		for text in recent_messages:
			lines = [line.strip() for line in str(text).splitlines() if line.strip()]
			compact = " / ".join(lines[:3]) if lines else str(text).strip()
			summaries.append("- " + self._truncate_for_prompt(compact, 140))
		return self._truncate_for_prompt("\n".join(summaries), 420)

	def _estimate_pending_signal(self, user_input):
		text = str(user_input or "")
		positive = ("喜欢", "谢谢", "开心", "高兴", "温暖", "好喜欢", "信任", "陪我", "爱你")
		negative = ("讨厌", "烦", "滚", "讽刺", "失望", "生气", "恨", "无聊")
		sad = ("难过", "伤心", "低落", "沮丧", "孤独", "疲惫", "痛苦")
		if any(token in text for token in negative):
			return EmotionSignal(mood="受伤", intensity=0.35, valence=-0.45)
		if any(token in text for token in sad):
			return EmotionSignal(mood="关切", intensity=0.25, valence=-0.15)
		if any(token in text for token in positive):
			return EmotionSignal(mood="愉快", intensity=0.2, valence=0.25)
		return EmotionSignal(mood="平静", intensity=0.05, valence=0.0)

	def _thought_signal(self, thought_data):
		emotion_obj = thought_data.get("emotion_obj")
		raw_mood = str(thought_data.get("emotion", "平静") or "平静").strip()
		if (
			not raw_mood
			or len(raw_mood) > 12
			or "[" in raw_mood
			or "]" in raw_mood
			or "Intensity" in raw_mood
			or "event makes" in raw_mood.lower()
		):
			raw_mood = "平静"
		if emotion_obj is None:
			return EmotionSignal(mood=raw_mood, intensity=0.1, valence=0.0)
		intensity = max(0.0, min(1.0, emotion_obj.get_intensity()))
		valence = max(-1.0, min(1.0, emotion_obj.pleasure))
		if raw_mood == "平静":
			if valence >= 0.25:
				raw_mood = "愉快"
			elif valence <= -0.35:
				raw_mood = "难过"
			elif valence <= -0.15:
				raw_mood = "关切"
		return EmotionSignal(
			mood=raw_mood,
			intensity=intensity,
			valence=valence,
		)

	def _derive_relation_impact(self, signal):
		valence = signal.valence
		if valence > 0.15:
			return {"trust_delta": 0.01, "affection_delta": 0.01, "familiarity_delta": 0.015}
		if valence < -0.15:
			return {"trust_delta": -0.01, "affection_delta": -0.012, "familiarity_delta": 0.0}
		return {"familiarity_delta": 0.01}

	def _derive_topic_tags(self, user_input):
		text = str(user_input or "")
		tokens = [token for token in re.split(r"[\s，。！？、,.!?：:；;（）()\[\]\"“”]+", text) if 1 < len(token) <= 12]
		return tokens[:5]

	def _split_tool_context_by_mode(self, tool_report, route_decision):
		context = str(getattr(tool_report, "context", "") or "")
		if context in {"", "None"}:
			return "", ""
		if route_decision.web_search_mode == SearchMode.PERSONA_SEARCH:
			return context, ""
		if route_decision.web_search_mode == SearchMode.REALITY_SEARCH:
			return "", context
		if route_decision.web_search_mode == SearchMode.BOTH:
			return context, context
		return "", ""

	def _run_harness_pipeline(self, user_input):
		persona_result = self.persona_rag_engine.recall(user_input)
		is_public = True
		route_decision = self.query_router.route(user_input, persona_result, is_public=is_public)
		if self._requires_persona_grounding(user_input) and route_decision.web_search_mode == SearchMode.NONE and float(persona_result.coverage_score or 0.0) < 0.82:
			route_decision = RouteDecision(
				type=RouteType.E2 if is_public else RouteType.E2B,
				web_search_mode=SearchMode.PERSONA_SEARCH if is_public else SearchMode.NONE,
				search_hint=persona_result.activated_features,
				fallback=("conservative" if not is_public else None),
				info_domain="CHARACTER_INTERNAL",
			)
		elif self._requires_external_grounding(user_input) and route_decision.web_search_mode == SearchMode.NONE:
			route_decision = RouteDecision(
				type=RouteType.E4,
				web_search_mode=SearchMode.REALITY_SEARCH,
				info_domain="REALITY_FACTUAL",
			)
		rewrite_result = self.query_rewriter.rewrite(user_input, route_decision, self.config.name)
		memory_result = self.memory_rag_engine.recall(user_input, self.new_memory_state)
		tool_report = self.tool_runtime.execute(
			user_input,
			persona_name=self.config.name,
			recent_context=self._recent_tool_context_window(),
			search_mode=route_decision.web_search_mode,
			persona_query=rewrite_result.persona_query,
			reality_query=rewrite_result.reality_query,
		)
		deduped = self.recall_deduplicator.dedup(persona_result, memory_result)
		web_persona_context, web_reality_context = self._split_tool_context_by_mode(tool_report, route_decision)
		assembled = self.context_assembler.assemble(
			route_decision.type,
			deduped,
			web_persona_context=web_persona_context,
			web_reality_context=web_reality_context,
		)
		return route_decision, tool_report, assembled

	def _recent_tool_context_window(self, limit=4):
		window = []
		for message in reversed(self.buffer.messages):
			role = message.get("role")
			if role not in {"user", "assistant"}:
				continue
			content = message.get("content", "")
			if isinstance(content, list):
				text_parts = []
				for item in content:
					if isinstance(item, dict) and item.get("type") == "text":
						text_parts.append(str(item.get("text", "")).strip())
				content = "\n".join(part for part in text_parts if part)
			text = str(content or "").strip()
			if not text:
				continue
			window.append(f"{role}: {self._truncate_for_prompt(text, 100)}")
			if len(window) >= limit:
				break
		window.reverse()
		return "\n".join(window)

	def _requires_persona_grounding(self, user_input):
		text = user_input if isinstance(user_input, str) else str(user_input or "")
		if any(keyword in text for keyword in self.PERSONA_EVIDENCE_KEYWORDS):
			return True
		return bool(re.search(r"(介绍一下你自己|介绍你自己|你是谁|讲讲你的.*?(故事|经历)|可以给我讲.*?(故事|经历))", text))

	def _requires_external_grounding(self, user_input):
		text = user_input if isinstance(user_input, str) else str(user_input or "")
		if self._requires_persona_grounding(text):
			return False
		if any(keyword in text for keyword in self.EXTERNAL_FACT_KEYWORDS):
			return True
		if any(word in text.lower() for word in ("who is", "what is", "latest", "news", "player", "team", "match")):
			return True
		return bool(re.search(r"[A-Za-z]{2,}[A-Za-z0-9_-]*", text)) and (text.endswith(("?", "？")) or "评价" in text or "介绍" in text)

	def _has_tool_evidence(self, tool_context):
		text = str(tool_context or "")
		return text not in {"", "None"} and any(
			marker in text
			for marker in (
				"Reference Tool Result",
				"Realtime Tool Result",
				"Forced Persona Reference Search",
				"Forced External Reference Search",
			)
		)

	def _build_tool_context(self, user_input):
		text = user_input if isinstance(user_input, str) else str(user_input or "")
		contexts = []

		tool_report = self.tool_runtime.execute(
			text,
			persona_name=self.config.name,
			recent_context=self._recent_tool_context_window(),
		)
		if tool_report.context and tool_report.context != "None":
			contexts.append(
				"\n".join(
					[
						"Tool Harness Context:",
						tool_report.context,
						"If a tool result is present for this turn, treat it as grounded support and use it directly when relevant.",
					]
				)
			)

		if self._requires_persona_grounding(text):
			local_story_context = self.persona_system.build_story_context(text)
			if local_story_context:
				contexts.append("Local Persona Evidence:\n" + local_story_context)
			try:
				search_result = DEFAULT_TOOL_REGISTRY.run(
					"web_search",
					persona_name=self.config.name,
					query=text,
					max_results=3,
					timeout=8,
				)
			except Exception as exc:
				search_result = {"snippets": [{"source": "error", "title": "web_search", "text": str(exc)}]}
			snippets = search_result.get("snippets", [])
			if snippets:
				lines = [
					f"[{snippet.get('source', 'web')} | {snippet.get('title', '')}] {snippet.get('text', '')}"
					for snippet in snippets[:3]
				]
				contexts.append(
					"\n".join(
						[
							"Forced Persona Reference Search:",
							*lines,
							"For strong persona questions, answer from user-provided persona material first, then these references. If support is insufficient, answer in-character but stay evasive instead of inventing details.",
						]
					)
				)
			else:
				contexts.append(
					"Forced Persona Reference Search:\n"
					"No strong support was found in persona retrieval or reference search. If the user keeps asking for unsupported persona facts, answer briefly and evasively in character instead of inventing details."
				)
		elif self._requires_external_grounding(text):
			try:
				search_result = DEFAULT_TOOL_REGISTRY.run(
					"web_search",
					persona_name="",
					query=text,
					max_results=4,
					timeout=8,
				)
			except Exception as exc:
				search_result = {"snippets": [{"source": "error", "title": "web_search", "text": str(exc)}]}
			snippets = search_result.get("snippets", [])
			if snippets:
				lines = [
					f"[{snippet.get('source', 'web')} | {snippet.get('title', '')}] {snippet.get('text', '')}"
					for snippet in snippets[:4]
				]
				contexts.append(
					"\n".join(
						[
							"Forced External Reference Search:",
							*lines,
							"For external factual questions, answer from these references first. Do not improvise unsupported facts.",
						]
					)
				)
			else:
				contexts.append(
					"Forced External Reference Search:\n"
					"No supporting references were found for this turn. Do not invent facts; answer conservatively and acknowledge uncertainty."
				)

		return "\n\n".join(contexts) if contexts else "None"

	def _iter_persona_term_candidates(self):
		profile = getattr(self.persona_system, "profile", {}) or {}
		for section in ("likes", "dislikes", "appearance", "catchphrases", "core_setup", "life_experiences"):
			for item in profile.get(section, []) or []:
				text = str(item or "").strip()
				for part in re.split(r"[锛屻€侊紱銆?.|/()\[\]锛堬級\s]+", text):
					part = part.strip()
					if 2 <= len(part) <= 10:
						yield part

	def _overused_persona_terms(self, user_input):
		user_text = str(user_input or "")
		recent_messages = self._recent_assistant_messages(limit=4)
		if not recent_messages:
			return set()
		recent_text = "\n".join(str(message or "") for message in recent_messages)
		overused = set()
		for term in set(self._iter_persona_term_candidates()):
			if term in user_text:
				continue
			if recent_text.count(term) >= 1:
				overused.add(term)
		return overused

	def _suppress_overused_persona_details(self, text, user_input):
		overused_terms = self._overused_persona_terms(user_input)
		if not overused_terms:
			return text

		lines = [line.strip() for line in str(text or "").split("\n") if line.strip()]
		if not lines:
			return text

		filtered_lines = [line for line in lines if not any(term in line for term in overused_terms)]
		if filtered_lines:
			return "\n".join(filtered_lines).strip()

		sentences = [part.strip() for part in re.split(r"(?<=[銆傦紒锛??])\s*", str(text or "")) if part.strip()]
		filtered_sentences = [part for part in sentences if not any(term in part for term in overused_terms)]
		if filtered_sentences:
			return "".join(filtered_sentences).strip()
		return text

	def _suppress_repetitive_sentences(self, text):
		recent_messages = "\n".join(self._recent_assistant_messages(limit=4))
		if not recent_messages:
			return text

		sentences = [part.strip() for part in re.split(r"(?<=[銆傦紒锛??])\s*", str(text or "")) if part.strip()]
		if not sentences:
			return text

		filtered = []
		recent_normalized = re.sub(r"\s+", "", recent_messages)
		for sentence in sentences:
			normalized = re.sub(r"\s+", "", sentence)
			if len(normalized) >= 12 and normalized in recent_normalized:
				continue
			filtered.append(sentence)
		return "".join(filtered).strip() or text

	def _memory_safe_assistant_response(self, response, user_input=""):
		text = self._postprocess_assistant_response(response, user_input=user_input)
		if not isinstance(text, str):
			return str(text)

		recent_messages = "\n".join(self._recent_assistant_messages(limit=4))
		sentences = [part.strip() for part in re.split(r"(?<=[銆傦紒锛??])\s*", text) if part.strip()]
		if not sentences:
			return text

		filtered = []
		seen = set()
		for sentence in sentences:
			normalized = re.sub(r"\s+", "", sentence)
			if not normalized or normalized in seen:
				continue
			seen.add(normalized)
			if normalized and normalized in re.sub(r"\s+", "", recent_messages):
				continue
			filtered.append(sentence)

		collapsed = "".join(filtered).strip()
		return collapsed or text

	def _postprocess_assistant_response(self, response, user_input=""):
		if not isinstance(response, str):
			return response
		text = response.replace("\r\n", "\n").strip()
		text = self._enforce_direct_answer(text, user_input)
		lines = [line.strip() for line in text.split("\n") if line.strip()]
		if not lines:
			return text

		def is_action_line(value):
			return value.startswith("（") and value.endswith("）")

		while lines and is_action_line(lines[0]):
			first = lines[0]
			if len(lines) > 1:
				lines = lines[1:]
				continue
			if len(first) > 10 or any(mark in first for mark in ("，", "。", "！", "？", "“", "”")):
				lines = lines[1:]
				continue
			break

		cleaned = []
		action_count = 0
		for line in lines:
			if is_action_line(line):
				action_count += 1
				if action_count > 1:
					continue
				if len(line) > 10:
					continue
			cleaned.append(line)

		text = "\n".join(cleaned).strip()
		text = self._suppress_overused_persona_details(text, user_input)
		text = self._suppress_repetitive_sentences(text)
		return text

	def _is_self_intro_request(self, user_input):
		text = str(user_input or "").strip()
		if not text:
			return False
		patterns = (
			"自我介绍",
			"介绍一下你自己",
			"介绍你自己",
			"你是谁",
			"说说你自己",
			"请介绍一下自己",
			"做一个自我介绍",
		)
		return any(pattern in text for pattern in patterns)

	def _build_intro_prefix(self):
		name = str(getattr(self.config, "name", "") or "我")
		profile = getattr(self.persona_system, "profile", {}) or {}
		core_setup = [str(item).strip() for item in profile.get("core_setup", []) if str(item).strip()]
		values = [item for item in core_setup[:2] if len(item) <= 18]
		if values:
			return f"我是{name}，{values[0]}。"
		identity = getattr(getattr(self, "new_persona_state", None), "immutable_core", None)
		identity = getattr(identity, "identity", None)
		archetype = str(getattr(identity, "archetype", "") or "").strip()
		species = str(getattr(identity, "species", "") or "").strip()
		if archetype:
			return f"我是{name}，{archetype}。"
		if species:
			return f"我是{name}，一名{species}。"
		return f"我是{name}。"

	def _enforce_direct_answer(self, text, user_input):
		if not isinstance(text, str):
			return text
		clean = text.strip()
		if not clean:
			return clean
		if not self._is_self_intro_request(user_input):
			return clean
		head = clean[:120]
		direct_markers = (
			f"我是{self.config.name}",
			f"我叫{self.config.name}",
			self.config.name,
			"自我介绍",
			"介绍一下",
		)
		if any(marker in head for marker in direct_markers):
			return clean
		prefix = self._build_intro_prefix()
		if clean.startswith(prefix):
			return clean
		return f"{prefix}\n\n{clean}"

	def send_message(self, user_input: str, attached_image=None, return_json=False):
		"""鍚?AI 鍙戦€佹秷鎭苟杩斿洖鍥炲銆?"""
		self.tick()
		self.last_recall_tick = datetime.now()
		self.buffer.set_system_prompt(self.config.system_prompt)

		content = user_input
		if attached_image is not None:
			content = [
				{"type": "text", "text": user_input},
				{"type": "image_url", "image_url": attached_image},
			]
		self.buffer.add_message("user", content)
		self.relation_system.on_user_message(user_input)
		self.emotion_system.apply_user_signal(user_input)
		self.emotion_state_machine.queue_signal(self._estimate_pending_signal(user_input))

		history = self.get_message_history()
		memories, recalled_memories = self.memory_system.recall_memories(history)
		route_decision, tool_report, assembled_context = self._run_harness_pipeline(user_input)
		persona_context = "\n\n".join(
			part
			for part in (
				assembled_context.slots.get("immutable_core", ""),
				assembled_context.slots.get("web_persona_context", ""),
				assembled_context.slots.get("evidence_chunks", ""),
			)
			if part
		)
		if self._requires_persona_grounding(user_input):
			local_precise_context = self.persona_system.build_precise_query_context(user_input, top_k=6, char_budget=1000)
			local_story_context = self.persona_system.build_story_context(user_input)
			if local_precise_context or local_story_context:
				persona_context = "\n\n".join(
					part
					for part in (
						persona_context,
						("Local Persona Precise Evidence:\n" + local_precise_context) if local_precise_context else "",
						("Local Persona Story Evidence:\n" + local_story_context) if local_story_context else "",
					)
					if part
				)
		tool_context_for_turn = "\n\n".join(
			part
			for part in (
				assembled_context.slots.get("web_persona_context", ""),
				assembled_context.slots.get("web_reality_context", ""),
			)
			if part
		) or "None"
		thought_data = self.thought_adapter.think(
			assembled_context,
			self.get_message_history(False),
			memories,
			recalled_memories,
			self.last_message,
		)
		assembled_context.slots["thought_output"] = "\n".join(
			"- " + thought["content"] for thought in thought_data.get("thoughts", [])
		)
		self.emotion_state_machine.update_from_thought(self._thought_signal(thought_data))
		filtered_memories = self._select_prompt_memories(user_input, memories)

		content = history[-1]["content"]
		img_data = None
		if isinstance(content, list):
			assert len(content) == 2
			assert content[0]["type"] == "text"
			assert content[1]["type"] == "image_url"
			text_content = content[0]["text"] + "\n\n((The user attached an image to this message))"
			img_data = content[1]
		else:
			text_content = content

		tool_context = tool_context_for_turn if not attached_image else self._build_tool_context(text_content)

		prompt_content = USER_TEMPLATE.format(
			**self._get_format_data(text_content, thought_data, filtered_memories, persona_context, tool_context)
		)
		if img_data:
			prompt_content = [img_data, {"type": "text", "text": prompt_content}]

		history[-1]["content"] = prompt_content
		response = self.model.generate(
			history,
			temperature=0.8,
			max_tokens=2048,
			return_json=return_json,
		)
		if not return_json:
			response = self._postprocess_assistant_response(response, user_input=text_content)

		should_remember_response = not (self._requires_external_grounding(text_content) and not self._has_tool_evidence(tool_context))
		if should_remember_response:
			self.memory_system.remember(
				self._input_to_memory(user_input, response, attached_image),
				emotion=thought_data["emotion_obj"],
			)
			self.memory_writer.remember(
				self.new_memory_state,
				event_summary=self._truncate_for_prompt(f"用户提到：{user_input}；{self.config.name} 回应：{response}", 280),
				topic_tags=self._derive_topic_tags(user_input),
				relation_impact=self._derive_relation_impact(self._estimate_pending_signal(user_input)),
				importance=0.55,
				character_emotion=thought_data.get("emotion", "平静"),
			)
			self._sync_new_persona_state()
		self.last_message = datetime.now()
		self.tick()
		self.health_monitor.record_turn_metrics(
			coverage_score=assembled_context.metadata.get("coverage_score", 0.0)
		)

		new_response = response
		if return_json:
			response = json.dumps(new_response, indent=2)
		self.buffer.add_message("assistant", new_response)
		return response

	def set_thought_visibility(self, shown: bool):
		"""璁剧疆鏄惁鏄剧ず鍐呴儴鎬濊€冦€?"""
		self.thought_system.show_thoughts = shown

	def get_mood(self):
		"""鑾峰彇 AI 褰撳墠鎯呯华鐘舵€併€?"""
		return self.emotion_system.mood

	def get_beliefs(self):
		"""鑾峰彇 AI 褰撳墠淇″康銆?"""
		semantic_notes = [
			record.content
			for record in getattr(self.new_memory_state, "semantic_records", [])
			if getattr(record, "scope", "") == "USER_SPECIFIC" and getattr(record, "content", "")
		]
		if semantic_notes:
			return semantic_notes
		return self.memory_system.get_beliefs()

	def set_mood(self, pleasure=None, arousal=None, dominance=None):
		"""璁剧疆 AI 褰撳墠鎯呯华锛岃嫢鍧囦负绌哄垯鎭㈠鍒板熀绾跨姸鎬併€?"""
		if pleasure is None and arousal is None and dominance is None:
			self.emotion_system.reset_mood()
		else:
			self.emotion_system.set_emotion(
				pleasure=pleasure,
				arousal=arousal,
				dominance=dominance,
			)

	def set_relation(self, friendliness=None, dominance=None):
		"""璁剧疆 AI 瀵圭敤鎴风殑鍏崇郴鍊笺€?"""
		self.relation_system.set_relation(friendliness=friendliness, dominance=dominance)

	def get_memories(self):
		"""鑾峰彇鐭湡璁板繂銆?"""
		return self.memory_system.get_short_term_memories()

	def consolidate_memories(self):
		"""灏嗙煭鏈熻蹇嗘暣鍚堣繘闀挎湡璁板繂銆?"""
		self.memory_system.consolidate_memories()

	def tick(self):
		"""鎺ㄨ繘鍚勪釜绯荤粺鐨勭姸鎬併€?"""
		now = datetime.now()
		delta = (now - self.last_tick).total_seconds()
		self.emotion_system.tick()
		if self.thought_system.can_reflect():
			self.thought_system.reflect()
		self.memory_system.tick(delta)

		if (now - self.last_recall_tick).total_seconds() > 2 * 3600:
			self.memory_system.surface_random_thoughts()
			print("闅忔満璁板繂宸叉诞鐜拌嚦鐭湡璁板繂")
			self.last_recall_tick = now
		self.last_tick = now

	def save(self, path):
		"""灏?AI 绯荤粺淇濆瓨鍒版寚瀹氳矾寰勩€?"""
		self._save_new_architecture_state()
		with open(path, "wb") as file:
			pickle.dump(self, file)

	@staticmethod
	def load(path):
		"""从指定路径加载 AI 系统，不存在时返回 None。"""
		if os.path.exists(path):
			print("正在加载 Ireina...")
			try:
				with open(path, "rb") as file:
					ai_system = pickle.load(file)
			except (AttributeError, ModuleNotFoundError, pickle.PickleError, EOFError):
				print("检测到旧存档与当前启动方式不兼容，将自动创建新实例。")
				return None
			ai_system._repair_runtime_state()
			ai_system._load_new_architecture_state()
			return ai_system
		return None

	@classmethod
	def load_or_create(cls, path):
		"""从指定路径加载 AI 系统，不存在则新建。"""
		ai_system = cls.load(path)
		is_new = ai_system is None
		if is_new:
			print("正在初始化 Ireina...")
			ai_system = AISystem()
			print("Ireina 初始化完成。")
		else:
			print("Ireina 加载完成。")

		ai_system.on_startup()
		if is_new:
			print(ai_system.send_message("*User logs in for the first time. Greet them warmly and make sure to introduce yourself, but keep it brief.*"))
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
		model.generate("Hello", max_tokens=1)
	except requests.HTTPError as exc:
		if exc.response.status_code == 401:
			print("提供的 API 密钥无效，请检查 .env 文件中的配置后重新运行。")
		else:
			print("暂时无法完成在线验证，将先启动程序，真正发送消息时再重试。")
			return True
		return False
	except requests.RequestException:
		print("网络连接出现问题，将先启动程序，真正发送消息时再重试。")
		return True
	except RuntimeError as exc:
		print(f"当前 LLM 配置不可用：{exc}")
		return False

	print("验证成功！")
	return True





"""运行 AI 的主模块。"""

import copy
import json
import os
import pickle
import traceback
from collections import deque
from datetime import datetime

import requests
from pydantic import BaseModel, Field

from const import AI_SYSTEM_PROMPT, SAVE_PATH, USER_TEMPLATE
from emotion_system import EmotionSystem, PersonalitySystem, RelationshipSystem
from llm import FallbackMistralLLM, MistralLLM, get_active_llm_label, get_llm_settings, has_llm_api_key
from memory_system import MemorySystem
from persona_system import PersonaSystem
from tools import DEFAULT_TOOL_REGISTRY, ToolRuntime
from thought_system import ThoughtSystem
from utils import (
	clear_screen,
	format_date,
	format_memories_to_string,
	format_time,
	is_image_url,
	time_since_last_message_string,
)
from safe_colored import Fore, Style


class MessageBuffer:
	"""保存最近消息的缓冲区，超出上限时会自动淘汰旧消息。"""

	def __init__(self, max_messages):
		self.max_messages = max_messages
		self.messages = deque(maxlen=max_messages)
		self.system_prompt = ""

	def set_system_prompt(self, prompt):
		"""设置系统提示词。"""
		self.system_prompt = prompt.strip()

	def add_message(self, role, content):
		"""向缓冲区添加一条消息。"""
		self.messages.append({"role": role, "content": content})

	def pop(self):
		"""移除并返回最后一条消息。"""
		return self.messages.pop()

	def flush(self):
		"""清空缓冲区。"""
		self.messages.clear()

	def to_list(self, include_system_prompt=True):
		"""将缓冲区转换为列表，可选是否附带系统提示词。"""
		history = []
		if include_system_prompt and self.system_prompt:
			history.append({"role": "system", "content": self.system_prompt})
		history.extend(message.copy() for message in self.messages)
		return history


GENERATE_USER_RESPONSES_PROMPT = """# Task

The human is chatting with Ireina, a friendly and empathetic virtual companion.
It aims to connect on a deeper level, and is good at providing emotional support when needed.

Given the following conversation, please suggest 3 to 5 possible responses that the HUMAN could respond to the last AI message given the conversation context.
Try to match the human's tone and style as closely as possible.

# Role descriptions

- **HUMAN**: These are messages from the human
- **AI**: These are responses from the AI model

# Format Instructions

Respond in JSON format:
```
{{
	"possible_responses": list[str]
}}
```

# Conversation History

Today is {date}. The current time is {time}

Here is the conversation history so far:

```
{conversation_history}
```

Remember, try to match the human's tone and style as closely as possible.

Possible **HUMAN** responses:"""


def suggest_responses(conversation):
	"""根据对话历史生成用户可能的回复。"""
	role_map = {
		"user": "HUMAN",
		"assistant": "AI",
	}
	if conversation:
		history_str = "\n\n".join(
			f"{role_map[msg['role']]}: {msg['content']}"
			for msg in conversation
			if msg["role"] != "system"
		)
	else:
		history_str = "No conversation yet; generate suggested greetings/starters for the human."
	now = datetime.now()
	model = MistralLLM("mistral-medium-latest")
	prompt = GENERATE_USER_RESPONSES_PROMPT.format(
		conversation_history=history_str,
		date=format_date(now),
		time=format_time(now),
	)
	data = model.generate(
		prompt,
		temperature=1.0,
		presence_penalty=1.5,
		return_json=True,
	)
	return data["possible_responses"]


class PersonalityConfig(BaseModel):
	"""AI 人格配置。"""

	open: float = Field(ge=-1.0, le=1.0)
	conscientious: float = Field(ge=-1.0, le=1.0)
	agreeable: float = Field(ge=-1.0, le=1.0)
	extrovert: float = Field(ge=-1.0, le=1.0)
	neurotic: float = Field(ge=-1.0, le=1.0)


class AIConfig(BaseModel):
	"""AI 总配置。"""

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
	"""包含各个子系统的 AI 主系统。"""

	PERSONA_EVIDENCE_KEYWORDS = (
		"故事", "经历", "设定", "背景", "过去", "原作", "剧情",
		"性格", "口头禅", "口癖", "说话方式", "语气", "称呼",
		"喜欢", "讨厌", "厌恶", "价值观", "世界观", "身份",
		"外貌", "外观", "发色", "关系", "喜好", "习惯", "经典台词",
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

		self.num_messages = 0
		self.last_message = None
		self.last_recall_tick = datetime.now()
		self.last_tick = datetime.now()

		self.buffer = MessageBuffer(20)
		self.buffer.set_system_prompt(config.system_prompt)

	def set_config(self, config):
		"""更新配置。"""
		self.config = config
		self.memory_system.config = config
		self.memory_system.belief_system.config = config
		self.thought_system.config = config
		self.emotion_system.config = config
		self.persona_system.persona_name = config.name
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
		"""为旧存档补齐运行时字段。"""
		if not hasattr(self, "tool_runtime") or self.tool_runtime is None:
			self.tool_runtime = ToolRuntime(DEFAULT_TOOL_REGISTRY)
		if not hasattr(self, "persona_system") or self.persona_system is None:
			self.persona_system = PersonaSystem(persona_name=self.config.name)
		elif not getattr(self.persona_system, "persona_name", None):
			self.persona_system.persona_name = self.config.name

	def get_message_history(self, include_system_prompt=True):
		"""获取当前对话历史。"""
		return self.buffer.to_list(include_system_prompt)

	def on_startup(self):
		"""在系统加载后执行初始化。"""
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
		return f"User: {user_msg}\n\n{self.config.name}: {ai_response}"

	def _select_prompt_memories(self, user_input, memories):
		"""为 prompt 选择最相关的短期记忆。"""
		relevant_memories = self.memory_system.short_term.retrieve_bm25(user_input, top_k=5)
		if relevant_memories:
			return relevant_memories
		if not memories:
			return []
		return memories[-5:]

	def _truncate_for_prompt(self, value, max_chars):
		"""按字符预算裁剪注入 prompt 的文本。"""
		if len(value) <= max_chars:
			return value
		return value[: max_chars - 3].rstrip() + "..."

	def _recent_assistant_messages(self, limit=3):
		"""收集最近几轮 assistant 回复，用于避免重复提及同一人设细节。"""
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

	def _get_format_data(self, content, thought_data, memories, persona_context):
		now = datetime.now()
		user_emotions = thought_data["possible_user_emotions"]
		user_emotion_list_str = ", ".join(user_emotions)
		if user_emotions:
			user_emotion_str = "The user appears to be feeling the following emotions: " + user_emotion_list_str
		else:
			user_emotion_str = "The user doesn't appear to show any strong emotion."

		thought_str = "\n".join("- " + thought["content"] for thought in thought_data["thoughts"])
		beliefs = self.memory_system.get_beliefs()
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
			"tool_context": self._build_tool_context(content),
			"recent_assistant_context": self._recent_assistant_context(),
			"persona_grounding_required": "yes" if self._requires_persona_grounding(content) else "no",
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
		return any(keyword in text for keyword in self.PERSONA_EVIDENCE_KEYWORDS)

	def _build_persona_context(self, user_input):
		text = user_input if isinstance(user_input, str) else str(user_input or "")
		base_context = self.persona_system.build_context(text)
		if not any(keyword in text for keyword in ("故事", "经历", "设定", "背景", "过去", "原作", "剧情")):
			return base_context

		story_context = self.persona_system.build_story_context(text)
		if not story_context:
			return base_context

		sections = [part for part in [base_context, "Story-Relevant Persona Evidence:\n" + story_context] if part]
		return "\n\n".join(sections)

	def _build_tool_context_legacy(self, user_input):
		return "None"
		text = user_input if isinstance(user_input, str) else str(user_input)
		contexts = []
		lower_text = text.lower()

		if ("天气" in text or "weather" in lower_text or "气温" in text) and ("今天" in text or "weather" in lower_text or "天气" in text):
			try:
				weather_result = DEFAULT_TOOL_REGISTRY.run("weather", query=text)
			except Exception as exc:
				weather_result = {"ok": False, "summary": f"天气工具调用失败：{exc}"}
			contexts.append("Weather Skill:\n" + weather_result.get("summary", ""))

		if any(keyword in text for keyword in ("故事", "经历", "设定", "背景")):
			local_story_context = self.persona_system.build_story_context(text)
			if local_story_context:
				contexts.append("Local Persona Story Evidence:\n" + local_story_context)
			try:
				search_result = DEFAULT_TOOL_REGISTRY.run(
					"web_search",
					persona_name=self.config.name,
					query=text,
					max_results=2,
					timeout=8,
				)
			except Exception as exc:
				search_result = {"snippets": [{"source": "error", "title": "web_search", "text": str(exc)}]}
			snippets = search_result.get("snippets", [])
			if snippets:
				lines = []
				for snippet in snippets[:2]:
					lines.append(f"[{snippet.get('source', 'web')} | {snippet.get('title', '')}] {snippet.get('text', '')}")
				contexts.append("Reference Search Skill:\n" + "\n".join(lines))

		return "\n\n".join(contexts) if contexts else "None"

	def _build_tool_context(self, user_input):
		text = user_input if isinstance(user_input, str) else str(user_input)
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

		if any(keyword in text for keyword in ("故事", "经历", "设定", "背景")):
			local_story_context = self.persona_system.build_story_context(text)
			if local_story_context:
				contexts.append("Local Persona Story Evidence:\n" + local_story_context)

		return "\n\n".join(contexts) if contexts else "None"

	def _build_persona_context(self, user_input):
		text = user_input if isinstance(user_input, str) else str(user_input or "")
		base_context = self.persona_system.build_context(text)
		if not self._requires_persona_grounding(text):
			return base_context

		story_context = self.persona_system.build_story_context(text)
		if not story_context:
			return base_context

		sections = [part for part in [base_context, "Story-Relevant Persona Evidence:\n" + story_context] if part]
		return "\n\n".join(sections)

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
							"For strong persona questions, answer from user-provided persona material first, then these references. If support is insufficient, say so instead of inventing details.",
						]
					)
				)

		return "\n\n".join(contexts) if contexts else "None"

	def _postprocess_assistant_response(self, response):
		if not isinstance(response, str):
			return response
		text = response.replace("\r\n", "\n").strip()
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
			if len(first) > 10 or any(mark in first for mark in ("，", "。", "、", "；", "“", "”")):
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
		return text

	def send_message(self, user_input: str, attached_image=None, return_json=False):
		"""向 AI 发送消息并返回回复。"""
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

		history = self.get_message_history()
		memories, recalled_memories = self.memory_system.recall_memories(history)
		persona_context = self._build_persona_context(user_input)
		thought_data = self.thought_system.think(
			self.get_message_history(False),
			memories,
			recalled_memories,
			self.last_message,
			persona_context,
		)
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

		prompt_content = USER_TEMPLATE.format(
			**self._get_format_data(text_content, thought_data, filtered_memories, persona_context)
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
			response = self._postprocess_assistant_response(response)

		self.memory_system.remember(
			self._input_to_memory(user_input, response, attached_image),
			emotion=thought_data["emotion_obj"],
		)
		self.last_message = datetime.now()
		self.tick()

		new_response = response
		if return_json:
			response = json.dumps(new_response, indent=2)
		self.buffer.add_message("assistant", new_response)
		return response

	def set_thought_visibility(self, shown: bool):
		"""设置是否显示内部思考。"""
		self.thought_system.show_thoughts = shown

	def get_mood(self):
		"""获取 AI 当前情绪状态。"""
		return self.emotion_system.mood

	def get_beliefs(self):
		"""获取 AI 当前信念。"""
		return self.memory_system.get_beliefs()

	def set_mood(self, pleasure=None, arousal=None, dominance=None):
		"""设置 AI 当前情绪，若均为空则恢复到基线状态。"""
		if pleasure is None and arousal is None and dominance is None:
			self.emotion_system.reset_mood()
		else:
			self.emotion_system.set_emotion(
				pleasure=pleasure,
				arousal=arousal,
				dominance=dominance,
			)

	def set_relation(self, friendliness=None, dominance=None):
		"""设置 AI 对用户的关系值。"""
		self.relation_system.set_relation(friendliness=friendliness, dominance=dominance)

	def get_memories(self):
		"""获取短期记忆。"""
		return self.memory_system.get_short_term_memories()

	def consolidate_memories(self):
		"""将短期记忆整合进长期记忆。"""
		self.memory_system.consolidate_memories()

	def tick(self):
		"""推进各个系统的状态。"""
		now = datetime.now()
		delta = (now - self.last_tick).total_seconds()
		self.emotion_system.tick()
		if self.thought_system.can_reflect():
			self.thought_system.reflect()
		self.memory_system.tick(delta)

		if (now - self.last_recall_tick).total_seconds() > 2 * 3600:
			self.memory_system.surface_random_thoughts()
			print("随机记忆已浮现至短期记忆")
			self.last_recall_tick = now
		self.last_tick = now

	def save(self, path):
		"""将 AI 系统保存到指定路径。"""
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


def _try_convert_arg(arg):
	try:
		return int(arg)
	except ValueError:
		pass

	try:
		return float(arg)
	except ValueError:
		pass

	return arg


def _parse_args(arg_list_str):
	i = 0
	tokens = []
	last_tok = ""
	in_str = False
	escape = False
	while i < len(arg_list_str):
		char = arg_list_str[i]
		if not escape and char == '"':
			in_str = not in_str
			if not in_str:
				tokens.append(last_tok)
				last_tok = ""
		elif in_str:
			last_tok += char
		elif char == " ":
			if last_tok:
				tokens.append(_try_convert_arg(last_tok))
				last_tok = ""
		else:
			last_tok += char
		i += 1
	if last_tok:
		tokens.append(_try_convert_arg(last_tok))
	return tokens


def command_parse(string):
	"""将命令行字符串解析成命令与参数。"""
	split = string.split(None, 1)
	if len(split) == 2:
		command, remaining = split
	else:
		command, remaining = string, ""
	return command, _parse_args(remaining)


def check_has_valid_key():
	if not os.getenv("MISTRAL_API_KEY"):
		print("使用 Ireina 需要提供 Mistral API 密钥。")
		print('请在项目同目录创建 .env 文件，并写入 MISTRAL_API_KEY="[你的 API 密钥]"。')
		return False

	print("正在验证 Mistral API 密钥...")
	model = MistralLLM("mistral-medium-2505")
	try:
		model.generate("Hello", max_tokens=1)
	except requests.HTTPError as exc:
		if exc.response.status_code == 401:
			print("提供的 API 密钥无效，请检查 .env 文件中的密钥后重新运行。")
		else:
			print("暂时无法完成在线验证，将先启动程序，真正发送消息时再重试。")
			return True
		return False
	except requests.RequestException:
		print("网络连接出现问题，将先启动程序，真正发送消息时再重试。")
		return True

	print("验证成功！")
	return True


def check_has_valid_key():
	settings = get_llm_settings()
	if not has_llm_api_key():
		print(f"使用 Ireina 需要提供 {get_active_llm_label()} API 密钥。")
		print('请在项目同目录创建 .env 文件，并写入 LLM_PROVIDER、LLM_API_KEY、LLM_CHAT_MODEL 等配置。')
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


def main():
	"""程序主入口。"""
	if not check_has_valid_key():
		return

	attached_image = None
	ai = AISystem.load_or_create(SAVE_PATH)
	print(f"{Fore.yellow}注意：建议不要输入任何敏感信息。{Style.reset}")

	while True:
		ai.tick()
		ai.emotion_system.print_mood()
		if attached_image:
			print(f"已附加图片：{attached_image}")
		msg = input("用户：").strip()
		if not msg:
			ai.save(SAVE_PATH)
			continue

		if msg.startswith("/"):
			command, args = command_parse(msg[1:])
			if command == "set_pleasure" and len(args) == 1:
				value = args[0]
				if isinstance(value, (int, float)):
					ai.set_mood(pleasure=value)
			elif command == "set_arousal" and len(args) == 1:
				value = args[0]
				if isinstance(value, (int, float)):
					ai.set_mood(arousal=value)
			elif command == "set_dominance" and len(args) == 1:
				value = args[0]
				if isinstance(value, (int, float)):
					ai.set_mood(dominance=value)
			elif command == "set_relation_friendliness" and len(args) == 1:
				value = args[0]
				if isinstance(value, (int, float)):
					ai.set_relation(friendliness=value)
			elif command == "set_relation_dominance" and len(args) == 1:
				value = args[0]
				if isinstance(value, (int, float)):
					ai.set_relation(dominance=value)
			elif command == "add_emotion" and len(args) == 2:
				emotion = args[0]
				value = args[1]
				if isinstance(value, (int, float)):
					ai.emotion_system.experience_emotion(emotion, value)
			elif command == "show_thoughts":
				ai.set_thought_visibility(True)
			elif command == "hide_thoughts":
				ai.set_thought_visibility(False)
			elif command == "reset_mood":
				ai.set_mood()
			elif command == "consolidate_memories":
				ai.consolidate_memories()
			elif command == "attach_image" and len(args) == 1:
				url = args[0]
				if isinstance(url, str):
					if is_image_url(url):
						attached_image = url
					else:
						print("错误：不是有效的图片链接")
			elif command == "detach_image":
				attached_image = None
			elif command == "memories":
				print("当前记忆：")
				for memory in ai.get_memories():
					print(memory.format_memory())
			elif command == "suggest":
				history = ai.get_message_history(False)
				print("正在生成可能的用户回复建议...")
				possible_responses = suggest_responses(history)
				print("可能的回复：")
				for response in possible_responses:
					print("- " + response)
			elif command == "load_persona" and len(args) == 1:
				filepath = args[0]
				if not isinstance(filepath, str):
					continue
				n = ai.persona_system.load_file(filepath)
				if n:
					print(f"已从 {filepath} 载入 {n} 个文本块")
			elif command == "clear_persona":
				ai.persona_system.clear()
				print("人设知识库已清空。")
			elif command == "persona_info":
				print(f"当前人设：{ai.persona_system.persona_name}，已载入文本块数：{ai.persona_system.chunk_count}")
			elif command in ["wipe", "reset"]:
				if os.path.exists(SAVE_PATH):
					choice = input("确定要清除 Ireina 的所有已保存数据和记忆吗？输入 yes 或 是 确认清除，输入其他内容取消：")
					if choice.strip().lower() == "yes" or choice.strip() == "是":
						os.remove(SAVE_PATH)
						input("Ireina 已重置，按回车键继续。")
						clear_screen()
						ai = AISystem()
						ai.on_startup()
			elif command == "beliefs":
				beliefs = ai.get_beliefs()
				if beliefs:
					print("已形成以下信念：")
					for belief in beliefs:
						print("- " + belief)
				else:
					print("尚未形成任何信念")
			elif command == "configupdate":
				new_config = AIConfig()
				ai.set_config(new_config)
				ai.save(SAVE_PATH)
				print("配置已更新并保存！")
			else:
				print(f"无效命令：'/{command}'")
			continue

		print()
		backup_ai = copy.deepcopy(ai)
		try:
			message = ai.send_message(msg, attached_image=attached_image)
		except Exception as exc:
			ai = backup_ai
			traceback.print_exception(type(exc), exc, exc.__traceback__)
			print()
			print(f"{ai.config.name}：抱歉，我现在好像遇到了一些问题，请稍后再试。")
		else:
			print(f"{ai.config.name}：" + message)
			ai.save(SAVE_PATH)
			attached_image = None


if __name__ == "__main__":
	main()

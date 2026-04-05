import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import json_repair
import requests
from dotenv import load_dotenv

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

EMBED_MAX_ITEMS_PER_BATCH = 96
EMBED_MAX_CHARS_PER_BATCH = 24000

DEFAULT_PROVIDER_PRESETS = {
	"mistral": {
		"label": "Mistral",
		"base_url": "https://api.mistral.ai/v1",
		"chat_model": "mistral-medium-latest",
		"embedding_model": "mistral-embed",
	},
	"openai": {
		"label": "OpenAI",
		"base_url": "https://api.openai.com/v1",
		"chat_model": "gpt-4.1-mini",
		"embedding_model": "text-embedding-3-small",
	},
	"openrouter": {
		"label": "OpenRouter",
		"base_url": "https://openrouter.ai/api/v1",
		"chat_model": "openai/gpt-4.1-mini",
		"embedding_model": None,
	},
	"siliconflow": {
		"label": "SiliconFlow",
		"base_url": "https://api.siliconflow.cn/v1",
		"chat_model": "Qwen/Qwen2.5-72B-Instruct",
		"embedding_model": "BAAI/bge-m3",
	},
	"deepseek": {
		"label": "DeepSeek",
		"base_url": "https://api.deepseek.com/v1",
		"chat_model": "deepseek-chat",
		"embedding_model": None,
	},
	"custom": {
		"label": "Custom",
		"base_url": "",
		"chat_model": "",
		"embedding_model": "",
	},
	"openai_compatible": {
		"label": "OpenAI-Compatible",
		"base_url": "",
		"chat_model": "",
		"embedding_model": "",
	},
}

LEGACY_KEY_ENV_BY_PROVIDER = {
	"mistral": "MISTRAL_API_KEY",
	"openai": "OPENAI_API_KEY",
	"deepseek": "DEEPSEEK_API_KEY",
	"openrouter": "OPENROUTER_API_KEY",
	"siliconflow": "SILICONFLOW_API_KEY",
}


class MistralEmbeddingCapacityError(RuntimeError):
	pass


@dataclass
class LLMSettings:
	provider: str
	label: str
	api_key: str
	base_url: str
	chat_model: str
	embedding_model: str | None


def _normalize_provider_name(name: str | None) -> str:
	value = (name or "").strip().lower()
	if not value:
		return "mistral"
	aliases = {
		"openai-compatible": "openai_compatible",
		"openai_compatible": "openai_compatible",
		"custom-openai": "openai_compatible",
	}
	return aliases.get(value, value)


def _get_env_api_key(provider: str, explicit_env: str | None = None, allow_global_fallback: bool = True) -> str:
	if explicit_env and os.getenv(explicit_env):
		return os.getenv(explicit_env, "").strip()
	if allow_global_fallback and os.getenv("LLM_API_KEY"):
		return os.getenv("LLM_API_KEY", "").strip()
	legacy_env = LEGACY_KEY_ENV_BY_PROVIDER.get(provider)
	if legacy_env and os.getenv(legacy_env):
		return os.getenv(legacy_env, "").strip()
	if allow_global_fallback and os.getenv("MISTRAL_API_KEY"):
		return os.getenv("MISTRAL_API_KEY", "").strip()
	return ""


def _resolve_settings(
	provider_env: str,
	api_key_env: str,
	base_url_env: str,
	chat_model_env: str,
	embedding_model_env: str,
	allow_global_key_fallback: bool = True,
) -> LLMSettings:
	provider = _normalize_provider_name(os.getenv(provider_env) or os.getenv("LLM_NAME"))
	preset = DEFAULT_PROVIDER_PRESETS.get(provider, DEFAULT_PROVIDER_PRESETS["custom"])
	api_key = _get_env_api_key(provider, explicit_env=api_key_env, allow_global_fallback=allow_global_key_fallback)
	base_url = (os.getenv(base_url_env) or preset["base_url"] or "").strip().rstrip("/")
	chat_model = (os.getenv(chat_model_env) or preset["chat_model"] or "").strip()
	embedding_model = (os.getenv(embedding_model_env) or preset["embedding_model"] or "").strip() or None
	label = preset["label"] if preset else provider
	return LLMSettings(
		provider=provider,
		label=label,
		api_key=api_key,
		base_url=base_url,
		chat_model=chat_model,
		embedding_model=embedding_model,
	)


def get_llm_settings() -> LLMSettings:
	return _resolve_settings(
		provider_env="LLM_PROVIDER",
		api_key_env="LLM_API_KEY",
		base_url_env="LLM_BASE_URL",
		chat_model_env="LLM_CHAT_MODEL",
		embedding_model_env="LLM_EMBEDDING_MODEL",
	)


def get_embedding_settings() -> LLMSettings:
	if any(
		os.getenv(name)
		for name in (
			"EMBEDDING_PROVIDER",
			"EMBEDDING_API_KEY",
			"EMBEDDING_BASE_URL",
			"EMBEDDING_MODEL",
			"LLM_EMBEDDING_PROVIDER",
			"LLM_EMBEDDING_API_KEY",
			"LLM_EMBEDDING_BASE_URL",
			"LLM_EMBEDDING_MODEL",
		)
	):
		provider_env = "EMBEDDING_PROVIDER" if os.getenv("EMBEDDING_PROVIDER") else "LLM_EMBEDDING_PROVIDER"
		api_key_env = "EMBEDDING_API_KEY" if os.getenv("EMBEDDING_API_KEY") else "LLM_EMBEDDING_API_KEY"
		base_url_env = "EMBEDDING_BASE_URL" if os.getenv("EMBEDDING_BASE_URL") else "LLM_EMBEDDING_BASE_URL"
		chat_model_env = "EMBEDDING_CHAT_MODEL" if os.getenv("EMBEDDING_CHAT_MODEL") else "LLM_EMBEDDING_CHAT_MODEL"
		embedding_model_env = "EMBEDDING_MODEL" if os.getenv("EMBEDDING_MODEL") else "LLM_EMBEDDING_MODEL"
		return _resolve_settings(
			provider_env=provider_env,
			api_key_env=api_key_env,
			base_url_env=base_url_env,
			chat_model_env=chat_model_env,
			embedding_model_env=embedding_model_env,
			allow_global_key_fallback=False,
		)
	return get_llm_settings()


def get_active_llm_label() -> str:
	return get_llm_settings().label


def has_llm_api_key() -> bool:
	return bool(get_llm_settings().api_key)


def _log_request(message: str):
	settings = get_llm_settings()
	print(f"[LLM:{settings.label}] {message}")


def _log_embedding_request(message: str):
	settings = get_embedding_settings()
	print(f"[Embed:{settings.label}] {message}")


def _is_legacy_mistral_model(model_name: str | None) -> bool:
	if not model_name:
		return False
	name = str(model_name).strip().lower()
	return name.startswith("mistral") or name.startswith("open-mistral")


def _resolve_chat_model(requested_model: str | None = None) -> str:
	settings = get_llm_settings()
	if requested_model and not (settings.provider != "mistral" and _is_legacy_mistral_model(requested_model)):
		return requested_model
	if settings.chat_model:
		return settings.chat_model
	raise RuntimeError("当前未配置聊天模型。请在 .env 中设置 LLM_CHAT_MODEL。")


def _resolve_embedding_model() -> str:
	settings = get_embedding_settings()
	if settings.embedding_model:
		return settings.embedding_model
	raise RuntimeError(
		f"当前 embedding 提供方 {settings.label} 未配置可用模型。"
		"请在 .env 中设置 EMBEDDING_MODEL 或 LLM_EMBEDDING_MODEL。"
	)


def _build_headers(settings: LLMSettings | None = None) -> dict:
	settings = settings or get_llm_settings()
	if not settings.api_key:
		raise RuntimeError("未配置 API Key。")
	headers = {
		"Accept": "application/json",
		"Authorization": f"Bearer {settings.api_key}",
		"Content-Type": "application/json",
	}
	if settings.provider == "openrouter":
		headers["HTTP-Referer"] = "https://local.wetalk"
		headers["X-Title"] = "Wetalk"
	return headers

def _chat_url() -> str:
	settings = get_llm_settings()
	if not settings.base_url:
		raise RuntimeError("未配置 LLM_BASE_URL。")
	return f"{settings.base_url}/chat/completions"


def _embed_url() -> str:
	settings = get_embedding_settings()
	if not settings.base_url:
		raise RuntimeError("未配置 embedding 的 BASE_URL。")
	return f"{settings.base_url}/embeddings"


def _chunk_embedding_inputs(inputs, max_items=EMBED_MAX_ITEMS_PER_BATCH, max_chars=EMBED_MAX_CHARS_PER_BATCH):
	if isinstance(inputs, str):
		return [inputs]

	batches = []
	current_batch = []
	current_chars = 0
	for text in inputs:
		text = "" if text is None else str(text)
		text_len = len(text)
		if current_batch and (len(current_batch) >= max_items or current_chars + text_len > max_chars):
			batches.append(current_batch)
			current_batch = []
			current_chars = 0
		current_batch.append(text)
		current_chars += text_len
		if text_len > max_chars:
			batches.append(current_batch)
			current_batch = []
			current_chars = 0

	if current_batch:
		batches.append(current_batch)
	return batches


def mistral_request(messages, model=None, max_tries=7, **kwargs):
	settings = get_llm_settings()
	headers = _build_headers(settings)
	request_timeout = kwargs.pop("timeout", 120)
	data = {
		"model": _resolve_chat_model(model),
		"messages": messages,
		**kwargs,
	}
	max_delay = 20
	for tries in range(max_tries):
		attempt = tries + 1
		_log_request(f"聊天请求，第 {attempt}/{max_tries} 次，模型：{data['model']}")
		try:
			response = requests.post(_chat_url(), json=data, headers=headers, timeout=request_timeout)
		except requests.Timeout as exc:
			wait_time = min(max_delay, 2 ** attempt)
			_log_request(f"聊天请求超时：{exc}，{wait_time} 秒后重试")
			if tries < max_tries - 1:
				time.sleep(wait_time)
				continue
			raise
		except requests.RequestException as exc:
			wait_time = min(max_delay, 2 ** attempt)
			_log_request(f"聊天请求网络异常：{type(exc).__name__}: {exc}")
			if tries < max_tries - 1:
				_log_request(f"{wait_time} 秒后重试")
				time.sleep(wait_time)
				continue
			raise
		if response.ok:
			_log_request(f"聊天请求成功，状态码：{response.status_code}")
			break
		status = response.status_code
		preview = response.text[:300].replace("\n", " ")
		_log_request(f"聊天请求失败，状态码：{status}，响应：{preview}")
		if status in (429, 502, 503, 504, 520) and tries < max_tries - 1:
			wait_time = min(max_delay, 2 ** (tries + 1))
			_log_request(f"{wait_time} 秒后自动重试")
			time.sleep(wait_time)
			continue
		response.raise_for_status()
	else:
		response.raise_for_status()
	return response.json()


def _embed_batch(inputs):
	settings = get_embedding_settings()
	headers = _build_headers(settings)
	data = {
		"model": _resolve_embedding_model(),
		"input": inputs,
	}
	max_delay = 20
	for tries in range(5):
		attempt = tries + 1
		size_info = 1 if isinstance(inputs, str) else len(inputs)
		_log_embedding_request(f"embedding 请求，第 {attempt}/5 次，文本块数：{size_info}")
		try:
			response = requests.post(_embed_url(), json=data, headers=headers, timeout=30)
		except requests.Timeout as exc:
			wait_time = min(max_delay, 2 ** attempt)
			_log_embedding_request(f"embedding 请求超时：{exc}，{wait_time} 秒后重试")
			if tries < 4:
				time.sleep(wait_time)
				continue
			raise
		except requests.RequestException as exc:
			wait_time = min(max_delay, 2 ** attempt)
			_log_embedding_request(f"embedding 请求网络异常：{type(exc).__name__}: {exc}")
			if tries < 4:
				_log_embedding_request(f"{wait_time} 秒后重试")
				time.sleep(wait_time)
				continue
			raise
		if response.ok:
			_log_embedding_request(f"embedding 请求成功，状态码：{response.status_code}")
			break
		status = response.status_code
		preview = response.text[:300].replace("\n", " ")
		_log_embedding_request(f"embedding 请求失败，状态码：{status}，响应：{preview}")
		if status in (429, 502, 503, 504) and tries < 4:
			wait_time = min(max_delay, 2 ** (tries + 1))
			_log_embedding_request(f"{wait_time} 秒后自动重试")
			time.sleep(wait_time)
			continue
		if status == 429 and "service_tier_capacity_exceeded" in response.text:
			raise MistralEmbeddingCapacityError("Embedding service tier capacity exceeded.")
		response.raise_for_status()
	else:
		if response.status_code == 429 and "service_tier_capacity_exceeded" in response.text:
			raise MistralEmbeddingCapacityError("Embedding service tier capacity exceeded.")
		response.raise_for_status()

	embed_res = response.json()
	if isinstance(inputs, str):
		return embed_res["data"][0]["embedding"]
	return [obj["embedding"] for obj in embed_res["data"]]


def mistral_embed_texts(inputs, progress_callback=None):
	if isinstance(inputs, str):
		if progress_callback:
			progress_callback(current=1, total=1, stage="embedding", detail="单条文本向量化中")
		return _embed_batch(inputs)
	if not inputs:
		if progress_callback:
			progress_callback(current=0, total=0, stage="embedding", detail="没有需要向量化的文本")
		return []

	batches = _chunk_embedding_inputs(inputs)
	_log_embedding_request(f"embedding 总任务：{len(inputs)} 个文本块，已拆分为 {len(batches)} 个批次")
	all_embeddings = []
	processed = 0
	for index, batch in enumerate(batches, start=1):
		if progress_callback:
			progress_callback(current=processed, total=len(inputs), stage="embedding", detail=f"正在向量化第 {index}/{len(batches)} 批")
		_log_embedding_request(f"开始处理 embedding 批次 {index}/{len(batches)}")
		batch_embeddings = _embed_batch(batch)
		all_embeddings.extend(batch_embeddings)
		processed += len(batch)
		if progress_callback:
			progress_callback(current=processed, total=len(inputs), stage="embedding", detail=f"已完成 {processed}/{len(inputs)} 条向量化")
	return all_embeddings


def _convert_system_to_user(messages):
	new_messages = []
	for msg in messages:
		role = msg["role"]
		content = msg["content"]
		if role == "system":
			role = "user"
			content = f"<SYSTEM>{content}</SYSTEM>"
		new_messages.append({"role": role, "content": content})
	return new_messages


class MistralLLM:
	def __init__(self, model=None):
		self.model = _resolve_chat_model(model)

	def _extract_message_content(self, message):
		if not isinstance(message, dict):
			return ""
		content = message.get("content")
		if isinstance(content, str):
			return content
		if isinstance(content, list):
			parts = []
			for item in content:
				if isinstance(item, dict):
					text = item.get("text")
					if text:
						parts.append(str(text))
				elif item:
					parts.append(str(item))
			if parts:
				return "".join(parts)
		reasoning = message.get("reasoning_content")
		if isinstance(reasoning, str):
			return reasoning
		if isinstance(reasoning, list):
			parts = []
			for item in reasoning:
				if isinstance(item, dict):
					text = item.get("text")
					if text:
						parts.append(str(text))
				elif item:
					parts.append(str(item))
			if parts:
				return "".join(parts)
		return ""

	def _parse_json(self, response):
		try:
			return json.loads(response, strict=True)
		except json.JSONDecodeError:
			pass

		try:
			return json.loads(response, strict=False)
		except json.JSONDecodeError:
			return json_repair.loads(response, skip_json_loads=True)

	def generate(self, prompt, return_json=False, schema=None, n=None, **kwargs):
		if schema and not return_json:
			raise ValueError("return_json must be True if schema is provided")
		if isinstance(prompt, str):
			prompt = [{"role": "user", "content": prompt}]

		settings = get_llm_settings()
		if settings.provider == "mistral" and self.model not in [
			"mistral-small-latest",
			"mistral-medium-latest",
			"mistral-large-latest",
			"mistral-medium-2508",
			"mistral-medium-2505",
			"open-mistral-nemo",
		]:
			prompt = _convert_system_to_user(prompt)

		if schema:
			format_data = {
				"type": "json_schema",
				"json_schema": {
					"name": "json_object",
					"schema": schema,
					"strict": True,
				},
			}
		else:
			format_data = {"type": "json_object"} if return_json else {"type": "text"}

		response = mistral_request(
			prompt,
			**kwargs,
			n=(n or 1),
			model=self.model,
			response_format=format_data,
		)
		if n:
			outputs = [self._extract_message_content(choice.get("message", {})) for choice in response["choices"]]
			if return_json:
				return [self._parse_json(output) for output in outputs]
			return outputs

		output = self._extract_message_content(response["choices"][0].get("message", {}))
		if return_json:
			return self._parse_json(output)
		return output


_DEFAULT_FALLBACK_PREF = [
	"mistral-medium-2508",
	"mistral-medium-2505",
	"mistral-small-latest",
	"mistral-large-2411",
]


class FallbackMistralLLM(MistralLLM):
	def __init__(self, models=_DEFAULT_FALLBACK_PREF):
		settings = get_llm_settings()
		if settings.provider == "mistral":
			resolved_models = list(models)
		else:
			resolved_models = [settings.chat_model or _resolve_chat_model()]
		super().__init__(resolved_models[0])
		self.models = resolved_models

	def generate(self, prompt, return_json=False, schema=None, n=None, **kwargs):
		last_error = None
		for model in self.models:
			try:
				self.model = _resolve_chat_model(model)
				return super().generate(
					prompt,
					return_json=return_json,
					schema=schema,
					n=n,
					**kwargs,
				)
			except requests.HTTPError as exc:
				last_error = exc
				if exc.response.status_code not in (429, 502, 503, 504, 520):
					raise
		if last_error is not None:
			raise last_error
		raise RuntimeError("All models failed")


if __name__ == "__main__":
	model = FallbackMistralLLM()
	print(model.generate("Hello! What can you do?"))

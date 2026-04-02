import math
import random
import re
import uuid
from collections import deque
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from belief_system import BeliefSystem
from const import *
from emotion_system import Emotion
from llm import MistralLLM, mistral_embed_texts
from utils import conversation_to_string, get_approx_time_ago_str, normalize_text
from utils import format_timestamp


IMPORTANCE_PROMPT = """Your task is to rate the importance of the given memory from 1 to 10.

- A score of 1 represents trivial things or basic chit-chat with no information of importance.
- A score of 10 represents things that are very important.

Return ONLY an integer, nothing else. Your response must not contain any non-numeric characters.

<memory>
{memory}
</memory>

The importance score of the above memory is <fill_in_the_blank_here>/10.
"""


def get_importance(memory):
	"""将给定记忆的重要性评分为 1 到 10。"""
	model = MistralLLM("open-mistral-nemo")
	prompt = IMPORTANCE_PROMPT.format(memory=memory)
	output = model.generate(prompt, temperature=0.0)
	try:
		score = int(output)
	except ValueError:
		score = 3

	return max(1, min(score, 10))


def cosine_similarity(x, y):
	x = np.array(x, dtype=np.float32)
	y = np.array(y, dtype=np.float32)
	assert x.ndim == 1 and y.ndim == 1
	x = x[np.newaxis, ...]
	y = y[np.newaxis, ...]
	sim = x @ y.T
	sim /= np.linalg.norm(x) * np.linalg.norm(y, axis=1)
	return np.squeeze(sim)


def normalize_vector(vector):
	"""将向量做 L2 归一化，便于用点积等价计算余弦相似度。"""
	array = np.asarray(vector, dtype=np.float32)
	norm = np.linalg.norm(array)
	if norm == 0:
		return array
	return array / norm


def tokenize_for_bm25(text):
	"""将文本切成适合 BM25 的 token，并兼容中日文无空格文本。"""
	normalized = normalize_text(text)
	tokens = normalized.split()
	if len(tokens) > 1:
		return tokens

	compact = normalized.replace(" ", "")
	if not compact:
		return []

	if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]", compact):
		char_tokens = [char for char in compact if not char.isspace()]
		if len(char_tokens) == 1:
			return char_tokens
		bigrams = [
			"".join(char_tokens[index:index + 2])
			for index in range(len(char_tokens) - 1)
		]
		return char_tokens + bigrams

	return tokens or [compact]


class Memory:
	"""表示一条已存储的记忆。"""

	def __init__(self, content, strength=1.0, emotion=None):
		now = datetime.now()
		self.timestamp = now
		self.last_accessed = now
		self.content = content
		self.embedding = None
		self.id = str(uuid.uuid4())
		self.strength = strength
		self.emotion = emotion or Emotion()

	def get_recency_factor(self, from_creation=False):
		"""根据时间与强度返回记忆的新近度。"""
		timestamp = self.timestamp if from_creation else self.last_accessed
		seconds = (datetime.now() - timestamp).total_seconds()
		days = seconds / 86400
		return math.exp(-days / (self.strength * MEMORY_DECAY_TIME_MULT))

	def get_retention_prob(self):
		"""计算这条记忆每 24 小时被保留的概率。"""
		recency = self.get_recency_factor()
		if recency > MEMORY_RECENCY_FORGET_THRESHOLD:
			return 1.0
		return math.exp(-1 / (MEMORY_DECAY_TIME_MULT * self.strength))

	def reinforce(self):
		"""在记忆被回想时强化它。"""
		self.strength += 0.5
		self.last_accessed = datetime.now()

	def format_memory(self):
		"""将记忆格式化为字符串。"""
		timedelta = datetime.now() - self.timestamp
		time_ago_str = get_approx_time_ago_str(timedelta)
		time_format = format_timestamp(self.timestamp)
		return (
			f"<memory timestamp=\"{time_format}\""
			f" time_ago=\"{time_ago_str}\">{self.content}</memory>"
		)

	def encode(self, embedding=None):
		"""在尚未生成时为记忆创建语义嵌入。"""
		if self.embedding is None:
			embed = embedding
			if embed is None:
				embed = mistral_embed_texts([self.content])[0]
			self.embedding = normalize_vector(embed)


class FAISSMemory:
	"""使用 FAISS 保存长期记忆，并用删除标记处理移除。"""

	def __init__(self, embed_size, rebuild_threshold=500):
		self.embed_size = embed_size
		self.rebuild_threshold = rebuild_threshold
		self.index = faiss.IndexFlatIP(embed_size)
		self.memories = []
		self.index_to_memory_idx = []
		self.memory_ids = {}
		self.count = 0
		self._writes_since_rebuild = 0

	def __getstate__(self):
		"""序列化时只保留有效记忆和 FAISS 索引字节。"""
		self._rebuild_index()
		state = self.__dict__.copy()
		state["index"] = faiss.serialize_index(self.index)
		return state

	def __setstate__(self, state):
		"""反序列化后重建索引对象与映射。"""
		self.__dict__.update(state)
		index_data = self.index
		if index_data is not None and not hasattr(index_data, "ntotal"):
			self.index = faiss.deserialize_index(index_data)
		self.memory_ids = {}
		for idx, memory in enumerate(self.memories):
			if memory is None:
				continue
			self.memory_ids[memory.id] = idx
		self.count = sum(memory is not None for memory in self.memories)
		self.index_to_memory_idx = [idx for idx, memory in enumerate(self.memories) if memory is not None]
		self._writes_since_rebuild = 0

	def _rebuild_index(self):
		"""重建索引并移除被删除标记占用的空间。"""
		active_memories = [memory for memory in self.memories if memory is not None]
		self.index = faiss.IndexFlatIP(self.embed_size)
		self.memories = []
		self.index_to_memory_idx = []
		self.memory_ids = {}
		self.count = 0
		for memory in active_memories:
			memory.embedding = normalize_vector(memory.embedding)
			self.memories.append(memory)
			self.index_to_memory_idx.append(len(self.memories) - 1)
			self.memory_ids[memory.id] = len(self.memories) - 1
			self.count += 1
		if active_memories:
			vectors = np.vstack([memory.embedding for memory in active_memories]).astype(np.float32)
			self.index.add(vectors)
		self._writes_since_rebuild = 0

	def add_memory(self, memory):
		"""添加一条记忆。"""
		memory.embedding = normalize_vector(memory.embedding)
		vector = np.asarray(memory.embedding, dtype=np.float32)[np.newaxis, :]
		self.memories.append(memory)
		memory_idx = len(self.memories) - 1
		self.index_to_memory_idx.append(memory_idx)
		self.memory_ids[memory.id] = memory_idx
		self.index.add(vector)
		self.count += 1
		self._writes_since_rebuild += 1
		if self._writes_since_rebuild >= self.rebuild_threshold:
			self._rebuild_index()

	def delete_memory(self, memory):
		"""用删除标记移除一条记忆。"""
		memory_idx = self.memory_ids.pop(memory.id, None)
		if memory_idx is None:
			return
		if self.memories[memory_idx] is not None:
			self.memories[memory_idx] = None
			self.count -= 1

	def _get_active_memories(self):
		return [memory for memory in self.memories if memory is not None]

	def retrieve(self, query, k, remove=False):
		"""返回最相关的前 K 条记忆。"""
		if not self.count or self.index.ntotal == 0:
			return []

		query_vec = normalize_vector(mistral_embed_texts([query])[0]).astype(np.float32)
		candidate_count = min(self.index.ntotal, max(k * 3, k))
		scores, indices = self.index.search(query_vec[np.newaxis, :], candidate_count)

		candidates = []
		for similarity, index_idx in zip(scores[0], indices[0]):
			if index_idx < 0:
				continue
			memory_idx = self.index_to_memory_idx[index_idx]
			memory = self.memories[memory_idx]
			if memory is None:
				continue
			recency = memory.get_recency_factor()
			score = float(similarity) + 0.5 * recency
			candidates.append((score, memory))

		if len(candidates) < k and candidate_count < self.index.ntotal:
			full_scores, full_indices = self.index.search(query_vec[np.newaxis, :], self.index.ntotal)
			seen = {memory.id for _, memory in candidates}
			for similarity, index_idx in zip(full_scores[0], full_indices[0]):
				if index_idx < 0:
					continue
				memory_idx = self.index_to_memory_idx[index_idx]
				memory = self.memories[memory_idx]
				if memory is None or memory.id in seen:
					continue
				recency = memory.get_recency_factor()
				score = float(similarity) + 0.5 * recency
				candidates.append((score, memory))

		candidates.sort(key=lambda item: item[0], reverse=True)
		retrieved = [memory for _, memory in candidates[:k]]
		if remove:
			for memory in retrieved:
				self.delete_memory(memory)
		return retrieved

	def get_memories(self):
		"""获取所有有效记忆。"""
		return self._get_active_memories()

	def recall_random(self, remove=False):
		"""按新近度加权随机回想一小批记忆。"""
		memories = self._get_active_memories()
		if not memories:
			return []

		sample_size = min(20, len(memories))
		sampled = random.sample(memories, sample_size)
		weights = [memory.get_recency_factor() for memory in sampled]
		recalled = []
		pool = sampled[:]
		pool_weights = weights[:]
		for _ in range(min(5, len(pool))):
			choice = random.choices(pool, pool_weights)[0]
			index = pool.index(choice)
			recalled.append(pool.pop(index))
			pool_weights.pop(index)

		if remove:
			for memory in recalled:
				self.delete_memory(memory)
		return recalled


class ShortTermMemory:
	"""保存最近被访问或经历过的短期记忆。"""

	capacity = 20

	def __init__(self):
		self.memories = deque()

	def add_memory(self, memory):
		"""添加一条新记忆。"""
		for existing in self.memories:
			if existing.content.lower() == memory.content.lower():
				self._move_to_end(existing)
				break
		else:
			self.memories.append(memory)

	def _move_to_end(self, memory):
		if memory in self.memories:
			self.memories.remove(memory)
			self.memories.append(memory)

	def add_memories(self, memories):
		"""添加一组记忆。"""
		for memory in memories:
			self.add_memory(memory)

	def flush_old_memories(self):
		"""移出超出容量的记忆并返回它们。"""
		old_memories = []
		while len(self.memories) > self.capacity:
			old_memories.append(self.memories.popleft())
		return old_memories

	def clear_memories(self):
		"""清空所有短期记忆。"""
		self.memories.clear()

	def get_memories(self):
		"""返回全部短期记忆列表。"""
		return list(self.memories)

	def retrieve_bm25(self, query, top_k=5):
		"""使用 BM25 从短期记忆中检索最相关的内容。"""
		if not self.memories:
			return []
		corpus = [memory.content for memory in self.memories]
		tokenized_corpus = [tokenize_for_bm25(text) for text in corpus]
		bm25 = BM25Okapi(tokenized_corpus)
		query_tokens = tokenize_for_bm25(query)
		if not query_tokens:
			return []
		scores = bm25.get_scores(query_tokens)
		scored = [
			(score, memory)
			for memory, score in zip(self.memories, scores)
			if score > 0
		]
		scored.sort(key=lambda item: item[0], reverse=True)
		return [memory for _, memory in scored[:top_k]]

	def rehearse(self, query):
		"""强化与查询相似的记忆。"""
		if not self.memories:
			return

		# 相似的记忆更有可能被强化。
		corpus = [memory.content for memory in self.memories]
		tokenized_corpus = [tokenize_for_bm25(text) for text in corpus]
		bm25 = BM25Okapi(tokenized_corpus)
		query_tokens = tokenize_for_bm25(query)
		if not query_tokens:
			return
		scores = bm25.get_scores(query_tokens)

		reinforced = []
		for memory, score in zip(self.memories, scores):
			if random.random() < score:
				reinforced.append((score, memory))
		reinforced.sort(key=lambda pair: pair[0])
		reinforced = reinforced[-3:]
		for _, memory in reinforced:
			memory.reinforce()
			self._move_to_end(memory)


class LongTermMemory:
	"""长期保存记忆的系统。"""

	def __init__(self):
		self.lsh = FAISSMemory(LSH_VEC_DIM)

	def retrieve(self, query, k, remove=False):
		"""返回最相关的前 K 条记忆。"""
		return self.lsh.retrieve(query, k, remove=remove)

	def recall_random(self, remove=False):
		"""随机回想一小部分记忆。"""
		return self.lsh.recall_random(remove=remove)

	def add_memory(self, memory):
		"""添加一条长期记忆。"""
		memory.encode()
		self.lsh.add_memory(memory)

	def add_memories(self, memories):
		"""批量添加长期记忆。"""
		if not memories:
			return
		memory_texts = [memory.content for memory in memories]
		embeddings = mistral_embed_texts(memory_texts)
		for memory, embed in zip(memories, embeddings):
			memory.encode(embed)
			self.lsh.add_memory(memory)

	def get_memories(self):
		"""返回全部长期记忆。"""
		return self.lsh.get_memories()

	def forget_memory(self, memory):
		"""从长期记忆中移除一条记忆。"""
		self.lsh.delete_memory(memory)

	def tick(self, delta):
		"""执行一次记忆更新。"""
		for memory in self.get_memories():
			retain_prob = memory.get_retention_prob()
			if retain_prob >= 1.0:
				continue
			forget_prob = 1 - retain_prob
			prob = 1 - ((1 - forget_prob) ** (delta / 86400))
			if random.random() < prob:
				print("遗忘了一条长期未被回想的记忆。")
				print(f"已遗忘记忆：{memory.content}")
				self.forget_memory(memory)


class MemorySystem:
	"""AI 的记忆系统。"""

	def __init__(self, config):
		self.config = config
		self.short_term = ShortTermMemory()
		self.long_term = LongTermMemory()
		self.last_memory = datetime.now()
		self.belief_system = BeliefSystem(config)
		self.importance_counter = 0.0

	def get_beliefs(self):
		return self.belief_system.get_beliefs()

	def reset_importance(self):
		"""重置重要性累计值。"""
		self.importance_counter = 0.0

	def remember(self, content, emotion=None, is_insight=False):
		"""添加一条新记忆。"""
		importance = get_importance(content)
		strength = 1 + (importance - 1) / 2
		self.last_memory = datetime.now()
		self.short_term.add_memory(Memory(content, strength=strength, emotion=emotion))
		self.importance_counter += importance / 10
		if not is_insight and importance >= 6:
			self.belief_system.generate_new_belief(content, importance / 10)

	def recall(self, query):
		"""回想并返回最相关的记忆。"""
		self.short_term.rehearse(query)
		memories = self.long_term.retrieve(query, MEMORY_RETRIEVAL_TOP_K, remove=True)
		for memory in memories:
			memory.reinforce()
			self.short_term.add_memory(memory)
		return memories

	def tick(self, dt):
		"""执行一次系统更新。"""
		now = datetime.now()
		old_memories = self.short_term.flush_old_memories()
		for memory in old_memories:
			self.long_term.add_memory(memory)
		timedelta = now - self.last_memory
		if timedelta.total_seconds() > 6 * 3600:
			# 空闲 6 小时后，将短期记忆整合进长期记忆。
			self.consolidate_memories()
			self.last_memory = now

		self.long_term.tick(dt)
		self.belief_system.tick(dt)

	def consolidate_memories(self):
		"""将所有短期记忆整合进长期记忆。"""
		print("正在将所有短期记忆整合至长期记忆...")
		memories = self.short_term.get_memories()
		self.long_term.add_memories(memories)
		self.short_term.clear_memories()

	def surface_random_thoughts(self):
		"""将一小批随机思绪带回短期记忆。"""
		memories = self.long_term.recall_random(remove=True)
		for memory in memories:
			memory.reinforce()
		self.short_term.add_memories(memories)

	def get_short_term_memories(self):
		"""获取全部短期记忆。"""
		memories = self.short_term.get_memories()
		memories.sort(key=lambda memory: memory.timestamp)
		return memories

	def retrieve_long_term(self, query, top_k):
		"""从长期记忆中检索前 K 条最相关内容。"""
		return self.long_term.retrieve(query, top_k, remove=False)

	def recall_memories(self, messages):
		"""根据查询返回短期记忆和回想出的长期记忆。"""
		messages = [message for message in messages if message["role"] != "system"]
		context = conversation_to_string(messages[-3:])
		recalled_memories = self.recall(context)
		return self.get_short_term_memories(), recalled_memories

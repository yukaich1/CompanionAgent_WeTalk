"""管理 AI 情绪的系统。"""

import math
import time
import random
from datetime import datetime
from const import (
	SUMMARIZE_PERSONALITY,
	EMOTION_MAP,
	EMOTION_PROMPTS,
	MODD_INTENSITY_FACTOR,
	PERSONALITY_INTENSITY_FACTOR,
	MOOD_HALF_LIFE,
	EMOTION_HALF_LIFE,
	MOOD_CHANGE_VEL,
	APPRAISAL_PROMPT,
	EMOTION_APPRAISAL_CONTEXT_TEMPLATE,
	APPRAISAL_SCHEMA
)
from utils import (
	num_to_str_sign,
	val_to_symbol_color,
	format_memories_to_string,
	conversation_to_string
)
from safe_colored import Fore
from llm import MistralLLM, FallbackMistralLLM

RELATIONSHIP_FRIENDLINESS_GAIN = 0.55
RELATIONSHIP_DOMINANCE_GAIN = 0.5
EMOTION_RESPONSE_GAIN = 0.45
MESSAGE_SIGNAL_GAIN = 0.28


def get_default_mood(openness, conscientious, extrovert, agreeable, neurotic):
	"""将大五人格数值转换为 PAD 情绪向量。"""
	# 与其他维度不同，神经质越低越好。
	pleasure = 0.12 * extrovert + 0.59 * agreeable - 0.19 * neurotic
	arousal = 0.15 * openness + 0.3 * agreeable + 0.57 * neurotic
	dominance = 0.25 * openness + 0.17 * conscientious + 0.6 * extrovert - 0.32 * agreeable
	return (pleasure, arousal, dominance)
	

def summarize_personality(openness, conscientious, extrovert, agreeable, neurotic):
	"""将人格数值总结为自然语言人格描述。"""
	model = FallbackMistralLLM()
	
	# 将 -1.0 到 +1.0 的范围重新映射到 0 到 100。
	open_scaled = (openness + 1) * 50
	conscientious_scaled = (conscientious + 1) * 50
	extrovert_scaled = (extrovert + 1) * 50
	agreeable_scaled = (agreeable + 1) * 50
	stability_scaled = (1 - neurotic) * 50
	
	personality_str = "\n".join([
		f"Openness: {round(open_scaled)}/100",
		f"Conscientiousness: {round(conscientious_scaled)}/100",
		f"Extroversion: {round(extrovert_scaled)}/100",
		f"Agreeableness: {round(agreeable_scaled)}/100",
		f"Emotional Stability: {round(stability_scaled)}/100"
	])
	prompt = SUMMARIZE_PERSONALITY.format(
		personality_values=personality_str
	)
	#print(prompt)
	return model.generate(
		prompt,
		temperature=0.1
	)


class PersonalitySystem:
	"""定义 AI 人格的系统。"""

	def __init__(self, openness, conscientious, extrovert, agreeable, neurotic):
		self.open = openness
		self.conscientious = conscientious
		self.extrovert = extrovert
		self.agreeable = agreeable
		self.neurotic = neurotic
		
		self.summary = ""
	
	def get_summary(self):
		"""生成人格数值摘要。"""
		if not self.summary:
			self.summary = summarize_personality(
				self.open,
				self.conscientious,
				self.extrovert,
				self.agreeable,
				self.neurotic
			)
		return self.summary
	

class Emotion:
	"""定义情绪或心境的三维向量。"""

	def __init__(
		self,
		pleasure=0.0,
		arousal=0.0,
		dominance=0.0
	):
		self.pleasure = pleasure
		self.arousal = arousal
		self.dominance = dominance
	
	@classmethod
	def from_personality(cls, openness, conscientious, extrovert, agreeable, neurotic):
		"""将给定人格数值转换为 Emotion 向量对象。"""
		return cls(*get_default_mood(openness, conscientious, extrovert, agreeable, neurotic))
	
	def __add__(self, other):
		if isinstance(other, Emotion):
			return Emotion(
				self.pleasure + other.pleasure,
				self.arousal + other.arousal,
				self.dominance + other.dominance
			)
		return NotImplemented
		
	__radd__ = __add__
		
	def __iadd__(self, other):
		if isinstance(other, Emotion):
			self.pleasure += other.pleasure
			self.arousal += other.arousal
			self.dominance += other.dominance
			return self
		return NotImplemented

	def __sub__(self, other):
		if isinstance(other, Emotion):
			return Emotion(
				self.pleasure - other.pleasure,
				self.arousal - other.arousal,
				self.dominance - other.dominance
			)
		return NotImplemented

	def __isub__(self, other):
		if isinstance(other, Emotion):
			self.pleasure -= other.pleasure
			self.arousal -= other.arousal
			self.dominance -= other.dominance
			return self
		return NotImplemented

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			return self.__class__(
				self.pleasure * other,
				self.arousal * other,
				self.dominance * other
			)
		return NotImplemented

	__rmul__ = __mul__

	def __imul__(self, other):
		if isinstance(other, (int, float)):
			self.pleasure *= other
			self.arousal *= other
			self.dominance *= other
			return self
		return NotImplemented

	def __truediv__(self, other):
		if isinstance(other, (int, float)):
			return self.__class__(
				self.pleasure / other,
				self.arousal / other,
				self.dominance / other
			)
		return NotImplemented

	def __itruediv__(self, other):
		if isinstance(other, (int, float)):
			self.pleasure /= other
			self.arousal /= other
			self.dominance /= other
			return self
		return NotImplemented

	def dot(self, other):
		"""返回与给定情绪的点积对齐程度。"""
		return (
			self.pleasure * other.pleasure
			+ self.arousal * other.arousal
			+ self.dominance * other.dominance
		)

	def get_intensity(self):
		"""获取情绪强度。"""
		return math.sqrt(self.pleasure**2 + self.arousal ** 2 + self.dominance**2) / math.sqrt(3)

	def distance(self, other):
		"""返回两种情绪之间的距离。"""
		delta_pleasure = self.pleasure - other.pleasure
		delta_arousal = self.arousal - other.arousal
		delta_dominance = self.dominance - other.dominance

		return math.sqrt(delta_pleasure**2 + delta_arousal**2 + delta_dominance**2)

	def get_norm(self):
		return max(abs(self.pleasure), abs(self.arousal), abs(self.dominance))

	def clamp(self):
		"""若向量超出范围，则按范数进行裁剪。"""
		norm = self.get_norm()
		if norm > 1:
			self /= norm
	
	def copy(self):
		"""创建当前情绪向量的副本。
		对副本的修改不会影响原对象。"""
		return self.__class__(
			self.pleasure,
			self.arousal,
			self.dominance
		)
		
	def is_same_octant(self, other):
		"""检查两种情绪是否位于同一象限。"""
		return (
			(self.pleasure >= 0) == (other.pleasure >= 0)
			and (self.arousal >= 0) == (other.arousal >= 0)
			and (self.dominance >= 0) == (other.dominance >= 0)
		)
	
	def __repr__(self):
		return f"{self.__class__.__name__}({round(self.pleasure, 2):.2f}, " \
			f"{round(self.arousal, 2):.2f}, {round(self.dominance, 2):.2f})"


class RelationshipSystem:
	"""管理 AI 与用户关系的系统。"""
	
	def __init__(self):
		self.friendliness = 0.0
		self.dominance = 0.0
		self.last_signal = ""
		self.interaction_count = 0
		self.positive_event_count = 0
		
	def set_relation(
		self,
		friendliness=None,
		dominance=None
	):
		"""设置关系数值。"""
		if friendliness is not None:
			self.friendliness = max(-100, min(friendliness, 100))
		if dominance is not None:
			self.dominance = max(-100, min(dominance, 100))
	
	def tick(self, dt):
		"""执行一次关系更新。"""
		num_days = dt / 86400
		self.friendliness *= math.exp(-num_days/90)
		self.dominance *= math.exp(-num_days/120)

	def on_user_message(self, text):
		"""根据用户消息对关系进行轻微调整。"""
		text = (text or "").strip()
		if not text:
			return
		if not hasattr(self, "interaction_count"):
			self.interaction_count = 0
		if not hasattr(self, "positive_event_count"):
			self.positive_event_count = 0
		self.interaction_count += 1

		friendliness_delta = 0.0
		dominance_delta = 0.0
		praise_tokens = ("喜欢你", "想你", "爱你", "可爱", "厉害", "真棒", "好喜欢", "谢谢你", "辛苦了", "在意你")
		warm_tokens = ("早安", "晚安", "拜托啦", "陪陪我", "抱抱", "想和你聊", "一直陪着", "信任你")
		self_disclosure_tokens = ("我觉得", "我有点", "我今天", "我想", "我最近", "我其实", "我有", "我正在", "我担心", "我害怕")
		interest_tokens = ("旅行", "魔女", "自由", "风景", "故事", "书", "天空", "扫帚", "冒险")
		negative_tokens = ("闭嘴", "烦死了", "讨厌你", "滚", "无聊", "笨蛋", "别烦我", "懒得理你")
		conflict_tokens = ("你错了", "你真差", "没意思", "骗人", "失望")

		explicit_positive = any(token in text for token in praise_tokens)
		warm_contact = any(token in text for token in warm_tokens)
		self_disclosure = any(token in text for token in self_disclosure_tokens)
		shared_interest = any(token in text for token in interest_tokens)
		negative_contact = any(token in text for token in negative_tokens)
		value_conflict = any(token in text for token in conflict_tokens)

		if explicit_positive:
			friendliness_delta += 0.42
		if warm_contact:
			friendliness_delta += 0.2
		if self_disclosure:
			friendliness_delta += 0.14
		if shared_interest and (explicit_positive or self_disclosure or warm_contact):
			friendliness_delta += 0.1
		if negative_contact:
			friendliness_delta -= 0.55
			dominance_delta += 0.08
		if value_conflict:
			friendliness_delta -= 0.28
			dominance_delta += 0.05

		is_meaningful_positive = friendliness_delta > 0.18
		if is_meaningful_positive:
			self.positive_event_count += 1

		warmup_factor = min(1.0, 0.25 + self.interaction_count / 20)
		trust_factor = min(1.0, 0.35 + self.positive_event_count / 10)
		if friendliness_delta > 0:
			friendliness_delta *= warmup_factor * trust_factor
		else:
			friendliness_delta *= max(0.7, warmup_factor)

		friendliness_delta *= MESSAGE_SIGNAL_GAIN
		dominance_delta *= MESSAGE_SIGNAL_GAIN
		self.last_signal = text[:40]
		if friendliness_delta or dominance_delta:
			self.change_relationship(friendliness_delta, dominance_delta)

	def on_emotion(self, emotion, intensity):
		"""在触发情绪时调用。"""
		if emotion not in EMOTION_MAP or emotion == "Neutral":
			return
		if not hasattr(self, "positive_event_count"):
			self.positive_event_count = 0

		friendliness_weights = {
			"Joy": 0.18,
			"Gratitude": 0.35,
			"Love": 0.2,
			"HappyFor": 0.22,
			"Admiration": 0.18,
			"Pity": 0.08,
			"Hope": 0.08,
			"Relief": 0.1,
			"Anger": -0.3,
			"Hate": -0.4,
			"Resentment": -0.2,
			"Reproach": -0.18,
			"Distress": -0.08,
			"Disappointment": -0.12,
		}
		dominance_weights = {
			"Pride": 0.12,
			"Admiration": -0.05,
			"Anger": 0.12,
			"Reproach": 0.08,
			"Love": -0.05,
			"Gratitude": -0.04,
			"Fear": -0.08,
		}

		friendliness = friendliness_weights.get(emotion, 0.0) * intensity * RELATIONSHIP_FRIENDLINESS_GAIN
		dominance = dominance_weights.get(emotion, 0.0) * intensity * RELATIONSHIP_DOMINANCE_GAIN
		if friendliness > 0:
			friendliness *= min(1.0, 0.3 + self.positive_event_count / 10)
		if friendliness or dominance:
			self.change_relationship(friendliness, dominance)
		
	def change_relationship(self, friendliness, dominance):
		"""调整关系数值。"""
		friendliness = max(-0.85, min(0.85, friendliness))
		dominance = max(-0.45, min(0.45, dominance))
		
		if (friendliness > 0) == (self.friendliness > 0):
			friendliness *= 1 - abs(self.friendliness) / 105
		if (dominance > 0) == (self.dominance > 0):
			dominance *= 1 - abs(self.dominance) / 110
		
		self.set_relation(
			self.friendliness + friendliness,
			self.dominance + dominance
		)	
	
	def print_relation(self):
		"""显示关系数值。"""
		print("关系：")
		print("--------")
		string = val_to_symbol_color(self.friendliness, 20, Fore.green, Fore.red, val_scale=100)
		print(f"友好度：{string}")
		string = val_to_symbol_color(self.dominance, 20, Fore.cyan, Fore.light_magenta, val_scale=100)
		print(f"支配度：{string}")
	
	def get_string(self):
		"""将关系数值转换为字符串。"""
		return "\n".join((
			"Friendliness: " + val_to_symbol_color(self.friendliness, 20, val_scale=100),
			"Dominance: " + val_to_symbol_color(self.dominance, 20, val_scale=100)
		))
		
	
class EmotionSystem:
	"""管理情绪状态的系统。"""
	def __init__(
		self,
		personality_system,
		relation_system,
		config
	):
		base_mood = Emotion.from_personality(
			personality_system.open,
			personality_system.conscientious,
			personality_system.extrovert,
			personality_system.agreeable,
			personality_system.neurotic
		)
		self.personality_system = personality_system
		self.relation = relation_system
		self.base_mood = base_mood
		self.mood = self.get_base_mood()
		self.last_update = time.time()
		self.config = config
		
	def _emotions_from_appraisal(self, appraisal):
		events = appraisal["events"]
		actions = appraisal["actions"]
		
		self_event = events["self"]
		other_event = events["other"]
		
		is_prospect = self_event["is_prospective"]
	
		relation = self.relation.friendliness  # 0-100
		agreeable = self.personality_system.agreeable*100  # 乘以 100，将其映射到 0-100
		
		eff_relation = (agreeable + relation) / 2
		
		emotions = []
		if self_event["event"] and self_event["desirability"] != 0:
			desirability = self_event["desirability"]	
			intensity = abs(desirability) / 100
			if desirability > 0:
				emotion = "Hope" if is_prospect else "Joy"
			else:	
				emotion = "Fear" if is_prospect else "Distress"
			
			emotions.append((emotion, intensity))
			
		if other_event["event"] and other_event["desirability"] != 0:
			desirability = other_event["desirability"]
			relation_mod = math.sqrt(abs(eff_relation)/100)
			intensity = relation_mod * abs(desirability)/100
			if desirability > 0:
				emotion = "HappyFor" if eff_relation >= 0 else "Resentment"
			else:
				emotion = "Pity" if eff_relation >= 0 else "Gloating"
			emotions.append((emotion, intensity))
		
		self_act = actions["self"]
		other_act = actions["other"]
		
		if self_act["action"] and self_act["praiseworthiness"] != 0:
			praiseworthiness = self_act["praiseworthiness"]
			intensity = abs(praiseworthiness) / 100
			if praiseworthiness > 0:
				emotions.append(("Pride", intensity))
			else:
				emotions.append(("Shame", intensity))
			
		if other_act["action"] and other_act["praiseworthiness"] != 0:
			praiseworthiness = other_act["praiseworthiness"]
			intensity = abs(praiseworthiness) / 100	
			if praiseworthiness > 0:
				emotions.append(("Admiration", intensity))
			else:
				emotions.append(("Reproach", intensity))
	
		emotions.sort(key=lambda p: p[1], reverse=True)
		return emotions
		
	def appraisal(self, messages, memories, beliefs):
		memories_str = format_memories_to_string(memories, "None")
		sys_prompt = APPRAISAL_PROMPT.format(
			sys_prompt=self.config.system_prompt
		)
		if beliefs:
			belief_str = "\n".join(f"- {belief}" for belief in beliefs)
		else:
			belief_str = "None"
		
		content = messages[-1]["content"]

		img_data = None
		if isinstance(content, list):
			assert len(content) == 2
			assert content[0]["type"] == "text"
			assert content[1]["type"] == "image_url"
			text_content = content[0]["text"] + "\n\n((The user attached an image to this message - please see the attached image.))"
			img_data = content[1]
		else:
			text_content = content
			
		prompt = EMOTION_APPRAISAL_CONTEXT_TEMPLATE.format(
			memories=memories_str,
			history=conversation_to_string(messages),
			beliefs=belief_str,
			user_input=text_content
		)
		prompt_content = prompt
		if img_data:
			prompt_content = [
				{"type":"text", "text":prompt_content},
				img_data
			]
			
		history = [
			{"role":"system", "content":sys_prompt},
			{"role":"user", "content":"[BEGIN MESSAGE HISTORY]"},
			*messages[:-1],
			{"role":"user", "content":"[END MESSAGE HISTORY]"},
			{"role":"user", "content":prompt_content}
		]
		
		model = FallbackMistralLLM()
		emotion_data = model.generate(
			history,
			temperature=0.2,
			schema=APPRAISAL_SCHEMA,
			return_json=True
		)
		return self._emotions_from_appraisal(emotion_data)
		
	def set_emotion(
		self,
		pleasure=None,
		arousal=None,
		dominance=None
	):
		if pleasure is not None:
			self.mood.pleasure = max(-1.0, min(1.0, pleasure))
		if arousal is not None:
			self.mood.arousal = max(-1.0, min(1.0, arousal))
		if dominance is not None:
			self.mood.dominance = max(-1.0, min(1.0, dominance))

	def add_emotion(self, emotion):
		"""兼容旧调用：直接将情绪向量叠加到当前心境。"""
		if emotion is None:
			return
		if not isinstance(emotion, Emotion):
			return
		self.mood += emotion
		self.clamp_mood()

	def reset_mood(self):
		"""重置 AI 的心境。"""
		self.mood = self.get_base_mood()
	
	def _get_adv(self, val):
		if abs(val) > 0.9:
			adv = "extremely"
		elif abs(val) > 0.65:
			adv = "very"
		elif abs(val) > 0.35:
			adv = "moderately"
		else:
			adv = "slightly"
		return adv
	
	def _get_mood_word(self, val, pos_str, neg_str):
		if abs(val) < 0.04:
			return "neutral"
		
		return self._get_adv(val) + " " + (pos_str if val >= 0 else neg_str)
			
	def get_mood_long_description(self):
		mood = self.mood
		
		pleasure_desc = f"Pleasure: {num_to_str_sign(mood.pleasure, 2)} ({self._get_mood_word(mood.pleasure, 'pleasant', 'unpleasant')})"
		arousal_desc = f"Arousal: {num_to_str_sign(mood.arousal, 2)} ({self._get_mood_word(mood.arousal, 'energized', 'soporific')})"
		dominance_desc = f"Dominance: {num_to_str_sign(mood.dominance, 2)} ({self._get_mood_word(mood.dominance, 'dominant', 'submissive')})"	
		
		if mood.arousal > 0.04:
			adv = self._get_adv(mood.arousal)
			arousal_desc += f" - Will be {adv} upbeat and energetic in tone."
		elif mood.arousal < -0.04:
			adv = self._get_adv(mood.arousal)
			arousal_desc += f" - Will be {adv} soft and gentle in tone."
						
		if mood.dominance > 0.04:
			adv = self._get_adv(mood.dominance)
			dominance_desc += f" - Feels {adv} compelled to lead the conversation."
		elif mood.dominance < -0.04:
			adv = self._get_adv(mood.dominance)
			dominance_desc += f" - Feels {adv} compelled to defer more and let others lead the conversation."
										
		
		return "\n".join([
			pleasure_desc,
			arousal_desc,
			dominance_desc
		])
		
	def print_mood(self):
		mood = self.mood
		print("心境：")
		print("--------")
		string = val_to_symbol_color(mood.pleasure, 20, Fore.green, Fore.red)
		print(f"愉悦度：{string}")
		string = val_to_symbol_color(mood.arousal, 20, Fore.yellow, Fore.cornflower_blue)
		print(f"唤醒度：{string}")
		string = val_to_symbol_color(mood.dominance, 20, Fore.cyan, Fore.light_magenta)
		print(f"支配度：{string}")
		print()
		self.relation.print_relation()
		print()
		
	def get_mood_name(self):
		mood = self.mood
		if mood.get_intensity() < 0.05:
			return "neutral"
		
		if mood.pleasure >= 0:
			if mood.arousal >= 0:
				return "exuberant" if mood.dominance >= 0 else "dependent"
			else:
				return "relaxed" if mood.dominance >= 0 else "docile"
		else:
			if mood.arousal >= 0:
				return "hostile" if mood.dominance >= 0 else "anxious"
			else:
				return "disdainful" if mood.dominance >= 0 else "bored"

	def get_mood_description(self):
		mood_name = self.get_mood_name()
		if mood_name != "neutral":
			mood = self.mood
			adv = self._get_adv(mood.get_intensity())
			mood_name = adv + " " + mood_name
		
		return mood_name

	def get_mood_prompt(self):
		mood_desc = self.get_mood_description()
		prompt = EMOTION_PROMPTS[self.get_mood_name()]
		return f"{mood_desc} - {prompt}"

	def experience_emotion(self, name, intensity):
		emotion = Emotion(*EMOTION_MAP[name])
		emotion.pleasure *= random.triangular(0.9, 1.1)
		emotion.arousal *= random.triangular(0.9, 1.1)
		emotion.dominance *= random.triangular(0.9, 1.1)
		
		mood_align = emotion.dot(self.mood)
		personality_align = emotion.dot(self.get_base_mood())
		
		intensity_mod = (
			MODD_INTENSITY_FACTOR * mood_align
			+ PERSONALITY_INTENSITY_FACTOR * personality_align
		)
		intensity *= 1.0 + intensity_mod
		intensity *= EMOTION_RESPONSE_GAIN
		intensity = max(0.02, min(0.55, intensity))
		
		self.relation.on_emotion(name, intensity)
		emotion *= intensity
		self.mood += emotion
		return emotion
		
	def clamp_mood(self):
		self.mood.clamp()
			
	def get_base_mood(self):
		now = datetime.now()
		hour = now.hour + now.minute / 60 + now.second / 3600
		
		energy_cycle = -math.cos(math.pi * hour / 12)
		base_mood = self.base_mood.copy()
		
		if energy_cycle > 0:
			energy_cycle_mod = (1.0 - base_mood.arousal) * energy_cycle
		else:
			energy_cycle_mod = (-1.0 - base_mood.arousal) * abs(energy_cycle)
		
		energy_cycle_mod *= 0.5
		
		base_mood.pleasure += self.relation.friendliness / 100
		base_mood.dominance += self.relation.dominance / 100
		
		base_mood.arousal += energy_cycle_mod  # 白天更高，夜晚更低。
		base_mood.clamp()
		return base_mood

	def _tick_mood_decay(self, t):
		r = 0.5 ** (t / MOOD_HALF_LIFE)
		
		self.mood += (self.get_base_mood() - self.mood) * (1 - r)

	def tick(self, dt=None):
		if dt is None:
			dt = time.time() - self.last_update

		self.relation.tick(dt)
		self.last_update = time.time()
		t = dt
		
		substep = max(1, min(t / 10, 30))
		while t > 0:
			step = min(t, substep)
			self._apply_mood_noise(step)
			self._tick_mood_decay(step)
			t -= step

	def _apply_mood_noise(self, t):
		# 为心境变化加入少量随机扰动。
		neurotic_mult = 3**self.personality_system.neurotic
		mood_noise_stdev = 0.006 * neurotic_mult * math.sqrt(t)
		self.mood.pleasure += random.gauss(0, mood_noise_stdev)
		self.mood.arousal += random.gauss(0, mood_noise_stdev)
		self.mood.dominance += random.gauss(0, mood_noise_stdev)
		self.mood.clamp()
		
		
if __name__ == "__main__":
	kwargs = dict(
		openness=0.35,
		conscientious=0.22,
		extrovert=0.18,
		agreeable=0.93,
		neurotic=-0.1,
	)
	print(get_default_mood(**kwargs))
	print(summarize_personality(**kwargs))

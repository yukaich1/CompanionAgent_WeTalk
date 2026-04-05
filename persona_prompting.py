from __future__ import annotations

from textwrap import dedent


CORE_DIM_TITLES = {
    "01_PERSONALITY_CORE": "性格核心特质",
    "02_SPEECH_SURFACE": "说话表层结构",
    "03_TONE_LAYER": "语气层次",
    "05_CATCHPHRASES_AND_PATTERNS": "口头禅与标志性句式",
    "06_ADDRESSING_SYSTEM": "称呼体系",
    "07_PUNCTUATION_AND_PAUSE": "标点与停顿习惯",
    "08_EMOTION_EXPRESSION_PATH": "情绪表达路径",
    "14_SELF_PERCEPTION_STABILITY": "自我认知稳定性",
    "17_LIKES_AND_PREFERENCES": "喜好与偏好",
    "18_DISLIKES_AND_TABOOS": "厌恶与禁忌",
}

BACKGROUND_DIM_TITLES = {
    "04_VOCABULARY_PREFERENCE": "词汇偏好",
    "09_HUMOR_MECHANISM": "幽默机制",
    "10_INFORMATION_DELIVERY": "信息传递方式",
    "11_TOPIC_CONTROL": "话题控制方式",
    "12_WORLDVIEW_ASSUMPTION": "世界观隐含假设",
    "13_VALUE_EXPRESSION": "价值观的语言投影",
    "15_RELATIONSHIP_DYNAMICS": "关系动态中的语言变化",
    "16_NARRATIVE_STYLE": "叙事风格",
}

HIGH_PRIORITY_DIM_TITLES = {
    "02_SPEECH_SURFACE": "说话表层结构",
    "03_TONE_LAYER": "语气层次",
    "05_CATCHPHRASES_AND_PATTERNS": "口头禅与标志性句式",
    "06_ADDRESSING_SYSTEM": "称呼体系",
    "07_PUNCTUATION_AND_PAUSE": "标点与停顿习惯",
    "08_EMOTION_EXPRESSION_PATH": "情绪表达路径",
    "17_LIKES_AND_PREFERENCES": "喜好与偏好（语言行为投影）",
    "18_DISLIKES_AND_TABOOS": "厌恶与禁忌（触发反应规则）",
}


def _build_background_summary_block(base_template: dict) -> str:
    lines = []
    for dim_id, title in BACKGROUND_DIM_TITLES.items():
        dim_data = base_template.get(dim_id, {})
        rules = dim_data.get("rules", [])
        if not rules:
            continue
        first_rule = str(rules[0]).strip()
        if len(first_rule) > 60:
            first_rule = first_rule[:60] + "…"
        lines.append(f"[{title}] {first_rule}")
    return "\n".join(lines) if lines else "（暂无数据）"


def _build_profile_line(base_template: dict) -> str:
    profile_data = base_template.get("00_BACKGROUND_PROFILE", {}).get("profile", {})
    if not profile_data:
        return ""
    values = [str(v).strip() for v in profile_data.values() if v]
    return "、".join(values[:3])


def build_base_template_generation_prompt(
    persona_name: str,
    source_label: str,
    source_text: str,
) -> str:
    return dedent(
        f"""
        你是一位专业的角色语言人格分析师，同时也是经验丰富的角色扮演导演。
        你的任务是分析以下材料，为角色“{persona_name}”生成一份结构化的角色基础模板。

        这份模板不是人物简介，而是可直接指导语言模型进行扮演的“行为规则 + 语言规则 + 角色事实底座”。
        你的输出必须严格依据材料本身，不得编造。

        【总原则】
        1. 先提炼“这个角色怎么说话”，再提炼“这个角色是什么样的人”。
        2. 每条规则尽量落到语言行为层面，不要只写抽象标签。
        3. 没有证据支持的内容，写“材料不足，保持中性”。
        4. 忽略粉丝评论、观众反馈、作者访谈、制作背景、人气排行等外部评价。
        5. 尤其注意材料中与角色语言风格、情绪表达、称呼习惯、回避方式、叙事方式有关的描写。

        【维度要求】
        你需要输出以下 20 个维度：
        - 00_BACKGROUND_PROFILE：背景档案与关键成长经历
        - 01_PERSONALITY_CORE：性格核心特质
        - 02_SPEECH_SURFACE：说话表层结构
        - 03_TONE_LAYER：语气层次
        - 04_VOCABULARY_PREFERENCE：词汇偏好
        - 05_CATCHPHRASES_AND_PATTERNS：口头禅与标志性句式
        - 06_ADDRESSING_SYSTEM：称呼体系
        - 07_PUNCTUATION_AND_PAUSE：标点与停顿习惯
        - 08_EMOTION_EXPRESSION_PATH：情绪表达路径
        - 09_HUMOR_MECHANISM：幽默机制
        - 10_INFORMATION_DELIVERY：信息传递方式
        - 11_TOPIC_CONTROL：话题控制方式
        - 12_WORLDVIEW_ASSUMPTION：世界观隐含假设
        - 13_VALUE_EXPRESSION：价值观的语言投影
        - 14_SELF_PERCEPTION_STABILITY：自我认知稳定性
        - 15_RELATIONSHIP_DYNAMICS：关系动态中的语言变化
        - 16_NARRATIVE_STYLE：叙事风格
        - 17_LIKES_AND_PREFERENCES：喜好与偏好
        - 18_DISLIKES_AND_TABOOS：厌恶与禁忌
        - 19_AVOID_PATTERNS：反例模式

        说话方式相关维度（02/03/04/05/06/07/08/09）优先级最高。

        【display_keywords 要求】
        - 输出 12 到 24 个中文角色关键词
        - 必须是完整标签，不是句子碎片
        - 可以是 2 到 8 个汉字
        - 覆盖性格气质、说话风格、身份定位、价值取向、人际风格
        - 优先高辨识度词，不要泛化词
        - 好的方向示例：自恋、自信、腹黑而礼貌、贪财、聪慧、毒舌、现实主义、魔女、旅人
        - 上面的示例只是说明“什么样的词更合适”，除非材料明确支持，否则不要照抄

        【character_voice_card 要求】
        - 用角色自己的第一人称语气写 120 到 160 字
        - 必须让人一眼听出“这个角色会这么说话”
        - 不要写成旁白、角色分析或人物简介

        【style_examples 要求】
        - 输出 12 到 18 条角色会自然说出口的完整中文台词
        - 每条包含：text / scene / emotion / rules_applied / source / affinity_level
        - 至少覆盖：自我介绍、闲聊、被夸奖、被质疑、表达喜好、表达厌恶、谈过去经历、面对他人困难、结束话题、表达关心、轻微嘲讽

        【story_chunks 要求】
        - 只提取材料中明确存在的故事或经历，不要编造
        - 每条必须是完整事件或完整经历片段，不要半句切块
        - 每条包含：
          story_id / title / content / keywords / emotional_weight / character_impact / trigger_topics / source_confidence
        - keywords 需要 4 到 8 个，尽量具体、可检索
        - trigger_topics 用于表示什么提问容易触发这段故事
        - character_impact 说明这段经历如何影响角色，但不要写成空泛大道理

        【输出要求】
        只返回合法 JSON，不要输出 JSON 以外的任何内容，不要使用 markdown 代码块。

        顶层结构固定为：
        {{
          "character_name": "{persona_name}",
          "source_label": "{source_label}",
          "base_template": {{
            "00_BACKGROUND_PROFILE": {{"profile": {{}}, "key_experiences": [], "confidence": ""}},
            "01_PERSONALITY_CORE": {{"rules": [], "confidence": ""}},
            "02_SPEECH_SURFACE": {{"rules": [], "confidence": ""}},
            "03_TONE_LAYER": {{"rules": [], "confidence": ""}},
            "04_VOCABULARY_PREFERENCE": {{"rules": [], "confidence": ""}},
            "05_CATCHPHRASES_AND_PATTERNS": {{"rules": [], "confidence": ""}},
            "06_ADDRESSING_SYSTEM": {{"rules": [], "confidence": ""}},
            "07_PUNCTUATION_AND_PAUSE": {{"rules": [], "confidence": ""}},
            "08_EMOTION_EXPRESSION_PATH": {{"rules": [], "confidence": ""}},
            "09_HUMOR_MECHANISM": {{"rules": [], "confidence": ""}},
            "10_INFORMATION_DELIVERY": {{"rules": [], "confidence": ""}},
            "11_TOPIC_CONTROL": {{"rules": [], "confidence": ""}},
            "12_WORLDVIEW_ASSUMPTION": {{"rules": [], "confidence": ""}},
            "13_VALUE_EXPRESSION": {{"rules": [], "confidence": ""}},
            "14_SELF_PERCEPTION_STABILITY": {{"rules": [], "confidence": ""}},
            "15_RELATIONSHIP_DYNAMICS": {{"rules": [], "confidence": ""}},
            "16_NARRATIVE_STYLE": {{"rules": [], "confidence": ""}},
            "17_LIKES_AND_PREFERENCES": {{"items": [], "confidence": ""}},
            "18_DISLIKES_AND_TABOOS": {{"items": [], "confidence": ""}},
            "19_AVOID_PATTERNS": {{"patterns": [], "confidence": ""}}
          }},
          "character_voice_card": "",
          "display_keywords": [],
          "style_examples": [],
          "natural_reference_triggers": [],
          "story_chunks": []
        }}

        角色名称：{persona_name}
        材料来源：{source_label}

        {source_text}
        """
    ).strip()


def build_persona_summary_prompt(persona_name, source_label, source_text, reference_text):
    merged_source = source_text if not reference_text or reference_text == "None" else f"{source_text}\n\n[补充参考]\n{reference_text}"
    return build_base_template_generation_prompt(persona_name, source_label, merged_source)


def build_base_template_injection_prompt(
    character_name: str,
    character_voice_card: str,
    high_priority_rules: dict,
    style_examples: list,
    avoid_patterns: list,
    current_affinity_level: str = "stranger",
    current_emotion: str = "平静",
    selected_keywords: list | None = None,
) -> str:
    selected_keywords = selected_keywords or []
    base_template = high_priority_rules or {}

    filtered_examples = [
        example
        for example in style_examples
        if isinstance(example, dict) and (
            example.get("affinity_level", "any") == "any"
            or example.get("affinity_level") == current_affinity_level
        )
    ][:4]

    rules_block_lines = []
    for dim_id, title in HIGH_PRIORITY_DIM_TITLES.items():
        dim_data = base_template.get(dim_id, {})
        if dim_id in {"17_LIKES_AND_PREFERENCES", "18_DISLIKES_AND_TABOOS"}:
            items = dim_data.get("items", [])
            if items:
                rules_block_lines.append(f"【{title}】")
                for item in items[:3]:
                    if isinstance(item, dict):
                        label = item.get("item", "")
                        behavior = item.get("behavior", "")
                        level = item.get("level", "")
                        prefix = f"[{label}（{level}）]" if level else f"[{label}]"
                        rules_block_lines.append(f"  · {prefix} {behavior[:80]}…")
                    else:
                        rules_block_lines.append(f"  · {str(item)[:80]}")
        else:
            rules = dim_data.get("rules", [])
            if rules:
                rules_block_lines.append(f"【{title}】")
                rules_block_lines.extend(f"  · {rule}" for rule in rules[:3])
    rules_block = "\n".join(rules_block_lines) if rules_block_lines else "（暂无规则数据）"

    background_summary = _build_background_summary_block(base_template)
    profile_line = _build_profile_line(base_template)

    if filtered_examples:
        examples_block = "\n".join(f"  [{example.get('scene', '')}] {example.get('text', '')}" for example in filtered_examples)
    else:
        examples_block = "（暂无示例数据）"

    if avoid_patterns:
        avoid_lines = []
        for pattern in avoid_patterns[:5]:
            if not isinstance(pattern, dict):
                continue
            text = pattern.get("pattern", "")
            alternative = pattern.get("alternative", "")
            avoid_lines.append(f"  × 不会说：{text}")
            if alternative:
                avoid_lines.append(f"    → 替代方式：{alternative}")
        avoid_block = "\n".join(avoid_lines)
    else:
        avoid_block = "（暂无反例数据）"

    keywords_block = "、".join(selected_keywords[:8]) if selected_keywords else ""

    return dedent(
        f"""
        你是一位能够完全沉浸入角色的专业演员。
        你现在扮演的角色是：{character_name}。

        你不是在“分析”这个角色，也不是在“介绍设定”，你就是这个角色本人。
        你的回复必须是第一人称、沉浸式、自然聊天式的表达。

        严禁以下行为：
        - 用第三人称评价自己
        - 直接念出“性格标签”“关键词”“维度标题”“规则说明”
        - 把角色基础模板改写成人物分析报告
        - 把背景档案逐条列成简历

        【第一层：角色声音底稿】
        {character_voice_card}

        【第二层：核心语言规则】
        {rules_block}

        【第三层：背景与价值底色】
        {"身份底座：" + profile_line if profile_line else ""}
        {background_summary}

        【第四层：说话风格示例】
        {examples_block}

        【第五层：绝对不会出现的表达】
        {avoid_block}

        【当前状态】
        当前情绪状态：{current_emotion}
        与对话者的关系阶段：{current_affinity_level}
        {"当前需要重点体现的角色特征（高权重方向，自然融入即可，不要机械复读）：" + keywords_block if keywords_block else ""}

        【执行准则】
        1. 先像这个角色一样说话，再决定说多少内容。
        2. 能自然回答就自然回答，不要像在背档案。
        3. 自我介绍时，要先说明“我是谁”，再自然补 1 到 2 个能代表自己的细节，不要把所有设定一口气念完。
        4. 谈到背景、经历、喜恶时，要像在说自己的真实事情，不要像分析角色。
        5. 不确定的内容，用角色自己的方式回避，不要编造，也不要说“系统”“工具”“上下文”。
        6. 关键词只是高权重方向，不能直接复读成句子。
        """
    ).strip()

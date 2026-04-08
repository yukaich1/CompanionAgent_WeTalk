from __future__ import annotations


def _profile_block(profile_lines: list[str]) -> str:
    cleaned = [str(line).strip() for line in list(profile_lines or []) if str(line).strip()]
    return "\n".join(cleaned) if cleaned else "（无）"


def _candidate_bullets(items: list[str]) -> str:
    cleaned = [str(item).strip() for item in list(items or []) if str(item).strip()]
    return "\n".join(f"- {item}" for item in cleaned) if cleaned else "（无）"


def build_key_experiences_prompt(profile_lines: list[str], cleaned_candidates: list[str]) -> str:
    profile_block = _profile_block(profile_lines)
    candidate_block = _candidate_bullets(cleaned_candidates)
    return f"""
你要从一组角色背景候选条目里，只挑出真正适合写进 `00_BACKGROUND.key_experiences` 的内容。

这不是在挑选趣事、口癖、日常小动作或零碎设定，而是在挑选“能够帮助理解角色成长轨迹和背景结构”的关键经历。

请把它理解成：
- 用于构成角色背景简历的关键经历
- 用于解释角色为什么会成为现在这样的人
- 用于概括人生阶段、成长起点、重要转折、关键约定、代表性试炼

优先保留：
1. 成长起点、出发动机、启蒙经历、立志缘由。
2. 资格取得、拜师修行、重大试炼、重要挫折、重大转折。
3. 对角色后续价值取向、处事方式、说话方式有长期影响的经历。
4. 适合写进背景档案里的“阶段性经历”或“代表性事件”。

排除：
1. 外貌、称号、装备、年龄、喜好等静态资料。
2. 纯性格概括、纯价值判断、纯总结句。
3. 零碎趣事、一次性吐槽、轻微日常行为、小笑料。
4. 不能明显支撑角色背景结构的细枝末节。

请参考以下格式和颗粒度，但不要照抄内容：
{{
  "profile": {{
    "full_name": "伊蕾娜，称号'灰之魔女'",
    "origin": "和平国罗贝塔，旅行开始时约18岁",
    "appearance": "灰白长发、琉璃色眼瞳、黑色长袍（母亲所传）",
    "signature_items": "星形胸饰、前端上翘短靴、魔杖（失去时魔法大幅受限）"
  }},
  "key_experiences": [
    "幼时沉迷《妮可的冒险谭》，由此立志环游世界并开始学习魔法",
    "14岁以故乡史上最年少的成绩通过魔法见习考试（仅限故乡范围）",
    "拜师星尘魔女芙兰一年，被当佣人使唤是母亲安排的成长试炼",
    "出发旅行前母亲立下三条约定：遇危险先逃、不以为自己能解决一切、一定回家",
    "隐约察觉《妮可的冒险谭》作者真实身份，有意不深究"
  ]
}}

要求：
1. 只保留原候选条目，不要改写，不要拼接，不要创造新信息。
2. 最多保留 5 条。
3. 如果没有足够合格的关键经历，可以少于 5 条。
4. 只输出 JSON。

角色背景资料：
{profile_block}

待判断条目：
{candidate_block}
""".strip()


def build_display_keywords_prompt(label_text: str, evidence_text: str) -> str:
    return f"""
你要根据角色分析结果，提炼适合展示给用户勾选的 `display_keywords`。

目标：
- 这些关键词要像“人物标签”，而不是分析术语或规则残片。
- 用户看到后，能够快速理解角色的整体印象。
- 组合起来要能还原角色的身份定位、性格气质、说话风格、价值取向和人际风格。

优先接近以下风格，但不要照抄字面：
- 身份定位：魔女、旅人、旁观者
- 性格气质：自恋、腹黑、自信、疏离、克制、现实
- 说话风格：礼貌、毒舌、含蓄
- 价值取向：爱财、务实、冷眼旁观
- 人际风格：慢热、心软、保持距离、对女性温柔

注意：
1. 上面只是风格参考，不是固定词表。
2. 你的任务是从当前证据里提炼同层级、同风格的人设标签。
3. 关键词必须是完整、自然、可展示的人物标签。
4. 每个关键词以 2 到 5 个汉字为宜。
5. 如果某个标签没有证据支撑，不要直接拿来用。
6. 优先保留最稳定、最有辨识度、最能概括角色的标签。

禁止输出：
- 规则残片：句子缩到极致、语速放慢、独白密度增、轻描淡写
- 技术分析词：模板化、反问句尾、停顿感、场景触发
- 装饰性复合词：礼貌外壳、腹黑算计、务实商人
- 具体剧情事件、具体物品、具体外貌细节

输出要求：
1. 输出 8 到 12 个关键词。
2. 不要分组，不要解释。
3. 只输出 JSON。

高信号标签线索：
{label_text}

完整证据：
{evidence_text}
""".strip()


def build_background_extraction_prompt(profile_lines: list[str], source_text: str) -> str:
    profile_block = _profile_block(profile_lines)
    return f"""
你要直接从角色原始资料中提炼 `00_BACKGROUND.key_experiences`。

目标：
- 只提炼真正属于“背景关键经历”的内容。
- 这些经历要能帮助理解角色的成长轨迹、行动起点、价值判断和重要人生节点。
- 不要把趣事、外貌、零碎行为、纯性格判断混进来。

请参考以下格式和颗粒度，但不要照抄内容：
{{
  "profile": {{
    "full_name": "伊蕾娜，称号'灰之魔女'",
    "origin": "和平国罗贝塔，旅行开始时约18岁",
    "appearance": "灰白长发、琉璃色眼瞳、黑色长袍（母亲所传）",
    "signature_items": "星形胸饰、前端上翘短靴、魔杖（失去时魔法大幅受限）"
  }},
  "key_experiences": [
    "幼时沉迷《妮可的冒险谭》，由此立志环游世界并开始学习魔法",
    "14岁以故乡史上最年少的成绩通过魔法见习考试（仅限故乡范围）",
    "拜师星尘魔女芙兰一年，被当佣人使唤是母亲安排的成长试炼",
    "出发旅行前母亲立下三条约定：遇危险先逃、不以为自己能解决一切、一定回家",
    "隐约察觉《妮可的冒险谭》作者真实身份，有意不深究"
  ]
}}

筛选标准：
1. 优先保留成长起点、启蒙、资格取得、拜师修行、重大试炼、重要约定、出发动机、关键转折。
2. 只保留对角色整体背景有代表性的经历。
3. 不要写外貌、称号、装备、口头习惯、一般爱好。
4. 不要写零碎趣事、轻微抱怨、日常插曲。
5. 不要写纯性格概括、纯评价、纯总结。
6. 不要编造资料中没有的新经历。
7. 不要输出“在各地旅行并以魔法谋生”这类泛泛职业概括，除非原始资料明确把它作为关键背景节点来写。
8. 优先直接摘取或轻微裁剪原始资料中的事件表述，不要把多条信息压成新的总结句。
9. 如果某条不能在原始资料中找到明确支撑，就不要输出。
10. 输出 0 到 5 条即可，宁缺毋滥。
11. 只输出 JSON。

角色背景资料：
{profile_block}

原始资料：
{source_text[:3200]}
""".strip()


def build_key_experience_selection_prompt(profile_lines: list[str], candidates: list[str]) -> str:
    profile_block = _profile_block(profile_lines)
    candidate_block = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(candidates)) if candidates else "（无）"
    return f"""
你要从候选条目中选出适合放进 `00_BACKGROUND.key_experiences` 的编号。

这些候选都来自原始资料原句或原句切出的事件分句，所以你的任务不是改写，而是挑选真正合格的背景关键经历。

合格标准：
1. 应该是成长起点、启蒙、资格取得、拜师修行、重大试炼、重要约定、出发动机、关键转折。
2. 应该对角色整体背景有代表性，而不是一般状态或职业概括。
3. 不要选外貌、称号、装备、一般爱好、口头习惯。
4. 不要选零碎趣事、轻微抱怨、日常插曲。
5. 不要选纯性格评价、纯抽象总结。
6. 最多选 5 条，宁缺毋滥。
7. 只输出 JSON。

角色背景资料：
{profile_block}

候选条目：
{candidate_block}
""".strip()


def build_keyword_selection_prompt(evidence_text: str, candidate_text: str) -> str:
    return f"""
你要把一组“角色标签候选”整理成适合给用户勾选的展示标签。

要求：
1. 只保留符合 `display_keywords` 目标的人设标签。
2. 删除底层规则描述、技术说明、动作过程、分析残片。
3. 保留高层、稳定、可展示的人物印象标签。
4. 优先选择简洁、常见、可勾选的标准标签，不要制造装饰性复合词。
5. 如果“礼貌外壳 / 腹黑算计 / 务实商人”这类词只是把基础标签包装得更花哨，就优先保留“礼貌 / 腹黑 / 务实”。
6. 如果候选里有过细、过碎、明显像内部规则的词，直接删掉，不要改写成近义技术词。
7. 不要新增脱离证据的新信息。
8. 输出 8 到 12 个标签即可，只输出 JSON。

证据：
{evidence_text}

候选：
{candidate_text}
""".strip()

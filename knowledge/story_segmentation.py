from __future__ import annotations

from typing import Any


STORY_SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "section_type": {"type": "string"},
        "story_units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "paragraph_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["title", "paragraph_ids"],
            },
        },
    },
    "required": ["section_type", "story_units"],
}


def build_story_section_prompt(section: dict) -> str:
    heading_path = " > ".join(section.get("heading_path", []) or []) or "无标题"
    paragraphs = list(section.get("paragraphs", []) or [])
    numbered = "\n\n".join(f"[{idx}] {paragraph}" for idx, paragraph in enumerate(paragraphs, start=1))
    return f"""
你是角色资料结构分析器。请判断下面这个章节里，哪些段落共同组成了可以独立叙述的完整故事、事件或经历。

判断标准：
1. “故事 / 事件 / 经历”指具体发生过的一段过程，通常包含情境、经过、结果、变化中的至少一部分。
2. 纯基础档案、外貌介绍、设定说明、抽象性格总结、喜好列表、作品外评价，不算故事。
3. 如果同一章节里包含多个不同事件，可以拆成多个 `story_unit`。
4. 但如果这些段落其实属于同一个完整事件，请优先合并成一个更完整的 `story_unit`，而不是拆成多个零碎片段。
5. 每个 `story_unit` 都应该尽量自洽、完整、可单独复述；如果缺少前后文就会讲不完整，就应该把相关相邻段落一起纳入。
6. `paragraph_ids` 应尽量连续，且必须复用输入里的原始段落编号，不要改写内容，不要创造新段落。
7. 如果这一节根本没有完整故事，就返回空数组。

输出 JSON：
- `section_type`: `story_section` / `mixed_section` / `non_story`
- `story_units`: 数组，每项包含：
  - `title`: 这个故事单元的简短标题
  - `paragraph_ids`: 属于这个故事单元的段落编号数组

章节路径：{heading_path}

段落列表：
{numbered}
""".strip()


def build_story_repair_prompt(section: dict, units: list[dict[str, Any]]) -> str:
    paragraphs = list(section.get("paragraphs", []) or [])
    numbered = "\n\n".join(f"[{idx}] {paragraph}" for idx, paragraph in enumerate(paragraphs, start=1))
    current_units = "\n".join(
        f"- {str(unit.get('title', '') or '').strip() or '故事片段'}: {list(unit.get('paragraph_ids', []) or [])}"
        for unit in units
    )
    return f"""
你要检查一组已经识别出的故事单元是否足够完整。

目标：
1. 如果某个故事单元还缺少能让它单独讲完整的前置或后续段落，请把必要的相邻段落补进去。
2. 如果两个单元实际上属于同一个连续事件，请合并成一个更完整的单元。
3. 如果某个单元明显只是零碎片段，不足以单独复述，就优先并入最接近的完整单元。
4. 不能创造新段落，只能使用已有段落编号。
5. 最终每个单元都应该尽量成为“可以单独讲完的一段经历”。
6. 只输出 JSON。

段落列表：
{numbered}

当前单元：
{current_units}
""".strip()


def normalize_story_units(section: dict, raw_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paragraphs = list(section.get("paragraphs", []) or [])
    max_index = len(paragraphs)
    normalized_units: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for unit in list(raw_units or []):
        ids: list[int] = []
        for raw_id in list((unit or {}).get("paragraph_ids", []) or []):
            try:
                value = int(raw_id)
            except Exception:
                continue
            if 1 <= value <= max_index and value not in ids:
                ids.append(value)
        ids.sort()
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        title = str((unit or {}).get("title", "") or "").strip() or (section.get("title") or "故事片段")
        normalized_units.append({"title": title, "paragraph_ids": ids})
    return normalized_units


def segment_story_section(model, section: dict, timeout: int) -> dict[str, Any]:
    paragraphs = list(section.get("paragraphs", []) or [])
    if not paragraphs:
        return {"section_type": "non_story", "story_units": []}

    try:
        payload = model.generate(
            build_story_section_prompt(section),
            return_json=True,
            schema=STORY_SECTION_SCHEMA,
            temperature=0.0,
            max_tokens=800,
            timeout=timeout,
        )
    except Exception:
        return {"section_type": "non_story", "story_units": []}

    section_type = str((payload or {}).get("section_type", "non_story") or "non_story").strip() or "non_story"
    story_units = normalize_story_units(section, list((payload or {}).get("story_units", []) or []))
    if not story_units:
        return {"section_type": section_type, "story_units": []}

    try:
        repaired = model.generate(
            build_story_repair_prompt(section, story_units),
            return_json=True,
            schema=STORY_SECTION_SCHEMA,
            temperature=0.0,
            max_tokens=420,
            timeout=timeout,
        )
        repaired_units = normalize_story_units(section, list((repaired or {}).get("story_units", []) or []))
        if repaired_units:
            story_units = repaired_units
    except Exception:
        pass

    return {"section_type": section_type, "story_units": story_units}

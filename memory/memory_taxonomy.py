from __future__ import annotations

import re


MEMORY_TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "memory_type": {"type": "string"},
        "topic_room": {"type": "string"},
    },
    "required": ["memory_type", "topic_room"],
}


def normalize_topic_tags(tags: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for raw in list(tags or []):
        value = re.sub(r"\s+", " ", str(raw or "")).strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def build_verbatim_excerpt(user_text: str = "", assistant_text: str = "") -> str:
    parts: list[str] = []
    user_value = str(user_text or "").strip()
    assistant_value = str(assistant_text or "").strip()
    if user_value:
        parts.append(f"User: {user_value}")
    if assistant_value:
        parts.append(f"Assistant: {assistant_value}")
    return "\n".join(parts).strip()


def infer_memory_type(
    summary: str,
    user_text: str = "",
    assistant_text: str = "",
    topic_tags: list[str] | None = None,
    llm=None,
) -> str:
    return classify_memory_taxonomy(
        summary=summary,
        user_text=user_text,
        assistant_text=assistant_text,
        topic_tags=topic_tags,
        llm=llm,
    )["memory_type"]


def infer_topic_room(
    summary: str,
    user_text: str = "",
    assistant_text: str = "",
    topic_tags: list[str] | None = None,
    llm=None,
) -> str:
    return classify_memory_taxonomy(
        summary=summary,
        user_text=user_text,
        assistant_text=assistant_text,
        topic_tags=topic_tags,
        llm=llm,
    )["topic_room"]


def infer_recall_filters(query: str, llm=None) -> dict[str, str]:
    taxonomy = classify_memory_taxonomy(summary=query, topic_tags=str(query or "").split(), llm=llm)
    filters: dict[str, str] = {}
    memory_type = str(taxonomy.get("memory_type", "") or "").strip().lower()
    topic_room = str(taxonomy.get("topic_room", "") or "").strip().lower()
    if memory_type:
        filters["memory_type"] = memory_type
    if topic_room:
        filters["topic_room"] = topic_room
    return filters


def classify_memory_taxonomy(
    summary: str,
    user_text: str = "",
    assistant_text: str = "",
    topic_tags: list[str] | None = None,
    llm=None,
) -> dict[str, str]:
    normalized_tags = normalize_topic_tags(topic_tags)
    if llm is None:
        return {"memory_type": "", "topic_room": ""}

    prompt = f"""
你是一个记忆分类器。你的任务是判断一条记忆或查询应该落入什么记忆类型与主题房间。

要求：
1. 只根据输入内容判断，不要使用硬编码规则或外部假设。
2. `memory_type` 使用短英文 slug，例如 `preference`、`emotion`、`relationship`、`event`、`dialogue`、`external`。
3. `topic_room` 使用短英文 slug，例如 `sleep`、`school`、`food`、`weather`、`daily_chat`，如果主题不明确就返回 `general`。
4. 输出必须是 JSON。
5. 如果不确定，`memory_type` 返回最宽泛但仍合理的类型，`topic_room` 返回 `general`。

摘要：
{str(summary or "").strip() or "无"}

用户原话：
{str(user_text or "").strip() or "无"}

角色回复：
{str(assistant_text or "").strip() or "无"}

已有标签：
{", ".join(normalized_tags) if normalized_tags else "无"}
""".strip()
    try:
        payload = llm.generate(
            prompt,
            return_json=True,
            schema=MEMORY_TAXONOMY_SCHEMA,
            temperature=0.0,
            max_tokens=120,
        )
    except Exception:
        return {"memory_type": "", "topic_room": ""}

    memory_type = _normalize_slug((payload or {}).get("memory_type"))
    topic_room = _normalize_slug((payload or {}).get("topic_room"))
    return {
        "memory_type": memory_type,
        "topic_room": topic_room,
    }


def _normalize_slug(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9_\- ]+", "", text)
    text = re.sub(r"[\s\-]+", "_", text)
    return text.strip("_")

from __future__ import annotations

from prompting.prompt_models import PromptSections


class PromptComposer:
    def compose(self, sections: PromptSections, *, fallback: bool = False) -> str:
        stable_rules = "\n".join(f"- {item}" for item in sections.stable.base_rules)
        plan_rules = "\n".join(f"- {item}" for item in sections.turn_plan.constraints)
        fallback_rule = (
            "当前关键证据不足。你要诚实承认边界，但仍保持角色口吻，不要编造。"
            if fallback
            else "如果证据足够，就自然完成回答；如果不够，只承认证据不足。"
        )
        context = sections.turn_context
        plan = sections.turn_plan
        evidence_text = plan.evidence_text.strip() if plan.evidence_text.strip() else "None"
        selected_view = dict((context.metadata or {}).get("selected_context_view", {}) or {})
        stable_view = dict(selected_view.get("stable", {}) or {})
        session_view = dict(selected_view.get("session", {}) or {})
        turn_view = dict(selected_view.get("turn", {}) or {})
        context_view_block = self._context_view_block(stable_view, session_view, turn_view)
        return f"""
角色名：
{sections.stable.character_name}

稳定角色底色：
{sections.stable.style_prompt}

长期规则：
{stable_rules}

本轮任务：
{plan.task_label}

动态状态：
- 当前情绪：{context.dynamic_state.mood}
- 情绪描述：{context.dynamic_state.mood_detail}
- 关系状态：{context.dynamic_state.relation_state}
- 用户情绪提示：{context.dynamic_state.user_emotion_hint}

最近对话：
{context.dynamic_state.recent_dialogue}

工作记忆：
{context.dynamic_state.memory_snapshot}

会话上下文：
{context.dynamic_state.session_context or "None"}

上下文视图：
{context_view_block}

L0 Identity：
{context.evidence.l0_identity or "None"}

人设证据：
{context.evidence.persona or "None"}

故事证据：
{context.evidence.story or "None"}

外部证据：
{context.evidence.external or "None"}

本轮主要证据类型：{plan.evidence_kind}
本轮主要证据：
{evidence_text}

用户这句话：
{context.user_input}

本轮规则：
{plan_rules}
- {fallback_rule}
""".strip()

    def _context_view_block(self, stable_view: dict, session_view: dict, turn_view: dict) -> str:
        blocks: list[str] = []
        stable_lines: list[str] = []
        if str(stable_view.get("identity", "") or "").strip():
            stable_lines.append("identity: " + str(stable_view.get("identity", "")).strip())
        if str(stable_view.get("style", "") or "").strip():
            stable_lines.append("style: " + str(stable_view.get("style", "")).strip())
        if list(stable_view.get("rules", []) or []):
            stable_lines.append("rules: " + " / ".join(str(item or "").strip() for item in list(stable_view.get("rules", []) or []) if str(item or "").strip()))
        if stable_lines:
            blocks.append("[stable]\n" + "\n".join(stable_lines))

        session_lines: list[str] = []
        topics = [str(item or "").strip() for item in list(session_view.get("topics", []) or []) if str(item or "").strip()]
        threads = [str(item or "").strip() for item in list(session_view.get("threads", []) or []) if str(item or "").strip()]
        archived_threads = [str(item or "").strip() for item in list(session_view.get("archived_threads", []) or []) if str(item or "").strip()]
        pinned = [str(item or "").strip() for item in list(session_view.get("pinned_facts", []) or []) if str(item or "").strip()]
        if topics:
            session_lines.append("topics: " + " / ".join(topics))
        if threads:
            session_lines.append("threads: " + " / ".join(threads))
        if archived_threads:
            session_lines.append("archived: " + " / ".join(archived_threads))
        if str(session_view.get("relation_summary", "") or "").strip():
            session_lines.append("relation: " + str(session_view.get("relation_summary", "")).strip())
        if str(session_view.get("emotion_summary", "") or "").strip():
            session_lines.append("emotion: " + str(session_view.get("emotion_summary", "")).strip())
        if pinned:
            session_lines.append("pinned: " + " / ".join(pinned))
        if session_lines:
            blocks.append("[session]\n" + "\n".join(session_lines))

        turn_lines: list[str] = []
        if str(turn_view.get("response_mode", "") or "").strip():
            turn_lines.append("mode: " + str(turn_view.get("response_mode", "")).strip())
        if list(turn_view.get("evidence_sources", []) or []):
            turn_lines.append("sources: " + " / ".join(str(item or "").strip() for item in list(turn_view.get("evidence_sources", []) or []) if str(item or "").strip()))
        if str(turn_view.get("evidence_preview", "") or "").strip():
            turn_lines.append("evidence: " + str(turn_view.get("evidence_preview", "")).strip())
        if str(turn_view.get("recent_dialogue", "") or "").strip():
            turn_lines.append("recent: " + str(turn_view.get("recent_dialogue", "")).strip())
        if turn_lines:
            blocks.append("[turn]\n" + "\n".join(turn_lines))

        return "\n\n".join(blocks).strip() or "None"

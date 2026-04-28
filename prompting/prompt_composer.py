from __future__ import annotations

from prompting.prompt_models import PromptAssembly, PromptSections


class PromptComposer:
    def __init__(self) -> None:
        self._reactive_compact_failures = 0

    def compose(self, sections: PromptSections, *, fallback: bool = False) -> PromptAssembly:
        stable_rules = "\n".join(f"- {item}" for item in sections.stable.base_rules)
        plan_rules = "\n".join(f"- {item}" for item in sections.turn_plan.constraints)
        fallback_rule = (
            "当前关键证据不足。你要诚实承认边界，但仍保持角色口吻，不要编造。"
            if fallback
            else "如果证据足够，就自然完成回答；如果不够，只承认证据不足。"
        )
        context = sections.turn_context
        plan = sections.turn_plan
        report: dict[str, object] = {"levels": [], "breaker_open": False}
        evidence_text = self._bounded_text(plan.evidence_text.strip() if plan.evidence_text.strip() else "None", 1800)
        selected_view = dict((context.metadata or {}).get("selected_context_view", {}) or {})
        stable_view = dict(selected_view.get("stable", {}) or {})
        session_view = dict(selected_view.get("session", {}) or {})
        turn_view = dict(selected_view.get("turn", {}) or {})
        recent_dialogue = self._snip_block(context.dynamic_state.recent_dialogue, limit=1100, report=report)
        hot_memory_index = self._bounded_text(context.dynamic_state.hot_memory_index or context.dynamic_state.memory_snapshot or "None", 1100)
        warm_memory_context = self._microcompact_block(context.dynamic_state.warm_memory_context or "None", limit=1200, report=report)
        cold_memory_hint = self._bounded_text(context.dynamic_state.cold_memory_hint or "None", 320)
        session_context = self._collapse_block(context.dynamic_state.session_context or "None", limit=1400, report=report)
        context_view_block = self._context_view_block(stable_view, session_view, turn_view, limit=1400, report=report)
        identity_block = self._bounded_text(context.evidence.l0_identity or "None", 900)
        persona_block = self._microcompact_block(context.evidence.persona or "None", limit=1600, report=report)
        story_block = self._microcompact_block(context.evidence.story or "None", limit=1600, report=report)
        external_block = self._microcompact_block(context.evidence.external or "None", limit=1400, report=report)
        system_prompt = f"""
角色名：
{sections.stable.character_name}

稳定角色底色：
{sections.stable.static_persona_prompt}

长期规则：
{stable_rules}

{sections.stable.dynamic_boundary_marker}
""".strip()
        dynamic_prompt = f"""
本轮任务：
{plan.task_label}

动态状态：
- 当前情绪：{context.dynamic_state.mood}
- 情绪描述：{context.dynamic_state.mood_detail}
- 关系状态：{context.dynamic_state.relation_state}
- 用户情绪提示：{context.dynamic_state.user_emotion_hint}

最近对话：
{recent_dialogue}

热层记忆索引：
{hot_memory_index}

温层相关记忆：
{warm_memory_context}

冷层历史提示：
{cold_memory_hint}

会话上下文：
{session_context or "None"}

上下文视图：
{context_view_block}

L0 Identity：
{identity_block}

人设证据：
{persona_block}

故事证据：
{story_block}

外部证据：
{external_block}

本轮主要证据类型：{plan.evidence_kind}
本轮主要证据：
{evidence_text}

用户这句话：
{context.user_input}

本轮规则：
{plan_rules}
- {fallback_rule}
""".strip()
        dynamic_prompt = self._autocompact(dynamic_prompt, report=report)
        return PromptAssembly(
            system_prompt=system_prompt,
            dynamic_prompt=dynamic_prompt,
            cache_key=sections.stable.cache_key,
            boundary_marker=sections.stable.dynamic_boundary_marker,
            compaction_report=report,
        )

    def _context_view_block(self, stable_view: dict, session_view: dict, turn_view: dict, *, limit: int, report: dict[str, object]) -> str:
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

        joined = "\n\n".join(blocks).strip() or "None"
        return self._collapse_block(joined, limit=limit, report=report)

    def _bounded_text(self, text: str, limit: int) -> str:
        value = str(text or "").strip() or "None"
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "\n[提示] 内容已截断。"

    def _snip_block(self, text: str, *, limit: int, report: dict[str, object]) -> str:
        value = str(text or "").strip() or "None"
        if len(value) <= limit:
            return value
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        compact = "\n".join(lines[-6:]).strip() or value[:limit].rstrip()
        compact = compact if len(compact) <= limit else compact[:limit].rstrip()
        report["levels"] = [*list(report.get("levels", []) or []), "snip"]
        return compact + "\n[提示] 较早的最近对话已轻量裁剪。"

    def _microcompact_block(self, text: str, *, limit: int, report: dict[str, object]) -> str:
        value = str(text or "").strip() or "None"
        if len(value) <= limit:
            return value
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        head = lines[:6]
        tail = lines[-3:] if len(lines) > 9 else []
        merged = head + (["[中间若干细节已折叠]"] if tail else []) + tail
        compact = "\n".join(merged).strip() or value[:limit].rstrip()
        if len(compact) > limit:
            compact = compact[:limit].rstrip()
        report["levels"] = [*list(report.get("levels", []) or []), "microcompact"]
        return compact

    def _collapse_block(self, text: str, *, limit: int, report: dict[str, object]) -> str:
        value = str(text or "").strip() or "None"
        if len(value) <= limit:
            return value
        raw_lines = [line.strip(" -") for line in value.splitlines() if line.strip()]
        compact_lines: list[str] = []
        for line in raw_lines[:8]:
            if ":" in line:
                key, rest = line.split(":", 1)
                compact_lines.append(f"{key.strip()}: {rest.strip()[:120].rstrip()}")
            else:
                compact_lines.append(line[:120].rstrip())
        compact = "\n".join(compact_lines).strip() or value[:limit].rstrip()
        if len(compact) > limit:
            compact = compact[:limit].rstrip()
        report["levels"] = [*list(report.get("levels", []) or []), "context_collapse"]
        return compact + "\n[提示] 会话上下文已折叠。"

    def _autocompact(self, dynamic_prompt: str, *, report: dict[str, object]) -> str:
        soft_limit = 5600
        hard_limit = 7200
        text = str(dynamic_prompt or "").strip()
        if len(text) <= soft_limit:
            self._reactive_compact_failures = 0
            return text

        report["levels"] = [*list(report.get("levels", []) or []), "autocompact"]
        lines = [line for line in text.splitlines() if line.strip()]
        condensed = "\n".join(line[:220].rstrip() for line in lines[:42]).strip()
        if len(condensed) <= soft_limit:
            self._reactive_compact_failures = 0
            return condensed + "\n[提示] 本轮上下文已自动压缩。"

        if self._reactive_compact_failures >= 3:
            report["breaker_open"] = True
            return condensed[:hard_limit].rstrip() + "\n[提示] 应急压缩熔断已开启，仅保留核心上下文。"

        report["levels"] = [*list(report.get("levels", []) or []), "reactive_compact"]
        reactive = "\n".join(lines[:28]).strip()
        overflow = len(reactive) > hard_limit
        reactive = reactive[:hard_limit].rstrip()
        if overflow:
            self._reactive_compact_failures += 1
            report["reactive_failures"] = self._reactive_compact_failures
        else:
            self._reactive_compact_failures = 0
        return reactive + "\n[提示] 已触发应急压缩。"

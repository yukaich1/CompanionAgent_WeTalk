from __future__ import annotations

import uuid
from datetime import datetime

from knowledge.persona_shared import DISPLAY_KEYWORD_LIMIT, dedupe
from llm import FallbackMistralLLM
from persona_models import PersonaPreviewModel, PersonaSourceSnippet
from tools import DEFAULT_TOOL_REGISTRY


class PersonaPreviewService:
    def __init__(self, system):
        self.system = system
        self.model = FallbackMistralLLM()

    def build_persona_search_queries(self, persona_name: str, work_title: str = "") -> list[str]:
        anchor = " ".join(part for part in [work_title, persona_name] if part).strip() or persona_name.strip()
        if not anchor:
            return []
        prompt = (
            "你要为一个角色资料补充任务生成搜索查询词。\n"
            "目标是查到这个角色本人的人物资料、说话方式、经历和角色介绍。\n"
            "请输出 2 到 4 条简洁搜索词，每条一行，不要编号，不要解释。\n"
            f"角色名：{persona_name or '未知'}\n"
            f"作品名：{work_title or '未知'}\n"
            f"锚点：{anchor}"
        )
        try:
            result = str(self.model.generate(prompt, temperature=0.1, max_tokens=120) or "").strip()
            queries = [line.strip("- ").strip() for line in result.splitlines() if line.strip()]
        except Exception:
            queries = [anchor]
        return dedupe([query for query in queries if query], limit=4) or [anchor]

    def normalize_search_text(self, text: str) -> str:
        return "".join(str(text or "").lower().split())

    def is_relevant_web_snippet(self, item: dict, persona_name: str, work_title: str = "") -> bool:
        combined = self.normalize_search_text(f"{item.get('title', '')} {item.get('text', '')}")
        persona_key = self.normalize_search_text(persona_name)
        work_key = self.normalize_search_text(work_title)
        if persona_key and persona_key not in combined:
            return False
        if work_key and work_key not in combined:
            return False
        return True

    def collect_web_snippets(self, persona_name: str, work_title: str = "", local_canon: str = "", max_results: int = 5) -> list:
        query_candidates = self.build_persona_search_queries(persona_name, work_title)
        snippets = []
        seen_keys = set()
        seen_texts = set()
        for query in query_candidates:
            try:
                search_result = DEFAULT_TOOL_REGISTRY.run(
                    "web_search",
                    persona_name=persona_name,
                    query=query,
                    max_results=max_results,
                    timeout=10,
                    source_mode="persona_ordered",
                )
            except Exception:
                continue
            for item in search_result.get("snippets", []):
                text = (item.get("text", "") or "").strip()
                if not text or not self.is_relevant_web_snippet(item, persona_name, work_title):
                    continue
                decision = self.system.conflict_filter.should_keep(
                    text=text,
                    url=item.get("url", item.get("source", "")),
                    user_canon=local_canon,
                )
                if not decision.kept:
                    continue
                canonical_text = self.system._canonicalize_source_text(text)
                if canonical_text in seen_texts:
                    continue
                key = (item.get("source", "web"), item.get("title", ""), text[:120])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                seen_texts.add(canonical_text)
                snippets.append(PersonaSourceSnippet(source=item.get("source", "web"), title=item.get("title", ""), text=text))
            if len(snippets) >= max_results:
                break
        return snippets[:max_results]

    def summarize_web_snippets(self, persona_name: str, work_title: str, snippets: list) -> str:
        if not snippets:
            return ""
        reference_text = "\n\n".join(f"[{snippet.source} | {snippet.title}]\n{snippet.text}" for snippet in snippets[:6])[:4800]
        prompt = (
            "请根据以下公开资料，只总结与当前角色本人直接相关的人设补充信息。\n"
            "重点只保留：人物特点、说话方式、价值观、经历过的关键故事。\n"
            "不要写作品泛设定，不要写无关角色，不要写观众评价，不要编造。\n"
            "如果资料涉及说话方式，请优先总结角色说话的节奏、语气、措辞习惯和情感距离感。\n"
            "如果资料涉及故事或经历，只保留和当前角色本人直接有关、可在对话中自然提及的经历。\n"
            f"角色名：{persona_name}\n作品名：{work_title or '未知'}\n\n资料：\n{reference_text}\n\n"
            "请用简洁中文输出 1 到 4 段总结。"
        )
        try:
            result = self.system.model.generate(prompt, temperature=0.1, max_tokens=600)
            return str(result or "").strip()
        except Exception:
            return ""

    def preview_from_sources(
        self,
        persona_name: str = "",
        work_title: str = "",
        local_text: str = "",
        local_label: str = "manual_input",
        max_results: int = 5,
        local_snippets: list | None = None,
        enable_web_search: bool = True,
    ) -> PersonaPreviewModel:
        persona_name = (persona_name or self.system.persona_name or "").strip()
        work_title = (work_title or "").strip()
        local_text = (local_text or "").strip()
        local_label = (local_label or "manual_input").strip() or "manual_input"
        local_snippets = list(local_snippets or [])
        if not persona_name and not local_text and not local_snippets:
            raise ValueError("角色名或本地资料至少需要提供一项。")

        snippets: list[PersonaSourceSnippet] = []
        if local_snippets:
            for item in local_snippets:
                text = (item.get("text") or "").strip()
                if text:
                    snippets.append(
                        PersonaSourceSnippet(
                            source="local",
                            title=(item.get("title") or local_label).strip() or local_label,
                            text=text,
                        )
                    )
        elif local_text:
            snippets.append(PersonaSourceSnippet(source="local", title=local_label, text=local_text))

        local_canon = "\n\n".join(snippet.text for snippet in snippets if snippet.source == "local").strip()
        if persona_name and enable_web_search:
            web_snippets = self.collect_web_snippets(persona_name, work_title=work_title, local_canon=local_canon, max_results=max_results)
            web_summary = self.summarize_web_snippets(persona_name, work_title, web_snippets)
            if web_summary:
                snippets.append(PersonaSourceSnippet(source="web_summary", title="联网补充摘要", text=web_summary))

        if not snippets:
            raise ValueError("没有找到可用于预览的人设资料，请补充作品名或上传本地资料。")

        has_local = any(snippet.source == "local" for snippet in snippets)
        has_web = any(snippet.source != "local" for snippet in snippets)
        mode = "hybrid" if has_local and has_web else ("local_only" if has_local else "cold_start")
        source_label = mode
        if persona_name:
            source_label += f":{persona_name}"
        if work_title:
            source_label += f":{work_title}"

        source_text = "\n\n".join(f"[{snippet.source} | {snippet.title}]\n{snippet.text}" for snippet in snippets)
        summary = self.system._summarize_with_llm(source_text, source_label) or {}
        summary = self.system._summary_to_dict(summary)
        summary["display_keywords"] = self.system.ingest_service.selected_keyword_candidates(summary)

        preview = PersonaPreviewModel(
            preview_id=str(uuid.uuid4()),
            persona_name=persona_name or self.system.persona_name,
            work_title=work_title,
            source_label=source_label,
            source_text=source_text,
            base_template_text=self.system._base_template_text(summary),
            snippets=snippets,
            summary=summary,
            created_at=datetime.now().isoformat(timespec="seconds"),
            mode=mode,
            committed=False,
        )
        self.system.pending_previews[preview.preview_id] = preview.dict()
        return preview

    def commit_summary_only(self, source_label: str, summary: dict):
        if not summary:
            return
        self.system._merge_summary_data(summary)
        self.system._dedupe_storage()

    def confirm_preview(self, preview_id: str, selected_keywords: list[str] | None = None, progress_callback=None) -> dict:
        payload = self.system.pending_previews.get(preview_id)
        if not payload:
            raise KeyError(f"Unknown preview_id: {preview_id}")
        preview = PersonaPreviewModel(**payload)

        summary_dict = preview.summary if isinstance(preview.summary, dict) else self.system._summary_to_dict(preview.summary)
        if not summary_dict.get("display_keywords"):
            summary_dict["display_keywords"] = self.system.ingest_service.selected_keyword_candidates(summary_dict)
        if selected_keywords is not None:
            normalized_keywords = dedupe(
                [
                    self.system._normalize_display_keyword(value, max_chars=16)
                    for value in selected_keywords
                    if self.system._normalize_display_keyword(value, max_chars=16)
                ],
                limit=min(DISPLAY_KEYWORD_LIMIT, 8),
            )
            summary_dict["display_keywords"] = normalized_keywords
        count = 0
        applied_summary = False
        for snippet in preview.snippets:
            source_label = snippet.title or preview.source_label
            if snippet.source == "local":
                count += self.system._commit_summary_and_entries(
                    snippet.text,
                    source_label,
                    summary_dict if not applied_summary else None,
                    progress_callback=progress_callback,
                )
                applied_summary = True
            elif snippet.source == "web_summary":
                count += self.system.ingest_service.commit_entries_only(
                    snippet.text,
                    f"{preview.source_label}#web_summary",
                    kind="source_chunk",
                    priority=0.72,
                    progress_callback=progress_callback,
                )
        if not applied_summary:
            self.commit_summary_only(preview.source_label, summary_dict)

        self.system.display_keywords = list(summary_dict.get("display_keywords", []) or [])
        self.system.persona_name = preview.persona_name or self.system.persona_name
        committed_preview = preview.copy(update={"committed": True})
        self.system.pending_previews[preview.preview_id] = committed_preview.dict()
        return {
            "count": count,
            "preview": committed_preview,
            "keywords": dedupe(summary_dict.get("display_keywords", []), limit=min(DISPLAY_KEYWORD_LIMIT, 8)),
        }

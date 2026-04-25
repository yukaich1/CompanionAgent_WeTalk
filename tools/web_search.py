from __future__ import annotations

import html
import re
from urllib.parse import quote

import requests

from tools.base import AgentTool, ToolSpec


WIKI_SOURCES = [
    "zh.wikipedia.org",
    "ja.wikipedia.org",
    "en.wikipedia.org",
]

PERSONA_SITE_SOURCES = [
    "dic.pixiv.net",
    "zh.moegirl.org.cn",
]

DEFAULT_PERSONA_SOURCE_ORDER = [*PERSONA_SITE_SOURCES, *WIKI_SOURCES]
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WitchTalk/1.0"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"


def _clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe_snippets(items: list[dict], limit: int) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (_clean_text(item.get("title", "")), _clean_text(item.get("text", "")))
        if key in seen or not key[0] or not key[1]:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _ddg_search(query: str, max_results: int, timeout: int) -> list[dict]:
    try:
        response = requests.post(
            DUCKDUCKGO_HTML_URL,
            data={"q": query},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception:
        return []

    page = response.text
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.S,
    )

    results: list[dict] = []
    for match in pattern.finditer(page):
        title = _clean_text(match.group("title"))
        snippet = _clean_text(match.group("snippet"))
        href = html.unescape(match.group("href"))
        if not title or not snippet:
            continue
        results.append(
            {
                "source": "duckduckgo",
                "title": title,
                "text": snippet[:800],
                "url": href,
            }
        )
        if len(results) >= max_results:
            break
    return _dedupe_snippets(results, max_results)


def _site_ddg_search(domain: str, query: str, max_results: int, timeout: int) -> list[dict]:
    scoped_query = f"site:{domain} {query}".strip()
    results = _ddg_search(scoped_query, max_results=max_results, timeout=timeout)
    filtered = [
        {
            "source": domain,
            "title": item.get("title", ""),
            "text": item.get("text", ""),
            "url": item.get("url", ""),
        }
        for item in results
        if domain in str(item.get("url", "")).lower()
    ]
    return _dedupe_snippets(filtered, max_results)


def _wiki_search(search_terms: list[str], max_results: int, timeout: int, domains: list[str] | None = None) -> list[dict]:
    snippets: list[dict] = []
    seen_titles: set[tuple[str, str]] = set()
    for wiki in list(domains or WIKI_SOURCES):
        for term in search_terms:
            try:
                search_url = (
                    f"https://{wiki}/w/api.php?action=query&list=search"
                    f"&srsearch={quote(term)}&format=json&utf8=1&srlimit={max_results}"
                )
                search_res = requests.get(search_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
                search_res.raise_for_status()
                search_data = search_res.json()
            except Exception:
                continue

            for item in search_data.get("query", {}).get("search", []):
                title = item.get("title", "").strip()
                key = (wiki, title)
                if not title or key in seen_titles:
                    continue
                seen_titles.add(key)
                try:
                    detail_url = (
                        f"https://{wiki}/w/api.php?action=query&prop=extracts"
                        f"&explaintext=1&exintro=1&redirects=1&format=json&titles={quote(title)}"
                    )
                    detail_res = requests.get(detail_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
                    detail_res.raise_for_status()
                    detail_data = detail_res.json()
                    for page in (detail_data.get("query", {}).get("pages", {}) or {}).values():
                        extract = str(page.get("extract", "") or "").strip()
                        if extract:
                            snippets.append({"source": wiki, "title": title, "text": extract[:1200]})
                            break
                except Exception:
                    continue
                if len(snippets) >= max_results:
                    return _dedupe_snippets(snippets, max_results)
    return _dedupe_snippets(snippets, max_results)


class WebSearchTool(AgentTool):
    spec = ToolSpec(
        name="web_search",
        description="Search public web and wiki summaries for character or real-world reference material.",
        input_schema={
            "persona_name": "str",
            "query": "str",
            "max_results": "int",
            "timeout": "int",
            "source_mode": "str",
        },
        output_schema={"snippets": [{"source": "str", "title": "str", "text": "str"}]},
        tags=["reference", "persona", "search", "web"],
    )

    def _build_search_terms(self, persona_name: str, query: str) -> list[str]:
        terms: list[str] = []
        if persona_name:
            terms.append(persona_name)
        if persona_name and query:
            terms.append(f"{persona_name} {query}")
        if query and not persona_name:
            terms.append(query)
        return [term for term in terms if term]

    def _persona_search(self, search_terms: list[str], max_results: int, timeout: int) -> list[dict]:
        collected: list[dict] = []
        for domain in DEFAULT_PERSONA_SOURCE_ORDER:
            remaining = max_results - len(collected)
            if remaining <= 0:
                break
            if domain in WIKI_SOURCES:
                collected.extend(_wiki_search(search_terms, max_results=remaining, timeout=timeout, domains=[domain]))
            else:
                for term in search_terms:
                    collected.extend(_site_ddg_search(domain, term, max_results=remaining, timeout=timeout))
                    collected = _dedupe_snippets(collected, max_results)
                    if len(collected) >= max_results:
                        break
        return _dedupe_snippets(collected, max_results)

    def _general_search(self, search_terms: list[str], max_results: int, timeout: int) -> list[dict]:
        collected: list[dict] = []
        collected.extend(_wiki_search(search_terms, max_results=max_results, timeout=timeout))
        if len(collected) < max_results:
            for term in search_terms:
                remaining = max_results - len(collected)
                if remaining <= 0:
                    break
                collected.extend(_ddg_search(term, max_results=remaining, timeout=timeout))
                collected = _dedupe_snippets(collected, max_results)
        return _dedupe_snippets(collected, max_results)

    def run(self, persona_name, query="", max_results=3, timeout=8, source_mode="general"):
        if not persona_name and not query:
            return {"snippets": []}

        search_terms = self._build_search_terms(str(persona_name or "").strip(), str(query or "").strip())
        if not search_terms:
            return {"snippets": []}

        if source_mode == "persona_ordered":
            return {"snippets": self._persona_search(search_terms, max_results=max_results, timeout=timeout)}
        return {"snippets": self._general_search(search_terms, max_results=max_results, timeout=timeout)}


def fetch_character_reference_snippets(persona_name, query="", max_results=3, timeout=8):
    return WebSearchTool().run(persona_name=persona_name, query=query, max_results=max_results, timeout=timeout)["snippets"]

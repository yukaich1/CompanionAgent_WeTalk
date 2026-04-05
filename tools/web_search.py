import html
import re
from urllib.parse import quote

import requests

from tools.base import AgentTool, ToolSpec


PIXIV_DOMAIN = "dic.pixiv.net"
MOEGIRL_DOMAIN = "zh.moegirl.org.cn"
WIKI_DOMAINS = [
    "zh.wikipedia.org",
    "ja.wikipedia.org",
    "en.wikipedia.org",
]

SITE_SEARCH_DOMAINS = [
    PIXIV_DOMAIN,
    MOEGIRL_DOMAIN,
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Wetalk/1.0"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"


def _clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ddg_search(query: str, max_results: int, timeout: int):
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

    results = []
    seen = set()
    for match in pattern.finditer(page):
        title = _clean_text(match.group("title"))
        snippet = _clean_text(match.group("snippet"))
        href = html.unescape(match.group("href"))
        if not title or not snippet:
            continue
        key = (title, snippet)
        if key in seen:
            continue
        seen.add(key)
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
    return results


def _site_ddg_search(domain: str, query: str, max_results: int, timeout: int):
    scoped_query = f"site:{domain} {query}".strip()
    results = _ddg_search(scoped_query, max_results=max_results, timeout=timeout)
    filtered = []
    seen = set()
    for item in results:
        url = (item.get("url") or "").lower()
        if domain not in url:
            continue
        key = (_clean_text(item.get("title", "")), _clean_text(item.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(
            {
                "source": domain,
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "url": item.get("url", ""),
            }
        )
        if len(filtered) >= max_results:
            break
    return filtered


def _wiki_search(search_terms, max_results: int, timeout: int):
    snippets = []
    seen_titles = set()

    for wiki in WIKI_DOMAINS:
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
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                try:
                    detail_url = (
                        f"https://{wiki}/w/api.php?action=query&prop=extracts"
                        f"&explaintext=1&exintro=1&redirects=1&format=json&titles={quote(title)}"
                    )
                    detail_res = requests.get(detail_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
                    detail_res.raise_for_status()
                    detail_data = detail_res.json()
                    pages = detail_data.get("query", {}).get("pages", {})
                    for page in pages.values():
                        extract = (page.get("extract") or "").strip()
                        if extract:
                            snippets.append(
                                {
                                    "source": wiki,
                                    "title": title,
                                    "text": extract[:1200],
                                }
                            )
                            break
                except Exception:
                    continue
                if len(snippets) >= max_results:
                    return snippets
    return snippets


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
        output_schema={
            "snippets": [
                {
                    "source": "str",
                    "title": "str",
                    "text": "str",
                }
            ]
        },
        tags=["reference", "persona", "search", "web"],
    )

    def _ordered_persona_search(self, search_terms, max_results: int, timeout: int):
        deduped = []
        seen = set()

        for domain in (PIXIV_DOMAIN, MOEGIRL_DOMAIN):
            domain_hits = []
            for term in search_terms:
                for item in _site_ddg_search(domain, term, max_results=max_results, timeout=timeout):
                    key = (_clean_text(item.get("title", "")), _clean_text(item.get("text", "")))
                    if key in seen:
                        continue
                    seen.add(key)
                    domain_hits.append(item)
                    if len(domain_hits) >= max_results:
                        break
                if len(domain_hits) >= max_results:
                    break
            if domain_hits:
                deduped.extend(domain_hits[:max_results])
                return deduped[:max_results]

        wiki_hits = _wiki_search(search_terms, max_results=max_results, timeout=timeout)
        for item in wiki_hits:
            key = (_clean_text(item.get("title", "")), _clean_text(item.get("text", "")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_results:
                break
        return deduped[:max_results]

    def run(self, persona_name, query="", max_results=3, timeout=8, source_mode="general"):
        if not persona_name and not query:
            return {"snippets": []}

        search_terms = []
        if persona_name:
            search_terms.append(persona_name)
        if persona_name and query:
            search_terms.append(f"{persona_name} {query}")
        if query and not persona_name:
            search_terms.append(query)

        deduped = []
        seen = set()
        ddg_terms = []
        if persona_name:
            ddg_terms.append(persona_name)
        if persona_name and query:
            ddg_terms.append(f"{persona_name} {query}")
        if query and not persona_name:
            ddg_terms.append(query)
        ddg_terms = [term for term in ddg_terms if term]

        if source_mode == "persona_ordered":
            ordered_hits = self._ordered_persona_search(ddg_terms or search_terms, max_results=max_results, timeout=timeout)
            return {"snippets": ordered_hits[:max_results]}

        for term in ddg_terms:
            for domain in SITE_SEARCH_DOMAINS:
                more = _site_ddg_search(domain, term, max_results=max_results - len(deduped), timeout=timeout)
                for item in more:
                    key = (_clean_text(item.get("title", "")), _clean_text(item.get("text", "")))
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(item)
                    if len(deduped) >= max_results:
                        break
                if len(deduped) >= max_results:
                    break
            if len(deduped) >= max_results:
                break

        if len(deduped) < max_results:
            wiki_snippets = _wiki_search(search_terms, max_results=max_results - len(deduped), timeout=timeout)
            for item in wiki_snippets:
                key = (_clean_text(item.get("title", "")), _clean_text(item.get("text", "")))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)

        if len(deduped) < max_results:
            for term in ddg_terms:
                more = _ddg_search(term, max_results=max_results - len(deduped), timeout=timeout)
                for item in more:
                    key = (_clean_text(item.get("title", "")), _clean_text(item.get("text", "")))
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(item)
                    if len(deduped) >= max_results:
                        break
                if len(deduped) >= max_results:
                    break

        return {"snippets": deduped[:max_results]}


def fetch_character_reference_snippets(persona_name, query="", max_results=3, timeout=8):
    return WebSearchTool().run(persona_name=persona_name, query=query, max_results=max_results, timeout=timeout)["snippets"]

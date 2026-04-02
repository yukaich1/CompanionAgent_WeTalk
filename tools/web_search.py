from urllib.parse import quote

import requests

from tools.base import AgentTool, ToolSpec


WIKIS = [
    "zh.wikipedia.org",
    "ja.wikipedia.org",
    "en.wikipedia.org",
]


class WebSearchTool(AgentTool):
    spec = ToolSpec(
        name="web_search",
        description="Search public wiki summaries for character reference material.",
        input_schema={
            "persona_name": "str",
            "query": "str",
            "max_results": "int",
            "timeout": "int",
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
        tags=["reference", "persona", "search"],
    )

    def run(self, persona_name, query="", max_results=2, timeout=8):
        if not persona_name and not query:
            return {"snippets": []}

        snippets = []
        seen_titles = set()
        search_terms = []
        if persona_name and query:
            search_terms.append(f"{persona_name} {query}")
        if persona_name:
            search_terms.append(persona_name)
        if query:
            search_terms.append(query)

        for wiki in WIKIS:
            for term in search_terms:
                try:
                    search_url = (
                        f"https://{wiki}/w/api.php?action=query&list=search"
                        f"&srsearch={quote(term)}&format=json&utf8=1&srlimit={max_results}"
                    )
                    search_res = requests.get(search_url, timeout=timeout)
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
                        detail_res = requests.get(detail_url, timeout=timeout)
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
                        return {"snippets": snippets}
        return {"snippets": snippets}


def fetch_character_reference_snippets(persona_name, query="", max_results=2, timeout=8):
    return WebSearchTool().run(persona_name=persona_name, query=query, max_results=max_results, timeout=timeout)["snippets"]

import logging
import os

import httpx

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": os.getenv(
        "WIKI_USER_AGENT",
        "ResearchPilot/1.0 (+https://github.com/GustavoGMatias/researchpilot)",
    )
}


class WikipediaError(Exception):
    pass


def search_wikipedia(query: str, lang: str = "pt", limit: int = 3) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "exchars": 1200,
        "inprop": "url",
    }
    try:
        r = httpx.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params=params,
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except (httpx.HTTPError, ValueError) as e:
        log.warning("falha na Wikipedia: %s", e)
        raise WikipediaError("Wikipedia indisponível no momento") from e
    ordered = sorted(pages.values(), key=lambda p: p.get("index", 0))
    results = [
        {"title": p["title"], "url": p["fullurl"], "text": p.get("extract", "")}
        for p in ordered
    ]
    return {"results": results, "sources": [x["url"] for x in results]}

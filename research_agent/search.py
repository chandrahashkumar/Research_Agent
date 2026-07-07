"""
Literature search tools.

Supports two backends:
  1. arXiv  — free, no key required (via `arxiv` package)
  2. Semantic Scholar — free REST API, optional API key for higher rate limits
"""

from __future__ import annotations

import time
import requests
import arxiv

from research_agent.config import Config


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Paper:
    """Lightweight, backend-agnostic representation of a research paper."""

    def __init__(
        self,
        *,
        paper_id: str,
        title: str,
        authors: list[str],
        year: str | int,
        abstract: str,
        url: str,
        source: str,
        doi: str = "",
        venue: str = "",
    ) -> None:
        self.paper_id = paper_id
        self.title = title
        self.authors = authors
        self.year = str(year)
        self.abstract = abstract
        self.url = url
        self.source = source
        self.doi = doi
        self.venue = venue

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "url": self.url,
            "source": self.source,
            "doi": self.doi,
            "venue": self.venue,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Paper {self.year} — {self.title[:60]}>"


# ---------------------------------------------------------------------------
# arXiv backend
# ---------------------------------------------------------------------------

class ArxivSearcher:
    """Search arXiv using the official Python client."""

    MAX_RESULTS_CAP = 25

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        max_results = min(max_results, self.MAX_RESULTS_CAP)
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers: list[Paper] = []
        for result in client.results(search):
            papers.append(
                Paper(
                    paper_id=result.entry_id,
                    title=result.title,
                    authors=[str(a) for a in result.authors],
                    year=result.published.year if result.published else "",
                    abstract=result.summary,
                    url=result.entry_id,
                    source="arxiv",
                    doi=result.doi or "",
                    venue="arXiv",
                )
            )
        return papers


# ---------------------------------------------------------------------------
# Semantic Scholar backend
# ---------------------------------------------------------------------------

class SemanticScholarSearcher:
    """Search the Semantic Scholar Graph API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "paperId,title,authors,year,abstract,externalIds,venue,openAccessPdf"

    def __init__(self, api_key: str = "") -> None:
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["x-api-key"] = api_key

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": self.FIELDS,
        }
        resp = requests.get(
            f"{self.BASE_URL}/paper/search",
            params=params,
            headers=self._headers,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        papers: list[Paper] = []
        for item in data:
            ext_ids = item.get("externalIds") or {}
            pdf_info = item.get("openAccessPdf") or {}
            papers.append(
                Paper(
                    paper_id=item.get("paperId", ""),
                    title=item.get("title", ""),
                    authors=[a["name"] for a in item.get("authors", [])],
                    year=item.get("year", ""),
                    abstract=item.get("abstract", ""),
                    url=pdf_info.get("url", f"https://www.semanticscholar.org/paper/{item.get('paperId','')}"),
                    source="semantic_scholar",
                    doi=ext_ids.get("DOI", ""),
                    venue=item.get("venue", ""),
                )
            )
            time.sleep(0.05)  # gentle rate-limit
        return papers


# ---------------------------------------------------------------------------
# Unified searcher
# ---------------------------------------------------------------------------

class LiteratureSearcher:
    """
    Unified literature search interface.
    Queries both arXiv and Semantic Scholar, deduplicates by title, and returns
    a merged list sorted by relevance / recency.
    """

    def __init__(self, config: Config) -> None:
        self._arxiv = ArxivSearcher()
        self._ss = SemanticScholarSearcher(api_key=config.SEMANTIC_SCHOLAR_API_KEY)
        self._max = config.MAX_SEARCH_RESULTS

    def search(self, query: str, max_results: int | None = None) -> list[Paper]:
        limit = max_results or self._max
        half = max(limit // 2, 3)

        arxiv_papers: list[Paper] = []
        ss_papers: list[Paper] = []

        try:
            arxiv_papers = self._arxiv.search(query, max_results=half)
        except Exception as exc:
            print(f"[arXiv] search error: {exc}")

        try:
            ss_papers = self._ss.search(query, max_results=half)
        except Exception as exc:
            print(f"[Semantic Scholar] search error: {exc}")

        # Deduplicate by normalised title
        seen: set[str] = set()
        merged: list[Paper] = []
        for p in arxiv_papers + ss_papers:
            key = p.title.lower().strip()
            if key not in seen:
                seen.add(key)
                merged.append(p)
            if len(merged) >= limit:
                break

        return merged

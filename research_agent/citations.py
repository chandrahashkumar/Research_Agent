"""
Citation and reference manager.

Handles:
  - Building BibTeX entries from Paper objects
  - Maintaining an in-memory reference list
  - Exporting to .bib files
  - Formatting inline citations (APA-style)
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.search import Paper
    from research_agent.summarizer import PaperSummary


class CitationManager:
    """Manages citations and BibTeX exports for a research session."""

    def __init__(self) -> None:
        self._refs: dict[str, dict] = {}   # cite_key -> metadata
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Adding references
    # ------------------------------------------------------------------

    def add_paper(self, paper: "Paper") -> str:
        """
        Register *paper* and return its citation key (e.g. ``smith2023``).
        Idempotent — calling twice for the same paper returns the same key.
        """
        # Check if already registered by paper_id
        for key, meta in self._refs.items():
            if meta["paper_id"] == paper.paper_id:
                return key

        cite_key = self._make_cite_key(paper)
        self._refs[cite_key] = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "abstract": paper.abstract,
            "url": paper.url,
            "doi": paper.doi,
            "venue": paper.venue,
            "source": paper.source,
        }
        return cite_key

    def add_from_summary(self, summary: "PaperSummary") -> str:
        """Register a paper from its PaperSummary object."""
        for key, meta in self._refs.items():
            if meta["paper_id"] == summary.paper_id:
                return key

        cite_key = self._make_cite_key_from_parts(summary.authors, summary.year, summary.title)
        self._refs[cite_key] = {
            "paper_id": summary.paper_id,
            "title": summary.title,
            "authors": summary.authors,
            "year": summary.year,
            "abstract": summary.abstract,
            "url": summary.url,
            "doi": "",
            "venue": "",
            "source": "summary",
        }
        return cite_key

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def list_citations(self) -> list[dict]:
        """Return all registered citations as a list of metadata dicts."""
        return [{"cite_key": k, **v} for k, v in self._refs.items()]

    def format_apa(self, cite_key: str) -> str:
        """Return an APA-style formatted reference string."""
        meta = self._refs.get(cite_key)
        if not meta:
            return f"[{cite_key}: not found]"

        authors = meta["authors"]
        if len(authors) > 3:
            author_str = f"{authors[0]} et al."
        elif authors:
            author_str = ", ".join(authors[:-1]) + (f", & {authors[-1]}" if len(authors) > 1 else authors[0])
        else:
            author_str = "Unknown"

        venue = f" *{meta['venue']}*." if meta.get("venue") else ""
        doi = f" https://doi.org/{meta['doi']}" if meta.get("doi") else (f" {meta['url']}" if meta.get("url") else "")
        return f"{author_str} ({meta['year']}). {meta['title']}.{venue}{doi}"

    def format_numbered_list(self) -> str:
        """Return a numbered reference list (as used in IEEE / report style)."""
        lines = []
        for i, (key, meta) in enumerate(self._refs.items(), start=1):
            lines.append(f"[{i}] {self.format_apa(key)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # BibTeX export
    # ------------------------------------------------------------------

    def to_bibtex(self) -> str:
        """Render all references as a BibTeX string."""
        entries: list[str] = []
        for cite_key, meta in self._refs.items():
            author_field = " and ".join(meta["authors"]) if meta["authors"] else "Unknown"
            doi_field = f'  doi = {{{meta["doi"]}}},\n' if meta.get("doi") else ""
            url_field = f'  url = {{{meta["url"]}}},\n' if meta.get("url") else ""
            entry = (
                f'@article{{{cite_key},\n'
                f'  title = {{{meta["title"]}}},\n'
                f'  author = {{{author_field}}},\n'
                f'  year = {{{meta["year"]}}},\n'
                f'  journal = {{{meta.get("venue", "arXiv/Online")}}},\n'
                f'{doi_field}'
                f'{url_field}'
                f'}}'
            )
            entries.append(entry)
        return "\n\n".join(entries)

    def save_bibtex(self, path: Path | str) -> Path:
        """Write BibTeX to *path* and return the resolved Path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_bibtex(), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_cite_key(self, paper: "Paper") -> str:
        return self._make_cite_key_from_parts(paper.authors, paper.year, paper.title)

    def _make_cite_key_from_parts(self, authors: list[str], year: str, title: str) -> str:
        first_author = (authors[0].split()[-1] if authors else "unknown").lower()
        first_author = re.sub(r"[^a-z]", "", first_author)
        year_str = str(year)[:4] if year else "0000"
        title_word = re.sub(r"[^a-z]", "", (title.split()[0] if title else "untitled").lower())
        base_key = f"{first_author}{year_str}{title_word}"

        # Disambiguate collisions
        if base_key not in self._refs:
            return base_key
        suffix_ord = ord("b")
        while f"{base_key}{chr(suffix_ord)}" in self._refs:
            suffix_ord += 1
        return f"{base_key}{chr(suffix_ord)}"

"""
Paper summarization and data extraction tools.

Uses IBM Granite (via GraniteClient) to:
  - Summarize individual papers
  - Extract structured metadata (methods, datasets, results)
  - Batch-summarize a list of papers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.llm import GraniteClient
    from research_agent.search import Paper


@dataclass
class PaperSummary:
    """Structured summary of a single paper."""

    paper_id: str
    title: str
    authors: list[str]
    year: str
    url: str
    abstract: str
    summary: str
    key_findings: str = ""
    methods: str = ""
    datasets: str = ""
    limitations: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "url": self.url,
            "abstract": self.abstract,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "methods": self.methods,
            "datasets": self.datasets,
            "limitations": self.limitations,
        }


class PaperSummarizer:
    """Summarizes papers using IBM Granite via GraniteClient."""

    def __init__(self, llm: "GraniteClient") -> None:
        self._llm = llm

    def summarize(self, paper: "Paper") -> PaperSummary:
        """
        Produce a detailed structured summary of a single paper.
        Extracts: summary, key findings, methods, datasets, limitations.
        """
        text = f"Title: {paper.title}\nAuthors: {', '.join(paper.authors)}\nYear: {paper.year}\nAbstract:\n{paper.abstract}"

        # Main summary
        summary_text = self._llm.summarize(text)

        # Structured extraction
        details = self._extract_details(text)

        return PaperSummary(
            paper_id=paper.paper_id,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            url=paper.url,
            abstract=paper.abstract,
            summary=summary_text,
            key_findings=details.get("key_findings", ""),
            methods=details.get("methods", ""),
            datasets=details.get("datasets", ""),
            limitations=details.get("limitations", ""),
        )

    def summarize_batch(
        self, papers: list["Paper"], max_papers: int = 10
    ) -> list[PaperSummary]:
        """Summarize up to *max_papers* papers."""
        results: list[PaperSummary] = []
        for paper in papers[:max_papers]:
            try:
                results.append(self.summarize(paper))
            except Exception as exc:
                print(f"  [summarizer] skipping '{paper.title[:50]}': {exc}")
        return results

    def _extract_details(self, text: str) -> dict[str, str]:
        """Extract structured fields from paper text using Granite."""
        raw = self._llm.extract_details(text)
        return self._parse_fields(raw)

    @staticmethod
    def _parse_fields(text: str) -> dict[str, str]:
        """Parse the structured Granite response into a dict."""
        fields = {
            "key_findings": "",
            "methods": "",
            "datasets": "",
            "limitations": "",
        }
        key_map = {
            "KEY FINDINGS": "key_findings",
            "METHODS": "methods",
            "DATASETS": "datasets",
            "LIMITATIONS": "limitations",
        }
        current_key: str | None = None
        for line in text.splitlines():
            for label, field_key in key_map.items():
                if line.upper().startswith(label + ":"):
                    current_key = field_key
                    fields[field_key] = line[len(label) + 1:].strip()
                    break
            else:
                if current_key:
                    fields[current_key] += " " + line.strip()

        # Clean up whitespace
        return {k: v.strip() for k, v in fields.items()}

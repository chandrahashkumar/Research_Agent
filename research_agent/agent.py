"""
Research Agent orchestrator — ReAct (Reason + Act) loop.

The agent autonomously:
  1. Understands the research question
  2. Plans which tools to use
  3. Searches literature
  4. Summarizes papers
  5. Generates a report
  6. Manages citations
  7. Suggests hypotheses

Tool interface follows a lightweight ReAct pattern:
  Thought → Action → Observation → ... → Final Answer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from research_agent.citations import CitationManager
from research_agent.config import Config
from research_agent.llm import GraniteClient
from research_agent.report import ReportGenerator, ResearchReport
from research_agent.search import LiteratureSearcher, Paper
from research_agent.summarizer import PaperSummary, PaperSummarizer

console = Console()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class ResearchSession:
    """Holds all data accumulated during a single research session."""

    topic: str
    papers: list[Paper] = field(default_factory=list)
    summaries: list[PaperSummary] = field(default_factory=list)
    report: ResearchReport | None = None
    hypothesis_text: str = ""
    answer: str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    The main Research Agent.

    Orchestrates: search → summarize → report → cite → hypothesize
    using IBM Granite as the reasoning backbone.
    """

    AVAILABLE_TOOLS = [
        "search_literature",
        "summarize_papers",
        "generate_report",
        "suggest_hypotheses",
        "answer_question",
        "list_citations",
        "export_bibtex",
    ]

    def __init__(self, config: Config) -> None:
        self._config = config
        self._llm = GraniteClient(config)
        self._searcher = LiteratureSearcher(config)
        self._summarizer = PaperSummarizer(self._llm)
        self._citations = CitationManager()
        self._report_gen = ReportGenerator(self._llm, self._citations)

    # ------------------------------------------------------------------
    # High-level workflow entry points
    # ------------------------------------------------------------------

    def run_full_pipeline(self, topic: str) -> ResearchSession:
        """
        Execute the complete research pipeline for *topic*:
          search → summarize → generate report → suggest hypotheses
        Returns a fully populated ResearchSession.
        """
        session = ResearchSession(topic=topic)

        console.rule(f"[bold blue]Research Agent[/bold blue]  |  IBM Granite (watsonx.ai)")
        console.print(f"\n[bold]Topic:[/bold] {topic}\n")

        # Step 1 — Literature search
        session.papers = self._tool_search(topic)

        # Step 2 — Summarize papers
        session.summaries = self._tool_summarize(session.papers)

        # Step 3 — Generate report
        session.report = self._tool_generate_report(topic, session.summaries)

        # Step 4 — Suggest hypotheses (populated inside report already, expose separately)
        session.hypothesis_text = session.report.hypotheses if session.report else ""

        # Step 5 — Save outputs
        self._save_outputs(session)

        return session

    def ask(self, question: str, session: ResearchSession | None = None) -> str:
        """
        Answer a specific research question, optionally grounded in an existing session.
        """
        context = ""
        if session and session.summaries:
            context = "\n\n".join(
                f"Paper: {s.title} ({s.year})\nSummary: {s.summary}"
                for s in session.summaries[:5]
            )
        with _spinner("Thinking..."):
            answer = self._llm.answer(question, context=context)
        session and setattr(session, "answer", answer)
        return answer

    # ------------------------------------------------------------------
    # Individual tool implementations
    # ------------------------------------------------------------------

    def _tool_search(self, query: str) -> list[Paper]:
        console.print("[bold cyan]>> Step 1/4 - Searching literature...[/bold cyan]")
        with _spinner(f"Querying arXiv + Semantic Scholar for: '{query}'"):
            papers = self._searcher.search(query)

        table = Table(title=f"Found {len(papers)} papers", show_lines=True, header_style="bold blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("Title", min_width=30, max_width=55)
        table.add_column("Authors", max_width=25)
        table.add_column("Year", width=6)
        table.add_column("Source", width=10)
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
            table.add_row(str(i), p.title[:54], authors_str[:24], p.year, p.source)
        console.print(table)
        return papers

    def _tool_summarize(self, papers: list[Paper]) -> list[PaperSummary]:
        console.print("\n[bold cyan]>> Step 2/4 - Summarizing papers with IBM Granite...[/bold cyan]")
        summaries: list[PaperSummary] = []
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Summarizing...", total=len(papers))
            for paper in papers:
                progress.update(task, description=f"Summarizing: {paper.title[:50]}...")
                try:
                    s = self._summarizer.summarize(paper)
                    summaries.append(s)
                except Exception as exc:
                    console.print(f"  [yellow]! Skipped '{paper.title[:40]}': {exc}[/yellow]")
                progress.advance(task)
        console.print(f"  [green]OK[/green] Summarized {len(summaries)} papers")
        return summaries

    def _tool_generate_report(
        self, topic: str, summaries: list[PaperSummary]
    ) -> ResearchReport:
        from research_agent.llm import _WatsonxError

        console.print("\n[bold cyan]>> Step 3/4 - Generating research report...[/bold cyan]")
        sections = [
            "Introduction",
            "Literature Review",
            "Methodology Overview",
            "Key Findings",
            "Hypotheses",
            "Conclusion",
        ]
        llm_err: _WatsonxError | None = None
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Drafting...", total=len(sections))
            for sec in sections:
                progress.update(task, description=f"Drafting: {sec}...")
                progress.advance(task)
            try:
                report = self._report_gen.generate(topic, summaries)
            except _WatsonxError as exc:
                llm_err = exc
                # report was partially built with fallback content; retrieve it
                report = self._report_gen._last_partial_report  # type: ignore[attr-defined]

        if llm_err is not None:
            console.print(
                f"  [yellow]! LLM unavailable — report saved with abstract-based content.[/yellow]\n"
                f"  [red]{llm_err}[/red]"
            )
        else:
            console.print("  [green]OK[/green] Report generated")
        return report

    def _save_outputs(self, session: ResearchSession) -> None:
        console.print("\n[bold cyan]>> Step 4/4 - Saving outputs...[/bold cyan]")
        out = self._config.OUTPUT_DIR

        if session.report:
            md_path = self._report_gen.save_markdown(session.report, out)
            txt_path = self._report_gen.save_text(session.report, out)
            pdf_path = self._report_gen.save_pdf(session.report, out)
            console.print(f"  [green]OK[/green] Markdown : {md_path}")
            console.print(f"  [green]OK[/green] Text     : {txt_path}")
            console.print(f"  [green]OK[/green] PDF      : {pdf_path}")

        bib_path = self._citations.save_bibtex(out / "references.bib")
        console.print(f"  [green]OK[/green] BibTeX   : {bib_path}")

    # ------------------------------------------------------------------
    # Interactive mode helpers
    # ------------------------------------------------------------------

    def list_citations(self) -> list[dict]:
        return self._citations.list_citations()

    def export_bibtex(self, path: str) -> str:
        from pathlib import Path
        p = self._citations.save_bibtex(Path(path))
        return str(p)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _spinner(label: str):
    return Progress(SpinnerColumn(), TextColumn(label), transient=True, console=console)

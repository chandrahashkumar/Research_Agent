"""
Research Agent - Command Line Interface.

Usage:
    python main.py research "large language models in drug discovery"
    python main.py ask "What are the main challenges in protein folding?"
    python main.py ask "What is RAG?" --topic "retrieval augmented generation"
    python main.py interactive
"""

from __future__ import annotations

import sys
import warnings
import argparse
from pathlib import Path

# Suppress IBM SDK deprecation/lifecycle warnings at startup
warnings.filterwarnings("ignore", category=DeprecationWarning)
import importlib as _il
for _mod_name, _cls_name in [
    ("ibm_watsonx_ai.wml_resource", "WatsonxAPIWarning"),
    ("ibm_watsonx_ai.foundation_models.utils.utils", "LifecycleWarning"),
]:
    try:
        _m = _il.import_module(_mod_name)
        _cls = getattr(_m, _cls_name, None)
        if _cls:
            warnings.filterwarnings("ignore", category=_cls)
    except Exception:
        pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from research_agent.agent import ResearchAgent, ResearchSession
from research_agent.config import Config

console = Console()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_research(args: argparse.Namespace, agent: ResearchAgent) -> int:
    """Run the full research pipeline for a topic."""
    topic = " ".join(args.topic)
    session = agent.run_full_pipeline(topic)

    console.rule("[bold green]Research Complete[/bold green]")
    if session.report:
        console.print(Panel(
            Markdown(session.report.to_markdown()[:3000] + "\n\n*(truncated - see output files)*"),
            title="[bold]Report Preview[/bold]",
            border_style="green",
            padding=(1, 2),
        ))
    return 0


def cmd_ask(args: argparse.Namespace, agent: ResearchAgent) -> int:
    """Answer a specific research question, with optional topic-search context."""
    question = " ".join(args.question)
    session: ResearchSession | None = None

    if getattr(args, "topic", None):
        topic = " ".join(args.topic)
        console.print(f"[dim]Searching context for: {topic}...[/dim]")
        papers = agent._searcher.search(topic, max_results=5)
        summaries = agent._summarizer.summarize_batch(papers, max_papers=5)
        session = ResearchSession(topic=topic, papers=papers, summaries=summaries)

    console.print(f"\n[bold]Question:[/bold] {question}\n")
    answer = agent.ask(question, session=session)
    console.print(Panel(
        Markdown(answer),
        title="[bold blue]IBM Granite Answer[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))
    return 0


def cmd_interactive(agent: ResearchAgent) -> int:
    """Start an interactive research session."""
    console.print(Panel(
        "[bold]Welcome to the IBM Granite Research Agent[/bold]\n\n"
        "Commands:\n"
        "  [cyan]research <topic>[/cyan]    - Full pipeline: search + summarize + report\n"
        "  [cyan]ask <question>[/cyan]      - Answer a research question\n"
        "  [cyan]citations[/cyan]           - List all citations in this session\n"
        "  [cyan]bibtex <path>[/cyan]       - Export BibTeX to file\n"
        "  [cyan]exit[/cyan]                - Quit",
        title="[bold blue]Research Agent[/bold blue]",
        border_style="blue",
    ))

    session: ResearchSession | None = None

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]>[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        elif cmd == "research" and arg:
            session = agent.run_full_pipeline(arg)

        elif cmd == "ask" and arg:
            answer = agent.ask(arg, session=session)
            console.print(Panel(Markdown(answer), title="Answer", border_style="blue"))

        elif cmd == "citations":
            refs = agent.list_citations()
            if not refs:
                console.print("[yellow]No citations yet. Run 'research <topic>' first.[/yellow]")
            else:
                for i, ref in enumerate(refs, 1):
                    console.print(f"[{i}] [bold]{ref['title']}[/bold] ({ref['year']}) - {ref['url']}")

        elif cmd == "bibtex":
            path = arg or "./output/references.bib"
            out = agent.export_bibtex(path)
            console.print(f"[green]OK BibTeX saved to {out}[/green]")

        elif cmd == "help":
            console.print("[cyan]Commands: research, ask, citations, bibtex, exit[/cyan]")

        else:
            console.print(f"[yellow]Unknown command or missing argument: '{user_input}'. Type 'help' for usage.[/yellow]")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="IBM Granite Research Agent - AI-powered academic research assistant",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # research
    p_research = subparsers.add_parser("research", help="Run full research pipeline for a topic")
    p_research.add_argument("topic", nargs="+", help="Research topic (can be multi-word)")

    # ask
    p_ask = subparsers.add_parser("ask", help="Answer a research question")
    p_ask.add_argument("question", nargs="+", help="Research question")
    p_ask.add_argument(
        "--topic", nargs="+",
        help="Optional topic to search for context before answering",
    )

    # interactive
    subparsers.add_parser("interactive", help="Launch interactive research session")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Load config (raises EnvironmentError with helpful message if .env not set)
    try:
        config = Config.load()
    except EnvironmentError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return 1

    agent = ResearchAgent(config)

    try:
        if args.command == "research":
            return cmd_research(args, agent)
        elif args.command == "ask":
            return cmd_ask(args, agent)
        elif args.command == "interactive":
            return cmd_interactive(agent)
        else:
            parser.print_help()
            return 1
    except Exception as exc:
        # Catch _WatsonxError (and any other unexpected errors) without a traceback
        from research_agent.llm import _WatsonxError
        if isinstance(exc, _WatsonxError):
            console.print(f"\n[bold red]IBM watsonx.ai error:[/bold red]\n{exc}")
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())

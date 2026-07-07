"""
Research Agent — FastAPI Web Server
====================================
Serves the modern web UI and exposes REST + SSE endpoints.

Endpoints:
  GET  /                     → web app (index.html)
  POST /api/research         → SSE stream: full pipeline
  POST /api/ask              → JSON: answer a question
  GET  /api/citations        → JSON: list citations
  GET  /api/download/{type}  → file download (md / pdf / bib)
  GET  /api/health           → health check
"""

from __future__ import annotations

import asyncio
import json
import warnings
import traceback
from pathlib import Path
from typing import AsyncGenerator

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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from research_agent.agent import ResearchAgent, ResearchSession
from research_agent.config import Config

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(title="Research Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Lazy agent — created once, reused across requests
# ---------------------------------------------------------------------------

_agent: ResearchAgent | None = None
_last_session: ResearchSession | None = None


def get_agent() -> ResearchAgent:
    global _agent
    if _agent is None:
        try:
            config = Config.load()
            _agent = ResearchAgent(config)
        except EnvironmentError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    return _agent


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    topic: str
    max_results: int = 10


class AskRequest(BaseModel):
    question: str
    use_session: bool = True


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

def _evt(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "ibm/granite-8b-code-instruct"}


@app.post("/api/research")
async def research_stream(req: ResearchRequest):
    """
    SSE stream that runs the full research pipeline and pushes progress events.

    Event types:
      status   — progress update  { step, message }
      papers   — search results   { papers: [...] }
      summary  — single paper     { index, total, title, summary }
      report   — final report     { sections: {...} }
      done     — pipeline done    { md_path, pdf_path, bib_path }
      error    — error            { message }
    """
    async def generate() -> AsyncGenerator:
        global _last_session
        agent = get_agent()

        try:
            # ── Step 1: Search ──────────────────────────────────────────
            yield _evt("status", {"step": 1, "message": f"Searching literature for: {req.topic}"})

            loop = asyncio.get_event_loop()
            papers = await loop.run_in_executor(
                None, lambda: agent._searcher.search(req.topic, max_results=req.max_results)
            )

            papers_data = [
                {
                    "title": p.title,
                    "authors": p.authors[:3],
                    "year": p.year,
                    "url": p.url,
                    "source": p.source,
                    "abstract": p.abstract[:300] + ("..." if len(p.abstract) > 300 else ""),
                }
                for p in papers
            ]
            yield _evt("papers", {"count": len(papers), "papers": papers_data})

            # ── Step 2: Summarize ────────────────────────────────────────
            yield _evt("status", {"step": 2, "message": f"Summarizing {len(papers)} papers with IBM Granite..."})

            from research_agent.summarizer import PaperSummarizer
            summarizer = PaperSummarizer(agent._llm)
            summaries = []
            for i, paper in enumerate(papers):
                yield _evt("status", {
                    "step": 2,
                    "message": f"Summarizing paper {i+1}/{len(papers)}: {paper.title[:55]}..."
                })
                try:
                    s = await loop.run_in_executor(None, lambda p=paper: summarizer.summarize(p))
                    summaries.append(s)
                    yield _evt("summary", {
                        "index": i + 1,
                        "total": len(papers),
                        "title": s.title,
                        "summary": s.summary,
                        "key_findings": s.key_findings,
                        "methods": s.methods,
                    })
                except Exception as exc:
                    yield _evt("status", {"step": 2, "message": f"Skipped: {paper.title[:40]}"})

            # ── Step 3: Generate report ─────────────────────────────────
            yield _evt("status", {"step": 3, "message": "Generating research report with IBM Granite..."})

            from research_agent.citations import CitationManager
            from research_agent.report import ReportGenerator
            citations = CitationManager()
            report_gen = ReportGenerator(agent._llm, citations)

            report = await loop.run_in_executor(
                None, lambda: report_gen.generate(req.topic, summaries)
            )

            yield _evt("report", {
                "topic": report.topic,
                "generated_at": report.generated_at,
                "sections": {
                    "introduction": report.introduction,
                    "literature_review": report.literature_review,
                    "methodology_overview": report.methodology_overview,
                    "key_findings": report.key_findings,
                    "hypotheses": report.hypotheses,
                    "conclusion": report.conclusion,
                    "references": report.references,
                }
            })

            # ── Step 4: Save files ──────────────────────────────────────
            yield _evt("status", {"step": 4, "message": "Saving report files..."})

            out = agent._config.OUTPUT_DIR
            md_path = report_gen.save_markdown(report, out)
            txt_path = report_gen.save_text(report, out)
            pdf_path = report_gen.save_pdf(report, out)
            bib_path = citations.save_bibtex(out / "references.bib")

            # Save session for /api/ask
            from research_agent.search import Paper
            _last_session = ResearchSession(
                topic=req.topic,
                papers=papers,
                summaries=summaries,
                report=report,
            )

            yield _evt("done", {
                "md_path": str(md_path),
                "txt_path": str(txt_path),
                "pdf_path": str(pdf_path),
                "bib_path": str(bib_path),
            })

        except Exception as exc:
            yield _evt("error", {"message": str(exc), "detail": traceback.format_exc()})

    return EventSourceResponse(generate())


@app.post("/api/ask")
async def ask_question(req: AskRequest):
    global _last_session
    agent = get_agent()

    loop = asyncio.get_event_loop()
    session = _last_session if req.use_session else None
    answer = await loop.run_in_executor(
        None, lambda: agent.ask(req.question, session=session)
    )
    return {"answer": answer}


@app.get("/api/citations")
async def get_citations():
    global _last_session
    if not _last_session:
        return {"citations": []}
    agent = get_agent()
    return {"citations": agent.list_citations()}


@app.get("/api/download/{file_type}")
async def download_file(file_type: str):
    agent = get_agent()
    out = agent._config.OUTPUT_DIR

    files = {
        "md": list(out.glob("*_report.md")),
        "pdf": list(out.glob("*_report.pdf")),
        "txt": list(out.glob("*_report.txt")),
        "bib": list(out.glob("*.bib")),
    }
    candidates = files.get(file_type, [])
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No {file_type} file found. Run a research query first.")

    # Return the most recently modified file
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    media = {
        "md": "text/markdown",
        "pdf": "application/pdf",
        "txt": "text/plain",
        "bib": "text/plain",
    }.get(file_type, "application/octet-stream")
    return FileResponse(str(latest), media_type=media, filename=latest.name)


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

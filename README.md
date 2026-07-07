# 🔬 Research Agent — IBM Granite (watsonx.ai)

An AI-powered academic research assistant that autonomously searches literature,
summarizes papers, manages citations, and generates structured research reports —
all powered by **IBM Granite** models on **IBM Cloud Lite** (free tier).

---

## Features

| Capability | Details |
|---|---|
| **Literature Search** | Queries arXiv + Semantic Scholar (dual-backend, deduplicated) |
| **Paper Summarization** | IBM Granite extracts summary, methods, findings, limitations |
| **Research Q&A** | Ask any research question; answer grounded in retrieved papers |
| **Report Generation** | Full structured report (Introduction → Conclusion) in MD + PDF |
| **Hypothesis Suggestion** | Granite proposes 3–5 novel testable hypotheses |
| **Citation Management** | APA formatting + BibTeX `.bib` export |
| **Interactive Mode** | Persistent session REPL with all commands |

---

## Prerequisites

- Python 3.10+
- An **IBM Cloud Lite** (free) account: https://cloud.ibm.com/registration
- A **watsonx.ai** project: https://dataplatform.cloud.ibm.com/

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd Research_Agent
pip install -r requirements.txt
```

### 2. Configure IBM Credentials

```bash
cp .env.example .env
# Edit .env and fill in:
#   WATSONX_API_KEY   — from IBM Cloud → Manage → Access (API keys)
#   WATSONX_PROJECT_ID — from your watsonx.ai project settings
```

### 3. Run

```bash
# Full research pipeline
python main.py research "transformer models in medical imaging"

# Answer a specific question (with context search)
python main.py ask "What are the limitations of BERT?" --topic "BERT NLP"

# Interactive session
python main.py interactive
```

---

## Project Structure

```
Research_Agent/
├── main.py                      # CLI entry point
├── requirements.txt
├── pyproject.toml
├── .env.example                 # Copy to .env and fill credentials
├── research_agent/
│   ├── __init__.py
│   ├── config.py                # Environment / credential loader
│   ├── llm.py                   # IBM Granite (watsonx.ai) client
│   ├── agent.py                 # ReAct orchestration loop
│   ├── search.py                # arXiv + Semantic Scholar search
│   ├── summarizer.py            # Paper summarization & data extraction
│   ├── citations.py             # Citation manager + BibTeX export
│   └── report.py                # Report generator (MD, TXT, PDF)
├── tests/
│   └── test_research_agent.py   # Unit tests (no credentials needed)
└── output/                      # Generated reports (auto-created)
```

---

## IBM Cloud Setup (Step-by-step)

1. Create a free IBM Cloud account at https://cloud.ibm.com/registration
2. Go to **Catalog → AI / Machine Learning → Watson Machine Learning** → Create (Lite plan)
3. Go to **Catalog → AI / Machine Learning → Watson Studio** → Create (Lite plan)
4. Open **watsonx.ai** → Create a new project
5. In the project, go to **Manage → General → Project ID** — copy this value
6. Go to **IBM Cloud → Manage → Access (IAM) → API keys** → Create → copy the key
7. Paste both into your `.env` file

---

## Available IBM Granite Models (IBM Cloud Lite)

| Model ID | Description |
|---|---|
| `ibm/granite-13b-instruct-v2` | Default — best for instruction-following |
| `ibm/granite-3-8b-instruct` | Lighter, faster |
| `ibm/granite-20b-multilingual` | Multilingual support |

Change the model by setting `GRANITE_MODEL_ID` in `.env`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests are fully mocked — no IBM credentials required.

---

## Sample Output

```
Research Agent  |  IBM Granite (watsonx.ai)

Topic: large language models in drug discovery

⟳ Step 1/4 — Searching literature…
  Found 10 papers  (arXiv + Semantic Scholar)

⟳ Step 2/4 — Summarizing papers with IBM Granite…
  ✓ Summarized 10 papers

⟳ Step 3/4 — Generating research report…
  ✓ Report generated

⟳ Step 4/4 — Saving outputs…
  ✓ Markdown : output/large_language_models_in_drug_discovery_report.md
  ✓ PDF      : output/large_language_models_in_drug_discovery_report.pdf
  ✓ BibTeX   : output/references.bib
```

---

## License

MIT — see LICENSE

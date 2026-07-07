"""
Unit tests for Research Agent components (no IBM credentials required).
Tests use mocks to avoid live API calls.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from research_agent.citations import CitationManager
from research_agent.search import Paper, ArxivSearcher
from research_agent.summarizer import PaperSummarizer, PaperSummary
from research_agent.report import ResearchReport, _slugify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(n: int = 1) -> Paper:
    return Paper(
        paper_id=f"arxiv:000{n}",
        title=f"Deep Learning for Science {n}",
        authors=[f"Alice Smith", f"Bob Jones"],
        year="2023",
        abstract="We present a novel deep learning framework for scientific discovery.",
        url=f"https://arxiv.org/abs/000{n}",
        source="arxiv",
        doi=f"10.1000/xyz{n}",
        venue="arXiv",
    )


# ---------------------------------------------------------------------------
# CitationManager tests
# ---------------------------------------------------------------------------

class TestCitationManager(unittest.TestCase):

    def test_add_paper_returns_key(self):
        mgr = CitationManager()
        paper = _make_paper(1)
        key = mgr.add_paper(paper)
        self.assertIsInstance(key, str)
        self.assertTrue(len(key) > 0)

    def test_add_paper_idempotent(self):
        mgr = CitationManager()
        paper = _make_paper(1)
        key1 = mgr.add_paper(paper)
        key2 = mgr.add_paper(paper)
        self.assertEqual(key1, key2)

    def test_format_apa(self):
        mgr = CitationManager()
        paper = _make_paper(1)
        key = mgr.add_paper(paper)
        apa = mgr.format_apa(key)
        self.assertIn("2023", apa)
        self.assertIn("Deep Learning", apa)

    def test_bibtex_output(self):
        mgr = CitationManager()
        paper = _make_paper(1)
        mgr.add_paper(paper)
        bibtex = mgr.to_bibtex()
        self.assertIn("@article{", bibtex)
        self.assertIn("Deep Learning for Science", bibtex)

    def test_numbered_list(self):
        mgr = CitationManager()
        for i in range(1, 4):
            mgr.add_paper(_make_paper(i))
        listing = mgr.format_numbered_list()
        self.assertIn("[1]", listing)
        self.assertIn("[3]", listing)

    def test_disambiguation(self):
        """Two papers from same first author and year should get distinct keys."""
        mgr = CitationManager()
        p1 = _make_paper(1)
        p2 = Paper(
            paper_id="arxiv:9999",
            title="Different Title Altogether",
            authors=["Alice Smith"],
            year="2023",
            abstract="Another abstract.",
            url="https://arxiv.org/abs/9999",
            source="arxiv",
        )
        k1 = mgr.add_paper(p1)
        k2 = mgr.add_paper(p2)
        self.assertNotEqual(k1, k2)


# ---------------------------------------------------------------------------
# Paper / Search tests
# ---------------------------------------------------------------------------

class TestPaper(unittest.TestCase):

    def test_to_dict_keys(self):
        paper = _make_paper(1)
        d = paper.to_dict()
        for key in ("paper_id", "title", "authors", "year", "abstract", "url", "source"):
            self.assertIn(key, d)

    def test_paper_authors_list(self):
        paper = _make_paper(1)
        self.assertIsInstance(paper.authors, list)


# ---------------------------------------------------------------------------
# Summarizer tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestPaperSummarizer(unittest.TestCase):

    def _make_llm(self, response: str = "Summary text.") -> MagicMock:
        llm = MagicMock()
        llm.summarize.return_value = response
        llm.extract_details.return_value = "KEY FINDINGS: Finding A\nMETHODS: Method B\nDATASETS: Dataset C\nLIMITATIONS: Limit D"
        return llm

    def test_summarize_returns_summary(self):
        llm = self._make_llm("Great summary.")
        summarizer = PaperSummarizer(llm)
        paper = _make_paper(1)
        result = summarizer.summarize(paper)
        self.assertIsInstance(result, PaperSummary)
        self.assertEqual(result.summary, "Great summary.")
        self.assertEqual(result.title, paper.title)

    def test_summarize_extracts_fields(self):
        llm = self._make_llm()
        summarizer = PaperSummarizer(llm)
        paper = _make_paper(1)
        result = summarizer.summarize(paper)
        self.assertIn("Finding A", result.key_findings)
        self.assertIn("Method B", result.methods)

    def test_batch_summarize_skips_errors(self):
        llm = MagicMock()
        llm.summarize.side_effect = [Exception("API error"), "OK summary"]
        llm.generate.return_value = ""
        summarizer = PaperSummarizer(llm)
        papers = [_make_paper(i) for i in range(1, 3)]
        results = summarizer.summarize_batch(papers)
        # Should return only the successful one
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].summary, "OK summary")


# ---------------------------------------------------------------------------
# ResearchReport tests
# ---------------------------------------------------------------------------

class TestResearchReport(unittest.TestCase):

    def test_to_markdown_contains_sections(self):
        report = ResearchReport("test topic")
        report.introduction = "Intro text."
        report.literature_review = "Lit review."
        report.conclusion = "Conclusion."
        md = report.to_markdown()
        self.assertIn("# Research Report", md)
        self.assertIn("Introduction", md)
        self.assertIn("Conclusion", md)

    def test_to_text_strips_markdown(self):
        report = ResearchReport("test topic")
        report.introduction = "Intro."
        text = report.to_text()
        self.assertNotIn("## ", text)

    def test_slugify(self):
        self.assertEqual(_slugify("Deep Learning!"), "deep_learning")
        self.assertEqual(_slugify("NLP & ML"), "nlp_ml")


# ---------------------------------------------------------------------------
# PaperSummarizer._parse_fields tests
# ---------------------------------------------------------------------------

class TestParseFields(unittest.TestCase):

    def test_parse_all_fields(self):
        raw = (
            "KEY FINDINGS: Finding one.\n"
            "METHODS: Neural networks.\n"
            "DATASETS: ImageNet.\n"
            "LIMITATIONS: Small sample size."
        )
        result = PaperSummarizer._parse_fields(raw)
        self.assertEqual(result["key_findings"], "Finding one.")
        self.assertEqual(result["methods"], "Neural networks.")
        self.assertEqual(result["datasets"], "ImageNet.")
        self.assertEqual(result["limitations"], "Small sample size.")

    def test_parse_missing_fields(self):
        raw = "KEY FINDINGS: Only this.\n"
        result = PaperSummarizer._parse_fields(raw)
        self.assertEqual(result["methods"], "")
        self.assertEqual(result["key_findings"], "Only this.")


if __name__ == "__main__":
    unittest.main()

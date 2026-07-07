"""
IBM watsonx.ai / Granite LLM client wrapper.  # noqa: E501

Uses the official `ibm-watsonx-ai` SDK (>=1.5.14) to call IBM Granite models
available on IBM Cloud Lite (au-syd).

Available text-generation Granite models on this account:
  - ibm/granite-8b-code-instruct   (default — instruction + text_chat)
  - ibm/granite-guardian-3-8b      (safety-tuned — text_chat)
"""

from __future__ import annotations

import warnings

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from research_agent.config import Config

try:
    from ibm_watsonx_ai.wml_client_error import ApiRequestFailure as _ApiRequestFailure
except ImportError:
    _ApiRequestFailure = Exception  # type: ignore[assignment,misc]


class _WatsonxError(RuntimeError):
    """Wraps ApiRequestFailure with a human-readable message."""

    def __init__(self, cause: Exception) -> None:
        body = str(cause)
        if "invalid_instance_status_error" in body and "Inactive" in body:
            msg = (
                "IBM WML instance is Inactive.\n\n"
                "To reactivate:\n"
                "  1. Go to https://cloud.ibm.com/resources\n"
                "  2. Find your Watson Machine Learning service\n"
                "  3. Click the service → check its Status/Plan\n"
                "  4. On Lite plan: open the service, click 'Launch' or interact\n"
                "     with it to wake it, or delete & recreate the Lite instance.\n"
                "  5. Re-run the agent once the instance shows 'Active'."
            )
        else:
            msg = f"IBM watsonx.ai API error: {cause}"
        super().__init__(msg)


# Suppress IBM SDK LifecycleWarning / WatsonxAPIWarning for deprecation notices
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from ibm_watsonx_ai.wml_client_error import WatsonxAPIWarning  # type: ignore
    warnings.filterwarnings("ignore", category=WatsonxAPIWarning)
except ImportError:
    pass
try:
    from ibm_watsonx_ai.foundation_models.utils.utils import LifecycleWarning  # type: ignore
    warnings.filterwarnings("ignore", category=LifecycleWarning)
except ImportError:
    pass


class GraniteClient:
    """Thin wrapper around IBM watsonx.ai ModelInference for Granite models."""

    def __init__(self, config: Config) -> None:
        self._config = config
        credentials = Credentials(
            api_key=config.WATSONX_API_KEY,
            url=config.WATSONX_URL,
        )
        self._api_client = APIClient(credentials=credentials)
        self._api_client.set.default_project(config.WATSONX_PROJECT_ID)

        self._model = ModelInference(
            model_id=config.GRANITE_MODEL_ID,
            api_client=self._api_client,
            project_id=config.WATSONX_PROJECT_ID,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: config.MAX_SUMMARY_TOKENS,
                GenParams.REPETITION_PENALTY: 1.05,
            },
        )

    # ------------------------------------------------------------------
    # Core generation — uses chat() which both Granite models support
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Send *prompt* as a user chat message to Granite and return the reply.

        Uses the chat API (text_chat function) which works on both
        granite-8b-code-instruct and granite-guardian-3-8b.
        """
        messages = [{"role": "user", "content": prompt}]
        params: dict = {}
        if max_tokens is not None:
            params[GenParams.MAX_NEW_TOKENS] = max_tokens

        try:
            response = self._model.chat(
                messages=messages,
                params=params if params else None,
            )
        except _ApiRequestFailure as exc:
            raise _WatsonxError(exc) from exc

        # SDK chat response: response["choices"][0]["message"]["content"]
        choices = response.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()

    def generate_with_system(
        self, system: str, user: str, max_tokens: int | None = None
    ) -> str:
        """Send a system + user message pair and return the reply."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        params: dict = {}
        if max_tokens is not None:
            params[GenParams.MAX_NEW_TOKENS] = max_tokens

        try:
            response = self._model.chat(
                messages=messages,
                params=params if params else None,
            )
        except _ApiRequestFailure as exc:
            raise _WatsonxError(exc) from exc

        choices = response.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def summarize(self, text: str, context: str = "") -> str:
        """Return a concise academic summary of *text*."""
        ctx = f"Context: {context}\n\n" if context else ""
        system = (
            "You are an expert academic research assistant. "
            "Summarize the following research content clearly and concisely "
            "for a scientific audience. Focus on: objectives, methods, key findings, "
            "and conclusions. Be direct — no preamble."
        )
        user = f"{ctx}Text to summarize:\n{text}"
        return self.generate_with_system(system, user, max_tokens=self._config.MAX_SUMMARY_TOKENS)

    def answer(self, question: str, context: str = "") -> str:
        """Answer a research *question*, optionally grounded in *context*."""
        system = (
            "You are an expert AI research assistant powered by IBM Granite. "
            "Answer the research question thoroughly, citing key concepts and evidence. "
            "Be precise and academic in tone."
        )
        ctx_block = f"\nRelevant context:\n{context}\n" if context else ""
        user = f"Research question: {question}{ctx_block}"
        return self.generate_with_system(system, user, max_tokens=self._config.MAX_REPORT_TOKENS)

    def generate_report_section(self, section: str, papers: list[dict]) -> str:
        """Draft a *section* of a research report grounded in *papers*."""
        papers_text = "\n\n".join(
            f"[{i+1}] Title: {p.get('title','')}\n"
            f"    Authors: {', '.join(p.get('authors', []))}\n"
            f"    Year: {p.get('year','')}\n"
            f"    Abstract: {p.get('abstract','')[:500]}"
            for i, p in enumerate(papers[:8])
        )
        system = (
            f"You are an expert academic writer. "
            f"Write a well-structured '{section}' section for a research report. "
            f"Use formal academic language, integrate the provided references, "
            f"and include in-text citations like [1], [2]. "
            f"Be thorough and analytical."
        )
        user = f"References:\n{papers_text}\n\nWrite the '{section}' section."
        return self.generate_with_system(system, user, max_tokens=self._config.MAX_REPORT_TOKENS)

    def suggest_hypotheses(self, topic: str, summaries: list[str]) -> str:
        """Suggest research hypotheses for *topic* based on existing *summaries*."""
        combined = "\n---\n".join(summaries[:5])
        system = (
            "You are a creative and rigorous research scientist. "
            "Based on the literature summaries provided, suggest 3 to 5 novel, testable "
            "research hypotheses related to the topic. Be specific and grounded in the literature."
        )
        user = f"Topic: {topic}\n\nLiterature summaries:\n{combined}"
        return self.generate_with_system(system, user, max_tokens=self._config.MAX_REPORT_TOKENS)

    def extract_details(self, text: str) -> str:
        """Extract structured fields (findings/methods/datasets/limitations) from paper text."""
        system = (
            "You are a research data extraction expert. "
            "Given the paper text, extract each field. "
            "If information is not available, write 'Not specified'. "
            "Reply in exactly this format:\n"
            "KEY FINDINGS: ...\nMETHODS: ...\nDATASETS: ...\nLIMITATIONS: ..."
        )
        return self.generate_with_system(system, text, max_tokens=400)

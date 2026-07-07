"""
Configuration loader for Research Agent.
Reads credentials and settings from environment / .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (or existing environment variables)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val or val.startswith("your_"):
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example → .env and fill in your IBM Cloud credentials."
        )
    return val


class Config:
    """Central configuration object."""

    # IBM watsonx.ai
    WATSONX_API_KEY: str = ""
    WATSONX_PROJECT_ID: str = ""
    WATSONX_URL: str = "https://au-syd.ml.cloud.ibm.com"

    # IBM Granite model to use (available on IBM Cloud Lite — au-syd)
    GRANITE_MODEL_ID: str = "ibm/granite-8b-code-instruct"

    # Semantic Scholar (optional — empty string = unauthenticated, 100 req/5min)
    SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    # Output
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output"))

    # Agent behaviour
    MAX_SEARCH_RESULTS: int = 10
    MAX_SUMMARY_TOKENS: int = 512
    MAX_REPORT_TOKENS: int = 1024
    TEMPERATURE: float = 0.3

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        cfg.WATSONX_API_KEY = _require("WATSONX_API_KEY")
        cfg.WATSONX_PROJECT_ID = _require("WATSONX_PROJECT_ID")
        cfg.WATSONX_URL = os.getenv("WATSONX_URL", cls.WATSONX_URL)
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return cfg

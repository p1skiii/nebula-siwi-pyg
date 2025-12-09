import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _bool_env(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _int_env(var_name: str, default: int) -> int:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Load .env from project root (three levels up from this file)
_ROOT_DIR = Path(__file__).resolve().parents[2]
_ENV_PATH = _ROOT_DIR / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """Lightweight settings pulled from environment with sane defaults."""

    debug: bool = _bool_env("API_DEBUG", True)
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = _int_env("API_PORT", 5000)

    rag_top_k: int = _int_env("RAG_TOP_K", 4)
    rag_data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data/demo_docs"))
    embed_backend: str = os.getenv("RAG_EMBED_BACKEND", "tfidf")

    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    llm_api_key: str = os.getenv("LLM_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    llm_allow_mock: bool = _bool_env("LLM_ALLOW_MOCK", False)
    llm_model: str = os.getenv("LLM_MODEL", "gemini-1.5-flash-002")

    enable_rag: bool = _bool_env("RAG_ENABLED", False)
    enable_graph: bool = _bool_env("ENABLE_GRAPH_BACKEND", False)
    agent_experimental: bool = _bool_env("AGENT_EXPERIMENTAL", False)


def get_settings() -> Settings:
    return Settings()

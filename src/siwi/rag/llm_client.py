from abc import ABC, abstractmethod
from typing import List

from siwi.rag.doc_loader import DocumentChunk


class ConfigError(Exception):
    """Raised when LLM configuration is invalid."""


class LLMClient(ABC):
    @abstractmethod
    def generate(self, query: str, context: List[DocumentChunk]) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Mock client that stitches retrieved context into a friendly reply."""

    def __init__(self, demo_prefix: str = ""):
        self.demo_prefix = demo_prefix

    def generate(self, query: str, context: List[DocumentChunk]) -> str:
        prefix = self.demo_prefix
        if not context:
            return f"{prefix}I could not find anything relevant. Echoing your question: {query}"

        highlights = []
        for chunk in context:
            highlights.append(f"[{chunk.title}] {chunk.content}")
        joined = " ".join(highlights)[:800]
        return f"{prefix}Based on what I found: {joined}"


class GeminiLLMClient(LLMClient):
    """Google Gemini client. Requires google-generativeai and a valid API key."""

    def __init__(self, api_key: str, model: str):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "google-generativeai is not installed. Install it to use Gemini."
            ) from exc

        if not api_key:
            raise ConfigError("Gemini API key is required for provider=gemini.")

        self.genai = genai
        self.genai.configure(api_key=api_key)
        self.model = model

    def generate(self, query: str, context: List[DocumentChunk]) -> str:
        prompt_parts = [f"Question: {query}"]
        if context:
            prompt_parts.append("Context:")
            for chunk in context:
                prompt_parts.append(f"- {chunk.title}: {chunk.content}")
        prompt = "\n".join(prompt_parts)

        model = self.genai.GenerativeModel(self.model)
        try:
            response = model.generate_content(prompt)
            return response.text or ""
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc


def build_llm_client(provider: str, api_key: str, allow_mock: bool = False, model: str = "") -> LLMClient:
    """Factory with explicit key requirement and controlled mock usage."""
    provider = (provider or "").lower()
    if provider == "mock":
        if not allow_mock:
            raise ConfigError("Mock LLM is demo-only. Set LLM_ALLOW_MOCK=1 to enable.")
        return MockLLMClient(demo_prefix="[DEMO MODE] ")

    if provider == "gemini":
        if not api_key:
            raise ConfigError("Gemini API key is required (set GEMINI_API_KEY or LLM_API_KEY).")
        return GeminiLLMClient(api_key=api_key, model=model or "gemini-1.5-flash-002")

    raise ConfigError(f"Unsupported LLM provider: {provider}")

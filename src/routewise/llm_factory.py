"""Build the configured LLM client (Groq or Ollama)."""

from routewise.config import Settings
from routewise.llm import LLMClient, OllamaClient
from routewise.llm_groq import GroqClient


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "groq":
        return GroqClient.from_settings(settings)
    return OllamaClient.from_settings(settings)

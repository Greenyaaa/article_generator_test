"""
LLM client factory. Returns the correct provider based on model name prefix.

Supported providers:
  claude-*           -> AnthropicClient
  gpt-*, o1-*, o3-*  -> OpenAIClient
"""

from typing import Optional

from providers.anthropic_client import AnthropicClient
from providers.openai_client import OpenAIClient

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4096

_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-")


def LLMClient(
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """
    Factory function. Returns the appropriate provider for the given model name.

    Args:
        model: Model name (e.g. 'claude-sonnet-4-5', 'gpt-4o').
        api_key: Optional API key. If not provided, read from environment variable.
        max_tokens: Maximum tokens for the response.

    Returns:
        A provider instance implementing BaseProvider.

    Raises:
        ValueError: If the model prefix is not recognised.
    """
    if model.startswith("claude-"):
        return AnthropicClient(model=model, api_key=api_key, max_tokens=max_tokens)
    elif model.startswith(_OPENAI_PREFIXES):
        return OpenAIClient(model=model, api_key=api_key, max_tokens=max_tokens)
    else:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Supported prefixes: 'claude-' (Anthropic), "
            f"'gpt-'/'o1-'/'o3-' (OpenAI)."
        )

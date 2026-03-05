"""
Anthropic API client with retry logic and error handling.
"""

import logging
import os
import time
from typing import Optional

import anthropic
from anthropic import APIConnectionError, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds


class LLMClient:
    """
    Wrapper around the Anthropic API with retry logic and logging.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        logger.info(f"LLMClient initialized with model='{self.model}'")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Retries up to MAX_RETRIES times on transient errors.

        Args:
            system_prompt: The system instruction for the model.
            user_prompt: The user message / content to process.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"LLM request attempt {attempt}/{MAX_RETRIES} "
                    f"(model={self.model}, max_tokens={self.max_tokens})"
                )
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                result = response.content[0].text
                logger.debug(
                    f"LLM response received: {len(result.split())} words, "
                    f"input_tokens={response.usage.input_tokens}, "
                    f"output_tokens={response.usage.output_tokens}"
                )
                return result

            except RateLimitError as e:
                wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Rate limit hit. Waiting {wait}s before retry... ({e})")
                time.sleep(wait)

            except APIConnectionError as e:
                wait = RETRY_BASE_DELAY * attempt
                logger.warning(f"Connection error. Waiting {wait}s before retry... ({e})")
                time.sleep(wait)

            except APIStatusError as e:
                # 5xx errors are retryable; 4xx (except 429) are not
                if e.status_code >= 500:
                    wait = RETRY_BASE_DELAY * attempt
                    logger.warning(
                        f"Server error {e.status_code}. Waiting {wait}s before retry..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Non-retryable API error {e.status_code}: {e.message}")
                    raise

        raise RuntimeError(
            f"LLM request failed after {MAX_RETRIES} attempts. Check logs for details."
        )

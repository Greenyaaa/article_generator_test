"""
OpenAI API provider with retry logic and error handling.
"""

import logging
import os
import time
from typing import Optional

import openai
from openai import APIConnectionError, APIStatusError, RateLimitError

from providers.base import BaseProvider

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds


class OpenAIClient(BaseProvider):
    """
    OpenAI API provider with retry logic and logging.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        logger.info(f"OpenAIClient initialized with model='{self.model}'")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a prompt to OpenAI and return the text response.

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
                    f"OpenAI request attempt {attempt}/{MAX_RETRIES} "
                    f"(model={self.model}, max_tokens={self.max_tokens})"
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                result = response.choices[0].message.content
                logger.debug(
                    f"OpenAI response received: {len(result.split())} words, "
                    f"input_tokens={response.usage.prompt_tokens}, "
                    f"output_tokens={response.usage.completion_tokens}"
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

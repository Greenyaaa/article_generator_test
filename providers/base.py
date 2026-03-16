"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Args:
            system_prompt: The system instruction for the model.
            user_prompt: The user message / content to process.

        Returns:
            The model's text response.
        """
        ...

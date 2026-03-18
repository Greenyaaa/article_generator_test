"""
Parser for the article generation input file.
Uses an LLM to extract structured data from free-form text.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from providers.base import BaseProvider

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class ArticleInput:
    topic: str
    bullets: list[str]
    language: str = "English"


def parse_input_file(file_path: str | Path, client: "BaseProvider") -> ArticleInput:
    """
    Parse an input file using an LLM to extract topic, bullets, and language
    from free-form text.

    Args:
        file_path: Path to the input text file.
        client:    An LLM client instance with a .complete() method.

    Returns:
        ArticleInput dataclass with extracted fields.

    Raises:
        FileNotFoundError: If the input file or prompt file does not exist.
        ValueError: If the LLM response cannot be parsed or required fields are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find input file: '{file_path}'. "
            "Please check the path and try again."
        )

    logger.info(f"Parsing input file via LLM: {path}")
    raw_text = path.read_text(encoding="utf-8")

    prompt_path = PROMPTS_DIR / "input_parser_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Parser prompt not found: {prompt_path}")
    system_prompt = prompt_path.read_text(encoding="utf-8")

    llm_response = client.complete(system_prompt=system_prompt, user_prompt=raw_text)
    article_input = _parse_llm_response(llm_response)

    logger.info(
        f"Parsed via LLM: topic='{article_input.topic}', "
        f"language='{article_input.language}', bullets={len(article_input.bullets)}"
    )
    return article_input


def _parse_llm_response(response: str) -> ArticleInput:
    """
    Parse the JSON response from the LLM into an ArticleInput.

    Defensively strips markdown code fences in case the model adds them.

    Raises:
        ValueError: If JSON is invalid or required fields are missing/empty.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON. Parse error: {e}\n"
            f"Raw response was:\n{response}"
        )

    topic = data.get("topic", "").strip()
    language = data.get("language", "English").strip() or "English"
    bullets = data.get("bullets", [])

    if not topic:
        raise ValueError(
            "LLM extraction returned an empty topic. "
            "Check your input file has a clear article subject."
        )
    if not isinstance(bullets, list) or not bullets:
        raise ValueError(
            "LLM extraction returned no bullet points. "
            "Check your input file includes key points to cover."
        )

    bullets = [str(b).strip() for b in bullets if str(b).strip()]
    if not bullets:
        raise ValueError(
            "LLM extraction returned bullet points that are all empty after stripping."
        )

    return ArticleInput(topic=topic, bullets=bullets, language=language)

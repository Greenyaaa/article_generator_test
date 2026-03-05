"""
Parser for the article generation input file.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ArticleInput:
    topic: str
    bullets: list[str]
    language: str = "English"


def parse_input_file(file_path: str | Path) -> ArticleInput:
    """
    Parse an input file with the following format:

        Topic: Why small businesses should automate repetitive processes
        Language: English
        Bullets:
        - manual work limits growth
        - hiring increases costs

    Args:
        file_path: Path to the input text file.

    Returns:
        ArticleInput dataclass with parsed fields.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing or malformed.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find input file: '{file_path}'. "
            "Please check the path and try again."
        )

    logger.info(f"Parsing input file: {path}")
    text = path.read_text(encoding="utf-8")

    topic = _extract_field(text, "topic")
    language = _extract_field(text, "language", default="English")
    bullets = _extract_bullets(text)

    if not topic:
        raise ValueError(
            "Your input file is missing a topic. "
            "Please add a line like: Topic: Why automation matters"
        )
    if not bullets:
        raise ValueError(
            "Your input file has no bullet points. "
            "Please add at least one line like: - your key point"
        )

    article_input = ArticleInput(topic=topic, bullets=bullets, language=language)
    logger.info(
        f"Parsed: topic='{topic}', language='{language}', bullets={len(bullets)}"
    )
    return article_input


def _extract_field(text: str, field: str, default: str | None = None) -> str | None:
    """Extract a single-line field like 'Topic: ...' from text."""
    for line in text.splitlines():
        if line.lower().startswith(f"{field.lower()}:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    return default


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet points (lines starting with '-') from text."""
    bullets = []
    in_bullets_section = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("bullets:"):
            in_bullets_section = True
            continue
        if in_bullets_section:
            if stripped.startswith("-"):
                bullet = stripped[1:].strip()
                if bullet:
                    bullets.append(bullet)
            elif stripped and not stripped.startswith("-"):
                # Stop if we hit a non-bullet, non-empty line
                break

    return bullets

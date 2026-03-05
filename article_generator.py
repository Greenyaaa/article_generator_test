"""
Article generation pipeline: draft → edit → save.
"""

import logging
from pathlib import Path

from input_parser import ArticleInput
from llm_client import LLMClient

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
OUTPUT_DIR = Path(__file__).parent / "output"


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _format_bullets(bullets: list[str]) -> str:
    """Format bullet points as a readable list for the prompt."""
    return "\n".join(f"- {b}" for b in bullets)


def _save_output(filename: str, content: str) -> Path:
    """Save content to the output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"Saved: {output_path}")
    return output_path


def generate_draft(
    client: LLMClient,
    article_input: ArticleInput,
    word_count: int,
) -> str:
    """
    Generate the first draft of the article.

    Args:
        client: LLMClient instance.
        article_input: Parsed article parameters.
        word_count: Target word count.

    Returns:
        Draft article text.
    """
    logger.info("Generating draft...")

    system_prompt = _load_prompt("draft_prompt.txt").format(
        language=article_input.language,
        word_count=word_count,
    )

    user_prompt = (
        f"Topic: {article_input.topic}\n\n"
        f"Key points to cover:\n{_format_bullets(article_input.bullets)}"
    )

    draft = client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    logger.info(f"Draft generated: {len(draft.split())} words")
    return draft


def edit_article(
    client: LLMClient,
    draft: str,
    article_input: ArticleInput,
    word_count: int,
) -> str:
    """
    Run the editor pass to improve the draft.

    Args:
        client: LLMClient instance.
        draft: The first draft text.
        article_input: Original article parameters (for language/length).
        word_count: Target word count.

    Returns:
        Improved article text.
    """
    logger.info("Running editor pass...")

    system_prompt = _load_prompt("editor_prompt.txt").format(
        language=article_input.language,
        word_count=word_count,
    )

    user_prompt = f"Please edit and improve the following article draft:\n\n{draft}"

    final = client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    logger.info(f"Final article generated: {len(final.split())} words")
    return final


def run_pipeline(
    article_input: ArticleInput,
    client: LLMClient,
    word_count: int = 800,
) -> tuple[Path, Path]:
    """
    Full pipeline: generate draft → edit → save both versions.

    Args:
        article_input: Parsed article input.
        client: LLMClient instance.
        word_count: Target word count for the article.

    Returns:
        Tuple of (draft_path, final_path).
    """
    draft = generate_draft(client, article_input, word_count)
    draft_path = _save_output("draft.md", draft)

    final = edit_article(client, draft, article_input, word_count)
    final_path = _save_output("final.md", final)

    return draft_path, final_path

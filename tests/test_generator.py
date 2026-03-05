"""
Tests for article_generator.py — LLM calls are mocked, no API key required.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_parser import ArticleInput
from article_generator import generate_draft, edit_article, run_pipeline, _format_bullets, _save_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input():
    return ArticleInput(
        topic="Why automation matters",
        bullets=["saves time", "reduces errors", "scales easily"],
        language="English",
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.complete.return_value = "This is a generated article body."
    return client


# ---------------------------------------------------------------------------
# _format_bullets
# ---------------------------------------------------------------------------

def test_format_bullets():
    result = _format_bullets(["point one", "point two"])
    assert result == "- point one\n- point two"


def test_format_bullets_single():
    assert _format_bullets(["only one"]) == "- only one"


# ---------------------------------------------------------------------------
# generate_draft
# ---------------------------------------------------------------------------

def test_generate_draft_calls_complete(sample_input, mock_client, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "draft_prompt.txt").write_text(
        "You are a writer. Language: {language}. Words: {word_count}.", encoding="utf-8"
    )

    with patch("article_generator.PROMPTS_DIR", prompts_dir):
        result = generate_draft(mock_client, sample_input, word_count=500)

    mock_client.complete.assert_called_once()
    call_kwargs = mock_client.complete.call_args
    assert "Why automation matters" in call_kwargs.kwargs["user_prompt"]
    assert "saves time" in call_kwargs.kwargs["user_prompt"]
    assert result == "This is a generated article body."


def test_generate_draft_missing_prompt_raises(sample_input, mock_client, tmp_path):
    empty_dir = tmp_path / "prompts"
    empty_dir.mkdir()

    with patch("article_generator.PROMPTS_DIR", empty_dir):
        with pytest.raises(FileNotFoundError):
            generate_draft(mock_client, sample_input, word_count=500)


# ---------------------------------------------------------------------------
# edit_article
# ---------------------------------------------------------------------------

def test_edit_article_calls_complete(sample_input, mock_client, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "editor_prompt.txt").write_text(
        "You are an editor. Language: {language}. Words: {word_count}.", encoding="utf-8"
    )
    mock_client.complete.return_value = "Edited article content."

    with patch("article_generator.PROMPTS_DIR", prompts_dir):
        result = edit_article(mock_client, "draft text", sample_input, word_count=500)

    mock_client.complete.assert_called_once()
    call_kwargs = mock_client.complete.call_args
    assert "draft text" in call_kwargs.kwargs["user_prompt"]
    assert result == "Edited article content."


# ---------------------------------------------------------------------------
# _save_output
# ---------------------------------------------------------------------------

def test_save_output_creates_file(tmp_path):
    with patch("article_generator.OUTPUT_DIR", tmp_path / "output"):
        path = _save_output("draft.md", "Article content here.")

    assert path.exists()
    assert path.read_text(encoding="utf-8") == "Article content here."
    assert path.name == "draft.md"


def test_save_output_creates_directory(tmp_path):
    output_dir = tmp_path / "new_output_dir"
    assert not output_dir.exists()

    with patch("article_generator.OUTPUT_DIR", output_dir):
        _save_output("final.md", "content")

    assert output_dir.exists()


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

def test_run_pipeline_returns_two_paths(sample_input, mock_client, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "draft_prompt.txt").write_text(
        "Draft prompt {language} {word_count}", encoding="utf-8"
    )
    (prompts_dir / "editor_prompt.txt").write_text(
        "Editor prompt {language} {word_count}", encoding="utf-8"
    )
    output_dir = tmp_path / "output"

    with patch("article_generator.PROMPTS_DIR", prompts_dir), \
         patch("article_generator.OUTPUT_DIR", output_dir):
        draft_path, final_path = run_pipeline(sample_input, mock_client, word_count=500)

    assert draft_path.name == "draft.md"
    assert final_path.name == "final.md"
    assert draft_path.exists()
    assert final_path.exists()


def test_run_pipeline_calls_llm_twice(sample_input, mock_client, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "draft_prompt.txt").write_text("Draft {language} {word_count}", encoding="utf-8")
    (prompts_dir / "editor_prompt.txt").write_text("Editor {language} {word_count}", encoding="utf-8")
    output_dir = tmp_path / "output"

    with patch("article_generator.PROMPTS_DIR", prompts_dir), \
         patch("article_generator.OUTPUT_DIR", output_dir):
        run_pipeline(sample_input, mock_client, word_count=500)

    assert mock_client.complete.call_count == 2

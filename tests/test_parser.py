"""
Tests for input_parser.py — LLM call is mocked, no API key required.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_parser import parse_input_file, ArticleInput, _parse_llm_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client(response: str) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


def write_input(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "input.txt"
    p.write_text(content, encoding="utf-8")
    return p


def valid_json_response(topic="Test topic", language="English", bullets=None):
    return json.dumps({
        "topic": topic,
        "language": language,
        "bullets": bullets or ["point one", "point two"],
    })


# ---------------------------------------------------------------------------
# parse_input_file — happy path
# ---------------------------------------------------------------------------

def test_parse_returns_article_input(tmp_path):
    f = write_input(tmp_path, "Write about automation.")
    client = make_client(valid_json_response())
    result = parse_input_file(f, client)
    assert isinstance(result, ArticleInput)
    assert result.topic == "Test topic"
    assert result.language == "English"
    assert result.bullets == ["point one", "point two"]


def test_parse_passes_file_content_to_llm(tmp_path):
    f = write_input(tmp_path, "My article idea text.")
    client = make_client(valid_json_response())
    parse_input_file(f, client)
    call_kwargs = client.complete.call_args
    assert "My article idea text." in call_kwargs.kwargs["user_prompt"]


def test_parse_uses_system_prompt_from_file(tmp_path):
    f = write_input(tmp_path, "Some input.")
    client = make_client(valid_json_response())
    parse_input_file(f, client)
    call_kwargs = client.complete.call_args
    assert len(call_kwargs.kwargs["system_prompt"]) > 0


def test_parse_custom_language(tmp_path):
    f = write_input(tmp_path, "Write in Russian about something.")
    client = make_client(valid_json_response(language="Russian"))
    result = parse_input_file(f, client)
    assert result.language == "Russian"


def test_parse_free_form_text(tmp_path):
    """Parser should handle completely free-form text, not just structured input."""
    f = write_input(tmp_path, "I need an article about climate change. Cover sea levels, wildfires, and policy.")
    client = make_client(valid_json_response(
        topic="Climate change",
        bullets=["sea levels", "wildfires", "policy"],
    ))
    result = parse_input_file(f, client)
    assert result.topic == "Climate change"
    assert len(result.bullets) == 3


# ---------------------------------------------------------------------------
# parse_input_file — error cases
# ---------------------------------------------------------------------------

def test_missing_file_raises():
    client = make_client("")
    with pytest.raises(FileNotFoundError, match="Could not find input file"):
        parse_input_file("/non/existent/file.txt", client)


def test_missing_file_does_not_call_llm():
    client = make_client("")
    try:
        parse_input_file("/non/existent/file.txt", client)
    except FileNotFoundError:
        pass
    client.complete.assert_not_called()


# ---------------------------------------------------------------------------
# _parse_llm_response — happy path
# ---------------------------------------------------------------------------

def test_parse_llm_response_basic():
    raw = valid_json_response(topic="AI trends", bullets=["fast", "powerful"])
    result = _parse_llm_response(raw)
    assert result.topic == "AI trends"
    assert result.bullets == ["fast", "powerful"]
    assert result.language == "English"


def test_parse_llm_response_strips_markdown_fences():
    raw = "```json\n" + valid_json_response() + "\n```"
    result = _parse_llm_response(raw)
    assert result.topic == "Test topic"


def test_parse_llm_response_strips_plain_fences():
    raw = "```\n" + valid_json_response() + "\n```"
    result = _parse_llm_response(raw)
    assert result.topic == "Test topic"


def test_parse_llm_response_defaults_language_to_english():
    raw = json.dumps({"topic": "Something", "bullets": ["x"]})
    result = _parse_llm_response(raw)
    assert result.language == "English"


def test_parse_llm_response_unicode_topic():
    raw = valid_json_response(topic="Почему автоматизация важна", language="Russian")
    result = _parse_llm_response(raw)
    assert result.topic == "Почему автоматизация важна"
    assert result.language == "Russian"


def test_parse_llm_response_special_chars_in_bullets():
    raw = valid_json_response(bullets=["cost: $500/month", "supports C++, Python & Go"])
    result = _parse_llm_response(raw)
    assert "cost: $500/month" in result.bullets
    assert "supports C++, Python & Go" in result.bullets


# ---------------------------------------------------------------------------
# _parse_llm_response — error cases
# ---------------------------------------------------------------------------

def test_parse_llm_response_invalid_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_llm_response("this is not json at all")


def test_parse_llm_response_empty_topic_raises():
    raw = json.dumps({"topic": "", "bullets": ["x"], "language": "English"})
    with pytest.raises(ValueError, match="empty topic"):
        _parse_llm_response(raw)


def test_parse_llm_response_missing_bullets_raises():
    raw = json.dumps({"topic": "Something", "bullets": [], "language": "English"})
    with pytest.raises(ValueError, match="no bullet points"):
        _parse_llm_response(raw)


def test_parse_llm_response_bullets_not_list_raises():
    raw = json.dumps({"topic": "Something", "bullets": "not a list", "language": "English"})
    with pytest.raises(ValueError, match="no bullet points"):
        _parse_llm_response(raw)

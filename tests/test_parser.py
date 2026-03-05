"""
Tests for input_parser.py
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_parser import parse_input_file, ArticleInput, _extract_field, _extract_bullets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_input(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "input.txt"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# parse_input_file — happy path
# ---------------------------------------------------------------------------

def test_parse_full_input(tmp_path):
    f = write_input(tmp_path, """\
Topic: Why small businesses should automate
Language: English
Bullets:
- manual work limits growth
- hiring increases costs
- automation improves scalability
""")
    result = parse_input_file(f)
    assert isinstance(result, ArticleInput)
    assert result.topic == "Why small businesses should automate"
    assert result.language == "English"
    assert result.bullets == [
        "manual work limits growth",
        "hiring increases costs",
        "automation improves scalability",
    ]


def test_parse_language_defaults_to_english(tmp_path):
    f = write_input(tmp_path, """\
Topic: Some topic
Bullets:
- point one
""")
    result = parse_input_file(f)
    assert result.language == "English"


def test_parse_custom_language(tmp_path):
    f = write_input(tmp_path, """\
Topic: Тема статьи
Language: Russian
Bullets:
- первый тезис
- второй тезис
""")
    result = parse_input_file(f)
    assert result.language == "Russian"
    assert len(result.bullets) == 2


# ---------------------------------------------------------------------------
# parse_input_file — error cases
# ---------------------------------------------------------------------------

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Could not find input file"):
        parse_input_file("/non/existent/file.txt")


def test_missing_topic_raises(tmp_path):
    f = write_input(tmp_path, """\
Bullets:
- only a bullet
""")
    with pytest.raises(ValueError, match="missing a topic"):
        parse_input_file(f)


def test_missing_bullets_raises(tmp_path):
    f = write_input(tmp_path, """\
Topic: A topic with no bullets
""")
    with pytest.raises(ValueError, match="no bullet points"):
        parse_input_file(f)


def test_empty_bullets_section_raises(tmp_path):
    f = write_input(tmp_path, """\
Topic: A topic
Bullets:
""")
    with pytest.raises(ValueError, match="no bullet points"):
        parse_input_file(f)


# ---------------------------------------------------------------------------
# _extract_field
# ---------------------------------------------------------------------------

def test_extract_field_found():
    assert _extract_field("Topic: Hello world\n", "topic") == "Hello world"


def test_extract_field_case_insensitive():
    assert _extract_field("TOPIC: Hello\n", "topic") == "Hello"


def test_extract_field_missing_returns_default():
    assert _extract_field("Bullets:\n- x\n", "topic", default="fallback") == "fallback"


def test_extract_field_missing_returns_none():
    assert _extract_field("Bullets:\n- x\n", "topic") is None


# ---------------------------------------------------------------------------
# _extract_bullets
# ---------------------------------------------------------------------------

def test_extract_bullets_basic():
    text = "Bullets:\n- first\n- second\n- third\n"
    assert _extract_bullets(text) == ["first", "second", "third"]


def test_extract_bullets_strips_whitespace():
    text = "Bullets:\n-   spaced bullet   \n"
    assert _extract_bullets(text) == ["spaced bullet"]


def test_extract_bullets_stops_at_non_bullet_line():
    text = "Bullets:\n- point one\nTopic: stops here\n- should not be included\n"
    assert _extract_bullets(text) == ["point one"]


def test_extract_bullets_empty_section():
    assert _extract_bullets("Bullets:\n") == []


def test_extract_bullets_no_section():
    assert _extract_bullets("Topic: no bullets here\n") == []

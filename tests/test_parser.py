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


# ---------------------------------------------------------------------------
# Special characters
# ---------------------------------------------------------------------------

def test_topic_with_special_characters(tmp_path):
    """Topic with punctuation, quotes, slashes — should parse without error."""
    f = write_input(tmp_path, """\
Topic: Why "automation" costs < $1000/month & saves time!
Bullets:
- first point
""")
    result = parse_input_file(f)
    assert result.topic == 'Why "automation" costs < $1000/month & saves time!'


def test_topic_with_unicode(tmp_path):
    """Non-latin scripts in topic — Chinese, Arabic, Cyrillic."""
    f = write_input(tmp_path, """\
Topic: 自动化如何帮助企业 / Как автоматизация помогает / كيف تساعد الأتمتة
Bullets:
- reduces manual work
""")
    result = parse_input_file(f)
    assert "自动化" in result.topic
    assert "автоматизация" in result.topic


def test_bullets_with_special_characters(tmp_path):
    """Bullets containing colons, dashes, symbols — should not confuse the parser."""
    f = write_input(tmp_path, """\
Topic: Test
Bullets:
- cost: $500/month
- rate is 99.9% uptime
- supports C++, Python & Go
- item — with em-dash
""")
    result = parse_input_file(f)
    assert len(result.bullets) == 4
    assert "cost: $500/month" in result.bullets
    assert "rate is 99.9% uptime" in result.bullets
    assert "supports C++, Python & Go" in result.bullets
    assert "item — with em-dash" in result.bullets


def test_bullets_with_unicode(tmp_path):
    """Bullets in non-latin scripts — should parse correctly."""
    f = write_input(tmp_path, """\
Topic: Some topic
Language: Russian
Bullets:
- ручной труд ограничивает рост
- найм увеличивает расходы
- автоматизация улучшает масштабируемость
""")
    result = parse_input_file(f)
    assert len(result.bullets) == 3
    assert "ручной труд ограничивает рост" in result.bullets


def test_topic_with_colon_inside(tmp_path):
    """Topic containing a colon should not be truncated at the colon."""
    f = write_input(tmp_path, """\
Topic: Automation: why it matters in 2025
Bullets:
- saves time
""")
    result = parse_input_file(f)
    assert result.topic == "Automation: why it matters in 2025"


def test_extra_whitespace_in_topic(tmp_path):
    """Leading/trailing whitespace in topic should be stripped."""
    f = write_input(tmp_path, """\
Topic:    lots of spaces around
Bullets:
- point
""")
    result = parse_input_file(f)
    assert result.topic == "lots of spaces around"

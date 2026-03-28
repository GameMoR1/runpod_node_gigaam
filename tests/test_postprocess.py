from __future__ import annotations

import pytest

from app.postprocess import postprocess_text, _has_triplet_repeat


@pytest.mark.unit
class TestPostprocess:
    def test_filters_short_lines(self):
        text = "ab\nпривет\nx"
        result = postprocess_text(text)
        assert "привет" in result
        assert "ab" not in result
        assert "x" not in result

    def test_filters_non_cyrillic_lines(self):
        text = "hello\nпривет\nworld"
        result = postprocess_text(text)
        assert "привет" in result
        assert "hello" not in result
        assert "world" not in result

    def test_filters_triplet_repeats(self):
        text = "нормально\nааа\nпривет"
        result = postprocess_text(text)
        assert "нормально" in result
        assert "привет" in result
        assert "ааа" not in result

    def test_keeps_valid_lines(self):
        text = "привет мир\nкак дела?\nвсё хорошо"
        result = postprocess_text(text)
        assert "привет мир" in result
        assert "как дела?" in result
        assert "всё хорошо" in result

    def test_empty_input_returns_empty(self):
        assert postprocess_text("") == ""
        assert postprocess_text("   ") == ""

    def test_triplet_repeat_detection(self):
        assert _has_triplet_repeat("aaa") is True
        assert _has_triplet_repeat("ааа") is True
        assert _has_triplet_repeat("abbb") is True
        assert _has_triplet_repeat("аббб") is True
        assert _has_triplet_repeat("abc") is False
        assert _has_triplet_repeat("абв") is False
        assert _has_triplet_repeat("aa") is False

    def test_handles_multiple_issues(self):
        text = "hi\nааа\nпривет\nab\nнормально"
        result = postprocess_text(text)
        assert result == "привет\nнормально"

    def test_preserves_line_order(self):
        text = "первая\nвторая\nтретья"
        result = postprocess_text(text)
        lines = result.splitlines()
        assert lines == ["первая", "вторая", "третья"]

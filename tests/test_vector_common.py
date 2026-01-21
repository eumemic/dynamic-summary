"""Tests for shared vector index utilities in vector_common.py."""

from __future__ import annotations

from ragzoom.backends.vector_common import (
    coerce_float,
    coerce_int,
    coerce_str,
    normalize_metadata_from_dict,
    normalize_metadata_from_object,
)


class TestCoerceFloat:
    """Tests for coerce_float helper."""

    def test_none_returns_none(self) -> None:
        assert coerce_float(None) is None

    def test_bool_true_returns_one_point_zero(self) -> None:
        assert coerce_float(True) == 1.0

    def test_bool_false_returns_zero_point_zero(self) -> None:
        assert coerce_float(False) == 0.0

    def test_int_converts_to_float(self) -> None:
        assert coerce_float(42) == 42.0

    def test_float_passthrough(self) -> None:
        assert coerce_float(3.14159) == 3.14159

    def test_numeric_string_parses(self) -> None:
        assert coerce_float("123.456") == 123.456

    def test_integer_string_parses(self) -> None:
        assert coerce_float("789") == 789.0

    def test_string_with_whitespace(self) -> None:
        assert coerce_float("  42.5  ") == 42.5

    def test_empty_string_returns_none(self) -> None:
        assert coerce_float("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert coerce_float("   ") is None

    def test_non_numeric_string_returns_none(self) -> None:
        assert coerce_float("hello") is None

    def test_scientific_notation_parses(self) -> None:
        assert coerce_float("1.5e10") == 1.5e10

    def test_negative_float_parses(self) -> None:
        assert coerce_float("-99.9") == -99.9


class TestCoerceInt:
    """Tests for coerce_int helper."""

    def test_bool_true_returns_one(self) -> None:
        assert coerce_int(True) == 1

    def test_bool_false_returns_zero(self) -> None:
        assert coerce_int(False) == 0

    def test_int_passthrough(self) -> None:
        assert coerce_int(42) == 42

    def test_float_truncates(self) -> None:
        assert coerce_int(3.7) == 3

    def test_digit_string_parses(self) -> None:
        assert coerce_int("123") == 123

    def test_digit_string_with_whitespace(self) -> None:
        assert coerce_int("  456  ") == 456

    def test_non_digit_string_returns_zero(self) -> None:
        assert coerce_int("hello") == 0

    def test_none_returns_zero(self) -> None:
        assert coerce_int(None) == 0


class TestCoerceStr:
    """Tests for coerce_str helper."""

    def test_none_returns_empty(self) -> None:
        assert coerce_str(None) == ""

    def test_string_passthrough(self) -> None:
        assert coerce_str("hello") == "hello"

    def test_int_converts(self) -> None:
        assert coerce_str(42) == "42"

    def test_float_converts(self) -> None:
        assert coerce_str(3.14) == "3.14"


class TestNormalizeMetadataFromDict:
    """Tests for normalize_metadata_from_dict."""

    def test_all_fields_present(self) -> None:
        meta = {
            "span_start": 10,
            "span_end": 20,
            "parent_id": "parent-1",
            "document_id": "doc-1",
            "is_leaf": True,
            "height": 2,
            "level_index": 3,
            "coord_version": 1,
        }
        result = normalize_metadata_from_dict(meta)
        assert result["span_start"] == 10
        assert result["span_end"] == 20
        assert result["parent_id"] == "parent-1"
        assert result["document_id"] == "doc-1"
        assert result["is_leaf"] == 1  # bool coerced to int
        assert result["height"] == 2
        assert result["level_index"] == 3
        assert result["coord_version"] == 1

    def test_empty_dict_uses_defaults(self) -> None:
        result = normalize_metadata_from_dict({})
        assert result["span_start"] == 0
        assert result["span_end"] == 0
        assert result["parent_id"] == ""
        assert result["document_id"] == ""
        assert result["is_leaf"] == 0
        assert result["height"] == 0
        assert result["level_index"] == 0
        assert result["coord_version"] == 0

    def test_partial_dict_fills_defaults(self) -> None:
        meta = {"span_start": 5, "document_id": "doc-x"}
        result = normalize_metadata_from_dict(meta)
        assert result["span_start"] == 5
        assert result["document_id"] == "doc-x"
        assert result["span_end"] == 0  # default
        assert result["parent_id"] == ""  # default


class MetaObject:
    """Mock object with metadata attributes for testing."""

    def __init__(
        self,
        span_start: int,
        span_end: int,
        parent_id: str,
        document_id: str,
        is_leaf: bool,
        height: int = 0,
        level_index: int = 0,
        coord_version: int = 0,
    ) -> None:
        self.span_start = span_start
        self.span_end = span_end
        self.parent_id = parent_id
        self.document_id = document_id
        self.is_leaf = is_leaf
        self.height = height
        self.level_index = level_index
        self.coord_version = coord_version


class TestNormalizeMetadataFromObject:
    """Tests for normalize_metadata_from_object."""

    def test_all_fields_present(self) -> None:
        obj = MetaObject(
            span_start=10,
            span_end=20,
            parent_id="parent-1",
            document_id="doc-1",
            is_leaf=True,
            height=2,
            level_index=3,
            coord_version=1,
        )
        result = normalize_metadata_from_object(obj)
        assert result["span_start"] == 10
        assert result["span_end"] == 20
        assert result["parent_id"] == "parent-1"
        assert result["document_id"] == "doc-1"
        assert result["is_leaf"] == 1  # bool coerced to int
        assert result["height"] == 2
        assert result["level_index"] == 3
        assert result["coord_version"] == 1

    def test_missing_optional_fields_use_defaults(self) -> None:
        # Object without height, level_index, coord_version attributes
        class MinimalMeta:
            span_start = 5
            span_end = 15
            parent_id = "p"
            document_id = "d"
            is_leaf = False

        result = normalize_metadata_from_object(MinimalMeta())
        assert result["span_start"] == 5
        assert result["span_end"] == 15
        assert result["height"] == 0  # default
        assert result["level_index"] == 0  # default
        assert result["coord_version"] == 0  # default

from __future__ import annotations

import pytest

from mdra.utils.yolo_label_utils import read_yolo_label, validate_yolo_label


def test_read_and_validate_valid_label(tmp_path):
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.2 0.4\n5 0.25 0.25 0.1 0.1\n", encoding="utf-8")

    records = read_yolo_label(label)
    result = validate_yolo_label(label, num_classes=6)

    assert len(records) == 2
    assert result["valid"] is True
    assert result["num_objects"] == 2
    assert records[0]["area_norm"] == pytest.approx(0.08)


def test_bbox_corner_overflow_is_detected(tmp_path):
    label = tmp_path / "overflow.txt"
    label.write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")

    result = validate_yolo_label(label, num_classes=6, check_corners=True)

    assert result["valid"] is False
    assert any("corners exceed" in error for error in result["errors"])


def test_invalid_class_and_column_count_are_detected(tmp_path):
    invalid_class = tmp_path / "invalid_class.txt"
    invalid_class.write_text("-1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    invalid_columns = tmp_path / "invalid_columns.txt"
    invalid_columns.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")

    class_result = validate_yolo_label(invalid_class, num_classes=6)
    column_result = validate_yolo_label(invalid_columns, num_classes=6)

    assert class_result["valid"] is False
    assert any("non-negative" in error for error in class_result["errors"])
    assert column_result["valid"] is False
    assert any("expected 5 columns" in error for error in column_result["errors"])


def test_empty_label_is_valid_and_marked_empty(tmp_path):
    label = tmp_path / "empty.txt"
    label.write_text("", encoding="utf-8")

    result = validate_yolo_label(label, num_classes=6)

    assert result["valid"] is True
    assert result["empty"] is True
    assert result["num_objects"] == 0

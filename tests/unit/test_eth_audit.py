import json
from pathlib import Path

import pytest

from trackvision.io.eth import (
    ETHFormatError,
    ETHObservation,
    ETHSplit,
    build_eth_windows,
    read_social_stgcnn_file,
    validate_social_stgcnn_file,
)
from trackvision.io.eth_audit import audit_social_stgcnn_dataset


EXPECTED = {
    "train": ("HOTEL",),
    "val": ("HOTEL",),
    "test": ("ETH",),
}


def _write_expected_layout(root: Path, train_rows: str, val_rows: str) -> Path:
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "test").mkdir()
    (root / "train" / "biwi_hotel_train.txt").write_text(train_rows, encoding="utf-8")
    (root / "val" / "biwi_hotel_val.txt").write_text(val_rows, encoding="utf-8")
    (root / "test" / "biwi_eth.txt").write_text(
        "0 9 2.0 3.0\n", encoding="utf-8"
    )
    return root


def _rows(start: int, count: int, pedestrian_id: int = 1) -> str:
    return "".join(
        f"{frame} {pedestrian_id} {frame / 10} {-frame / 10}\n"
        for frame in range(start, start + count * 10, 10)
    )


def _audit(root: Path):
    return audit_social_stgcnn_dataset(
        root,
        frame_step=10,
        expected_scenes_by_split=EXPECTED,
    )


def test_validation_reports_malformed_rows_and_reader_fails_clearly(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_text(
        "0 1 1 2\n"
        "10 1 missing 2\n"
        "20 1 nan 2\n"
        "30 1 3\n",
        encoding="utf-8",
    )

    validation = validate_social_stgcnn_file(path, "HOTEL", "train")

    assert len(validation.observations) == 1
    assert [(issue.code, issue.line_number) for issue in validation.issues] == [
        ("non_numeric_value", 2),
        ("non_finite_coordinate", 3),
        ("column_count", 4),
    ]
    with pytest.raises(ETHFormatError, match="invalid.txt:2"):
        read_social_stgcnn_file(path, "HOTEL", "train")


@pytest.mark.parametrize(
    "rows, expected_code",
    [
        ("0 1 1 2\n0 1 1 2\n", "duplicate_observation"),
        ("0 1 1 2\n0 1 3 4\n", "conflicting_observation"),
        ("10 1 1 2\n0 1 3 4\n", "out_of_order_observation"),
        ("-1 1 1 2\n", "invalid_frame_id"),
        ("0 -1 1 2\n", "invalid_pedestrian_id"),
        ("0 1 inf 2\n", "non_finite_coordinate"),
    ],
)
def test_validation_rejects_duplicate_order_and_invalid_values(
    tmp_path, rows, expected_code
):
    path = tmp_path / "invalid.txt"
    path.write_text(rows, encoding="utf-8")

    result = validate_social_stgcnn_file(path, "HOTEL", "train")

    assert expected_code in {issue.code for issue in result.issues}
    assert not result.is_valid


def test_missing_and_empty_files_are_reported(tmp_path):
    missing = validate_social_stgcnn_file(
        tmp_path / "missing.txt", "HOTEL", "train"
    )
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text(" \n\n", encoding="utf-8")
    empty = validate_social_stgcnn_file(empty_path, "HOTEL", "train")

    assert missing.issues[0].code == "missing_file"
    assert empty.issues[0].code == "empty_file"


def test_audit_reports_counts_gaps_windows_and_split_overlap_deterministically(tmp_path):
    root = _write_expected_layout(
        tmp_path / "eth",
        _rows(0, 20) + _rows(250, 20, pedestrian_id=2),
        _rows(200, 20),
    )

    first = _audit(root)
    second = _audit(root)
    report = first.as_dict()

    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == report
    train = next(
        item for item in report["split_scene_summaries"]
        if item["split"] == "train" and item["scene"] == "HOTEL"
    )
    assert train["observation_count"] == 40
    assert train["unique_pedestrian_count"] == 2
    assert train["frame_id_range"] == {"min": 0, "max": 440}
    assert train["frame_gap_distribution"] == {"10": 38}
    assert train["candidate_window_count"] == 2
    overlap = report["overlap_summaries"][0]
    assert overlap["shared_pedestrian_ids"] == 1
    assert overlap["shared_frame_pedestrian_rows"] == 0
    assert overlap["identical_full_observation_rows"] == 0
    assert overlap["minimum_nearest_frame_distance_for_shared_ids"] == 10
    assert overlap["validation_start_minus_train_end_frame"] == {
        "min": 10,
        "max": 10,
        "validation_after_train_for_all_shared_ids": True,
    }


def test_audit_flags_missing_expected_files_unexpected_entries_and_empty_data(tmp_path):
    root = tmp_path / "eth"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "test").mkdir()
    (root / "train" / "unexpected.txt").write_text("0 1 2 3\n", encoding="utf-8")
    (root / "train" / "extra-directory").mkdir()

    report = _audit(root)
    issue_codes = {issue["code"] for issue in report.issues}

    assert "missing_expected_file" in issue_codes
    assert "unexpected_file" in issue_codes
    assert "unexpected_directory" in issue_codes
    assert "empty_file" not in issue_codes
    assert all(
        item["candidate_window_count"] is None
        for item in report.split_scene_summaries
        if not item["file_present"]
    )


def test_audit_counts_non_finite_values_and_marks_bad_file_windows_unknown(tmp_path):
    root = _write_expected_layout(
        tmp_path / "eth",
        "0 1 nan inf\n10 1 not-a-number 2\n",
        _rows(0, 20),
    )

    report = _audit(root)
    train = next(
        item for item in report.split_scene_summaries
        if item["split"] == "train"
    )

    assert train["non_finite_value_count"] == 2
    assert train["rejected_row_count"] == 2
    assert train["candidate_window_count"] is None
    assert [issue["code"] for issue in train["invalid_or_rejected_rows"]] == [
        "non_finite_coordinate",
        "non_numeric_value",
    ]


def test_audit_marks_insufficient_valid_tracks_as_zero_windows(tmp_path):
    root = _write_expected_layout(tmp_path / "eth", _rows(0, 19), _rows(0, 19))

    report = _audit(root)

    train = next(
        item for item in report.split_scene_summaries
        if item["split"] == "train"
    )
    assert train["candidate_window_count"] == 0
    assert not report.issues


def test_window_builder_rejects_directly_constructed_duplicate_observations(tmp_path):
    dataset = ETHSplit(
        "HOTEL",
        "train",
        tmp_path / "manual.txt",
        (
            ETHObservation(0, 1, 0.0, 0.0),
            ETHObservation(0, 1, 1.0, 1.0),
        ),
    )

    with pytest.raises(ETHFormatError, match="duplicate frame/pedestrian pair"):
        build_eth_windows(dataset, frame_step=10)

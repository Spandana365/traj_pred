from collections import defaultdict
from pathlib import Path

import pytest

from trackvision.io.eth import (
    ETHFormatError,
    build_eth_windows,
    count_split_windows,
    discover_social_stgcnn_files,
    load_social_stgcnn_splits,
    read_social_stgcnn_file,
)


def test_reader_parses_four_fields_and_preserves_coordinates(tmp_path):
    path = tmp_path / "eth.txt"
    path.write_text("0.0\t7.0\t1.25\t-3.5\n10\t7\t2.5\t-2.0\n", encoding="utf-8")

    dataset = read_social_stgcnn_file(path, "ETH", "test")

    assert dataset.observations[0].frame == 0
    assert dataset.observations[0].pedestrian_id == 7
    assert (dataset.observations[0].x, dataset.observations[0].y) == (1.25, -3.5)
    assert dataset.observations[1].frame == 10


@pytest.mark.parametrize(
    "content, message",
    [
        ("0 1 2\n", "expected 4 columns"),
        ("0 1 nan 2\n", "values must be finite"),
        ("0.5 1 2 3\n", "frame_id must be an integer"),
        ("0 1.5 2 3\n", "ped_id must be an integer"),
        ("0 1 2 3\n0 1 4 5\n", "duplicate frame/pedestrian pair"),
    ],
)
def test_reader_rejects_malformed_or_ambiguous_rows(tmp_path, content, message):
    path = tmp_path / "invalid.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ETHFormatError, match=message):
        read_social_stgcnn_file(path, "ETH", "train")


def test_windows_require_exact_cadence_and_do_not_cross_pedestrians_or_gaps(tmp_path):
    path = tmp_path / "data.txt"
    rows = []
    for frame in range(0, 200, 10):
        rows.append(f"{frame} 1 {frame} {-frame}")
    for frame in range(0, 200, 10):
        rows.append(f"{frame} 2 {1000 + frame} {frame}")
    rows.extend(("220 1 220 -220", "230 1 230 -230"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = read_social_stgcnn_file(path, "ETH", "train")
    windows = build_eth_windows(dataset, frame_step=10, history_len=8, horizon_len=12)

    assert len(windows) == 2
    assert {window.pedestrian_id for window in windows} == {1, 2}
    assert all(len(window.observed) == 8 and len(window.future) == 12 for window in windows)
    assert all(window.observed[-1].frame + 10 == window.future[0].frame for window in windows)


def test_windows_keep_file_scene_and_split_boundaries(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    content = "\n".join(f"{frame} 1 {frame} {frame}" for frame in range(0, 200, 10))
    train.write_text(content, encoding="utf-8")
    val.write_text(content, encoding="utf-8")

    train_data = read_social_stgcnn_file(train, "HOTEL", "train")
    val_data = read_social_stgcnn_file(val, "HOTEL", "val")

    assert {window.split for window in build_eth_windows(train_data, 10)} == {"train"}
    assert {window.split for window in build_eth_windows(val_data, 10)} == {"val"}


def test_discovery_maps_known_scenes_and_splits(tmp_path):
    (tmp_path / "test").mkdir()
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "test" / "biwi_eth.txt").write_text("0 1 2 3\n", encoding="utf-8")
    (tmp_path / "train" / "biwi_hotel_train.txt").write_text(
        "0 1 2 3\n", encoding="utf-8"
    )
    (tmp_path / "val" / "biwi_hotel_val.txt").write_text(
        "0 1 2 3\n", encoding="utf-8"
    )

    assert discover_social_stgcnn_files(tmp_path) == (
        (tmp_path / "train" / "biwi_hotel_train.txt", "HOTEL", "train"),
        (tmp_path / "val" / "biwi_hotel_val.txt", "HOTEL", "val"),
        (tmp_path / "test" / "biwi_eth.txt", "ETH", "test"),
    )


DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "eth"
    / "Social-STGCNN"
    / "datasets"
    / "eth"
)


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="local Social-STGCNN ETH data not installed")
def test_local_social_stgcnn_eth_window_counts():
    splits = load_social_stgcnn_splits(DATASET_ROOT)

    assert count_split_windows(splits, frame_step=10) == {
        ("train", "HOTEL"): 877,
        ("train", "ZARA1"): 1976,
        ("train", "ZARA2"): 4477,
        ("train", "ZARA3"): 1760,
        ("train", "STUDENTS001"): 11691,
        ("train", "UNIV"): 8988,
        ("train", "UNI_EXAMPLES"): 538,
        ("val", "HOTEL"): 318,
        ("val", "ZARA1"): 337,
        ("val", "ZARA2"): 1259,
        ("val", "ZARA3"): 708,
        ("val", "STUDENTS001"): 1887,
        ("val", "UNIV"): 834,
        ("val", "UNI_EXAMPLES"): 79,
        ("test", "ETH"): 364,
    }


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="local Social-STGCNN ETH data not installed")
def test_local_social_stgcnn_eth_splits_have_no_shared_pedestrian_frame_rows():
    splits = load_social_stgcnn_splits(DATASET_ROOT)
    by_scene_split = {}
    for dataset in splits:
        rows = {(o.pedestrian_id, o.frame) for o in dataset.observations}
        by_scene_split[(dataset.scene, dataset.split)] = rows

    for scene in {dataset.scene for dataset in splits}:
        train_rows = by_scene_split.get((scene, "train"), set())
        val_rows = by_scene_split.get((scene, "val"), set())
        assert train_rows.isdisjoint(val_rows), scene

import pytest

from trackvision.io.ucy import (
    DatasetFormatError,
    load_ucy_scene,
    read_ucy_annotations,
)


def test_read_vsp_preserves_pixel_coordinates_and_source_frames(tmp_path):
    annotations = tmp_path / "students003.vsp"
    annotations.write_text(
        "2 - the number of splines\n"
        "2 - Num of control points\n"
        "-1.0 -2.0 0 -79.5 - (2D point, m_id)\n"
        "3.0 4.0 37 -75.0 - (2D point, m_id)\n"
        "- a comment\n"
        "1 - Num of control points\n"
        "5.0 6.0 10 0.0 - (2D point, m_id)\n"
        "7 - number of line obstacles\n"
        "0 0 1 1 0 - obstacle data\n",
        encoding="utf-8",
    )

    scene = read_ucy_annotations(annotations, "UNIV")

    assert scene.scene == "UNIV"
    assert len(scene.tracks) == 2
    assert scene.tracks[0].spline_id == 1
    assert [(point.x, point.y, point.frame) for point in scene.tracks[0].points] == [
        (-1.0, -2.0, 0),
        (3.0, 4.0, 37),
    ]
    assert scene.tracks[0].points[0].gaze_direction_degrees == -79.5
    assert scene.trailing_lines == (
        "7 - number of line obstacles",
        "0 0 1 1 0 - obstacle data",
    )


def test_vsp_declared_point_count_must_match_file(tmp_path):
    annotations = tmp_path / "broken.vsp"
    annotations.write_text("1 - splines\n2 - points\n0 0 0 0\n", encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="incomplete spline"):
        read_ucy_annotations(annotations, "UNIV")


def test_vsp_frames_must_be_strictly_increasing(tmp_path):
    annotations = tmp_path / "unordered.vsp"
    annotations.write_text(
        "1 - splines\n2 - points\n0 0 10 0\n1 1 10 0\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetFormatError, match="frames .* must increase"):
        read_ucy_annotations(annotations, "UNIV")


@pytest.mark.parametrize("coordinates", ["nan 0", "inf 0", "0 -inf"])
def test_vsp_rejects_non_finite_coordinates(tmp_path, coordinates):
    annotations = tmp_path / "non-finite.vsp"
    x, y = coordinates.split()
    annotations.write_text(
        f"1 - splines\n1 - points\n{x} {y} 0 0\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetFormatError, match="must be finite"):
        read_ucy_annotations(annotations, "UNIV")


def test_load_ucy_scene_uses_protocol_annotation_name(tmp_path):
    (tmp_path / "crowds_zara01.vsp").write_text(
        "1 - splines\n1 - points\n10 20 4 0\n",
        encoding="utf-8",
    )

    scene = load_ucy_scene(tmp_path, "zara1")

    assert scene.scene == "ZARA1"
    assert scene.tracks[0].points[0].frame == 4


def test_load_ucy_scene_rejects_unknown_scenes(tmp_path):
    with pytest.raises(ValueError, match="unsupported UCY scene"):
        load_ucy_scene(tmp_path, "ZARA3")

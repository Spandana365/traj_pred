"""Tests for the read-only MOT17 and additional ETH API endpoints."""

import asyncio
import csv
import json
import shutil
from pathlib import Path

import httpx
import pytest

from trackvision.api.app import create_app
from trackvision.api.mot17_repository import (
    COMBINED,
    EXPERIMENTS,
    FRAME_PANELS,
    SEQUENCES,
    Mot17Repository,
    _cached_csv,
    _closed_histogram,
    _histogram,
)
from trackvision.api.result_repository import (
    ETH_FIGURES,
    ArtifactInvalid,
    ArtifactNotFound,
    UnknownExperiment,
    _cached_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = PROJECT_ROOT / "runs"
MOT17 = RUNS / "mot17_tracking_analysis"
GMC = MOT17 / "gmc_fixed"
BASE = "/api/v1/datasets/mot17/tracking"

needs_results = pytest.mark.skipif(
    not (GMC / "gmc" / "diagnostics.json").is_file(),
    reason="Saved MOT17 results are not available.",
)


def _get(app, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(url, headers=headers)

    return asyncio.run(request())


@pytest.fixture
def api():
    return create_app(RUNS)


def _saved_metrics(root: Path) -> dict:
    return json.loads((root / "trackeval" / "metrics.json").read_text(encoding="utf-8"))["metrics"]


@needs_results
def test_summary_uses_gmc_run_as_primary_and_saved_values(api):
    body = _get(api, f"{BASE}/summary").json()
    saved = _saved_metrics(GMC)["COMBINED_SEQ"]
    archived = _saved_metrics(MOT17)["COMBINED_SEQ"]
    assert body["primary_experiment"] == "BoT-SORT_GMC_enabled"
    assert body["comparison_experiment"] == "BoT-SORT_no_GMC"
    assert body["dataset"] == "mot17"
    assert body["total_frames"] == 5316
    assert body["gmc_calls"] == 5316
    assert body["gmc_failed_calls"] == 0
    for metric in ("HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW", "CLR_FP", "Frag"):
        assert body["metrics"][metric] == pytest.approx(saved[metric])
        assert body["comparison_metrics"][metric] == pytest.approx(archived[metric])


@needs_results
def test_summary_never_presents_pixels_as_metres(api):
    body = _get(api, f"{BASE}/summary").json()
    assert "not metres" in body["coordinate_units"]
    assert "no MOT17 prediction" in body["scope_note"]


@needs_results
def test_sequences_match_the_saved_inventory_and_metrics(api):
    body = _get(api, f"{BASE}/sequences").json()
    saved = _saved_metrics(GMC)
    assert [row["sequence"] for row in body["sequences"]] == list(SEQUENCES)
    assert sum(row["frames"] for row in body["sequences"]) == 5316
    frame_rates = {row["sequence"]: row["frame_rate"] for row in body["sequences"]}
    assert frame_rates["MOT17-05"] == 14
    assert frame_rates["MOT17-13"] == 25
    for row in body["sequences"]:
        assert row["canonical_directory"] == f"{row['sequence']}-FRCNN"
        assert row["metrics"]["HOTA"] == pytest.approx(saved[row["sequence"]]["HOTA"])


@needs_results
def test_camera_classes_come_from_saved_motion(api):
    classes = {
        row["sequence"]: row["camera_class"]
        for row in _get(api, f"{BASE}/sequences").json()["sequences"]
    }
    assert {name for name, kind in classes.items() if kind == "static-looking"} == {
        "MOT17-02", "MOT17-04", "MOT17-09",
    }


@needs_results
def test_configuration_reports_saved_detector_and_tracker(api):
    body = _get(api, f"{BASE}/configuration").json()
    assert body["training_performed"] is False
    assert body["detector"]["model_name"] == "yolov8s"
    assert body["detector"]["image_size"] == 1280
    assert body["tracker_settings"]["gmc_method"] == "sparseOptFlow"
    assert body["tracker_settings"]["with_reid"] is False
    assert body["settings_changed_from_comparison"] is False
    assert len(body["weights_sha256"]) == 64


@needs_results
def test_gmc_diagnostics_account_for_every_frame(api):
    body = _get(api, f"{BASE}/gmc").json()
    totals = body["totals"]
    assert totals["calls"] == totals["successful_calls"] == 5316
    assert totals["failed_calls"] == 0
    assert totals["estimated_warps"] == 5309
    assert totals["first_frame_identity_warps"] == 7
    assert totals["non_first_identity_warps"] == 0
    assert body["control_identical_to_no_gmc"] is True
    assert len(body["sequences"]) == 7


@needs_results
@pytest.mark.parametrize("experiment, root", [("gmc_enabled", GMC), ("no_gmc", MOT17)])
def test_error_profile_is_labelled_and_built_from_that_experiment(api, experiment, root):
    body = _get(api, f"{BASE}/error-profile?experiment={experiment}").json()
    assert body["experiment"] == EXPERIMENTS[experiment].label
    profile = json.loads(
        (root / "error_profiles" / "error_profile.json").read_text(encoding="utf-8")
    )["overall"]
    assert body["overall"]["true_positives"] == profile["true_positives"]
    assert body["overall"]["iou"]["median"] == pytest.approx(profile["iou"]["median"])
    assert sum(row["count"] for row in body["iou_histogram"]) == profile["true_positives"]
    assert sum(row["count"] for row in body["center_error_histogram"]) == profile["true_positives"]
    assert sum(row["count"] for row in body["coverage_histogram"]) == 546
    assert sum(row["count"] for row in body["gap_histogram"]) == profile["fragmentations"]


@needs_results
def test_availability_pools_counts_without_averaging_percentages(api):
    body = _get(api, f"{BASE}/error-profile").json()
    with (GMC / "error_profiles" / "window_availability.csv").open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if float(row["window_seconds"]) == 8.0]
    available = sum(int(row["available_windows"]) for row in rows)
    possible = sum(int(row["possible_windows"]) for row in rows)
    pooled = next(row for row in body["availability"] if row["window_seconds"] == 8.0)
    assert pooled["availability"] == pytest.approx(available / possible)
    assert pooled["available_windows"] == 9757


@needs_results
def test_ablation_keeps_experiments_separate_and_reports_the_regression(api):
    body = _get(api, f"{BASE}/ablation").json()
    assert [row["sequence"] for row in body["rows"]] == [*SEQUENCES, COMBINED]
    regression = next(row for row in body["rows"] if row["sequence"] == "MOT17-11")
    mota = regression["metrics"]["MOTA"]
    assert mota["delta"] == pytest.approx(mota["gmc_enabled"] - mota["no_gmc"])
    assert mota["delta"] < 0
    evidence = body["regression"]
    assert evidence["sequence"] == "MOT17-11"
    assert evidence["delta"]["CLR_FP"] == 662
    assert evidence["delta"]["CLR_TP"] == 35
    assert evidence["output_boxes"] == {"no_gmc": 8954, "gmc_enabled": 9658}
    assert evidence["false_positive_tracks"]
    assert body["control_identical_to_no_gmc"] is True


@needs_results
def test_not_every_sequence_benefits_from_gmc(api):
    rows = _get(api, f"{BASE}/ablation").json()["rows"]
    deltas = {row["sequence"]: row["metrics"]["HOTA"]["delta"] for row in rows}
    assert deltas["MOT17-11"] < 0
    assert deltas["MOT17-10"] > 0


@needs_results
@pytest.mark.parametrize("figure", ["localization_errors", *FRAME_PANELS])
@pytest.mark.parametrize("experiment", ["gmc_enabled", "no_gmc"])
def test_allowlisted_mot17_figures_are_served(api, figure, experiment):
    response = _get(api, f"{BASE}/visualizations/{figure}?experiment={experiment}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    "figure", ["unknown", "..%2F..%2Fconfig%2Frun_config.json", "pedestrian_plot"]
)
def test_mot17_figures_reject_anything_else(api, figure):
    assert _get(api, f"{BASE}/visualizations/{figure}").status_code == 404


def test_mot17_experiment_query_rejects_unknown_labels(api):
    assert _get(api, f"{BASE}/error-profile?experiment=control").status_code == 422
    assert _get(api, f"{BASE}/visualizations/localization_errors?experiment=x").status_code == 422


def test_repository_rejects_unknown_experiment(tmp_path):
    with pytest.raises(UnknownExperiment):
        Mot17Repository(tmp_path).error_profile("control")


def test_missing_mot17_results_return_404_without_values(tmp_path):
    response = _get(create_app(tmp_path), f"{BASE}/summary")
    assert response.status_code == 404
    assert "unavailable" in response.json()["detail"]


def _copy_minimal_run(tmp_path: Path) -> Path:
    target = tmp_path / "mot17_tracking_analysis"
    for relative in (
        "config/dataset_inventory.json",
        "trackeval/metrics.json",
        "trackeval/botsort/pedestrian_detailed.csv",
        "gmc_fixed/config/run_status.json",
        "gmc_fixed/gmc/diagnostics.json",
        "gmc_fixed/trackeval/metrics.json",
        "gmc_fixed/trackeval/BoT-SORT_GMC_enabled/pedestrian_detailed.csv",
    ):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MOT17 / relative, destination)
    return target


@needs_results
def test_incomplete_primary_run_is_refused(tmp_path):
    target = _copy_minimal_run(tmp_path)
    path = target / "gmc_fixed/config/run_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    status["all_sequences_complete"] = False
    path.write_text(json.dumps(status), encoding="utf-8")
    _cached_json.cache_clear()
    with pytest.raises(ArtifactInvalid, match="did not complete"):
        Mot17Repository(tmp_path).summary()


@needs_results
def test_recorded_gmc_failure_is_refused(tmp_path):
    target = _copy_minimal_run(tmp_path)
    path = target / "gmc_fixed/gmc/diagnostics.json"
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    diagnostics["complete_without_failure"] = False
    path.write_text(json.dumps(diagnostics), encoding="utf-8")
    _cached_json.cache_clear()
    with pytest.raises(ArtifactInvalid, match="recorded a failure"):
        Mot17Repository(tmp_path).summary()


@needs_results
def test_disagreeing_saved_metrics_are_refused(tmp_path):
    target = _copy_minimal_run(tmp_path)
    path = target / "gmc_fixed/trackeval/metrics.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["metrics"]["COMBINED_SEQ"]["HOTA"] = 0.99
    path.write_text(json.dumps(saved), encoding="utf-8")
    _cached_json.cache_clear()
    with pytest.raises(ArtifactInvalid, match="disagree"):
        Mot17Repository(tmp_path).summary()


def test_csv_missing_columns_is_invalid(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ArtifactInvalid, match="required columns"):
        _cached_csv(path, ("a", "c"))
    with pytest.raises(ArtifactNotFound):
        _cached_csv(tmp_path / "missing.csv", ("a",))


def test_histograms_assign_edges_and_overflow():
    open_ended = _histogram([0.0, 0.5, 0.99, 1.0, 7.0], (0.0, 1.0, 2.0))
    assert [row["count"] for row in open_ended] == [3, 1, 1]
    assert open_ended[-1]["upper"] is None
    closed = _closed_histogram([0.0, 0.5, 1.0], (0.0, 0.5, 1.0))
    assert [row["count"] for row in closed] == [1, 2]
    assert sum(row["count"] for row in closed) == 3


def test_eth_figures_are_listed_and_served(api):
    body = _get(api, "/api/v1/datasets/eth/error-injections/figures").json()
    assert body["dataset"] == "eth"
    assert {item["figure_id"] for item in body["figures"]} == set(ETH_FIGURES)
    for figure_id in ETH_FIGURES:
        response = _get(api, f"/api/v1/datasets/eth/error-injections/figures/{figure_id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


def test_eth_figures_reject_unlisted_names(api):
    assert _get(api, "/api/v1/datasets/eth/error-injections/figures/secret").status_code == 404


def test_eth_figure_route_does_not_break_experiment_routes(api):
    response = _get(api, "/api/v1/datasets/eth/error-injections/identity_switch?severity=low")
    assert response.status_code == 200
    assert response.json()["experiment_id"] == "identity_switch"


def test_eth_dataset_audit_reports_saved_counts_without_test_metrics(api):
    body = _get(api, "/api/v1/datasets/eth/dataset-audit").json()
    assert body["issue_count"] == 0
    assert {row["split"] for row in body["split_summaries"]} == {"train", "val", "test"}
    assert "ade" not in json.dumps(body).lower().replace("fade", "")


def test_sealed_test_results_are_not_reachable(api):
    assert _get(
        api, "/api/v1/datasets/eth/prediction/baseline?split=test&model=lstm"
    ).status_code == 422


def test_new_endpoints_allow_the_local_frontend_origin(api):
    headers = {"Origin": "http://localhost:3000"}
    for path in (f"{BASE}/visualizations/unknown", "/api/v1/datasets/eth/dataset-audit"):
        response = _get(api, path, headers)
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

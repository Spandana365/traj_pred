import asyncio
import json
import statistics
from pathlib import Path

import httpx
import pytest

from trackvision.api.app import create_app
from trackvision.api.result_repository import (
    EXPERIMENT_IDS,
    SEVERITIES,
    ResultRepository,
    UnknownExperiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get(app, url: str) -> httpx.Response:
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(url)

    return asyncio.run(request())


def _options(app, url: str, headers: dict[str, str]) -> httpx.Response:
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.options(url, headers=headers)

    return asyncio.run(request())


@pytest.fixture
def api():
    return create_app(PROJECT_ROOT / "runs")


def test_health_endpoint(api):
    response = _get(api, "/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_dashboard_summary_uses_saved_validation_results(api):
    response = _get(api, "/api/v1/dashboard/summary?dataset=eth")

    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "eth"
    assert data["split"] == "val"
    assert data["candidate_window_count"] == 5422
    assert data["validation_scene_count"] == 7
    assert len(data["validation_scenes"]) == 7
    assert set(data["clean_baseline"]) == {"constant_velocity", "lstm"}
    assert data["lstm_model_configuration"]["hidden_size"] == 64
    assert "physical units unverified" in data["coordinate_units"]
    assert data["available_error_experiments"] == list(EXPERIMENT_IDS)


@pytest.mark.parametrize("model", ["constant_velocity", "lstm"])
def test_clean_baseline_endpoint(api, model):
    response = _get(
        api,
        f"/api/v1/datasets/eth/prediction/baseline?split=val&model={model}",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == model
    assert data["window_count"] == 5422
    assert len(data["per_scene"]) == 7
    assert len(data["ade_by_timestep"]) == 12
    assert set(data["metrics"]) == {"ade", "fde"}
    assert "physical units unverified" in data["coordinate_units"]


def test_all_error_injection_summaries_include_four_experiments_and_three_seeds(api):
    response = _get(api, "/api/v1/datasets/eth/error-injections")

    assert response.status_code == 200
    data = response.json()
    assert [item["experiment_id"] for item in data["experiments"]] == list(
        EXPERIMENT_IDS
    )
    for experiment in data["experiments"]:
        assert [item["severity"] for item in experiment["severities"]] == list(
            SEVERITIES
        )
        for severity in experiment["severities"]:
            assert severity["seed_ids"] == [42, 43, 44]
            assert severity["corrupted_metrics"]["ade"]["population_std"] is not None
            assert severity["windows"]["candidate_count"] == 5422
            assert severity["coordinate_units"]


@pytest.mark.parametrize("experiment", EXPERIMENT_IDS)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_experiment_detail_normalizes_every_severity(api, experiment, severity):
    response = _get(
        api,
        f"/api/v1/datasets/eth/error-injections/{experiment}?severity={severity}",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == experiment
    assert data["severity"] == severity
    assert data["selected_seed"] is None
    assert data["seed_ids"] == [42, 43, 44]
    assert data["global_clean_metrics"]["ade"]["mean"] > 0
    assert data["matched_clean_metrics"]["ade"]["mean"] > 0
    assert data["corrupted_metrics"]["ade"]["mean"] > 0
    assert data["percentage_degradation"]["ade"]["mean"] is not None
    assert len(data["per_scene"]) == 7
    assert len(data["per_timestep"]) == 12
    assert data["metadata"]["protocol"]["split_evaluated"] == "val"


def test_dropout_distinguishes_global_clean_from_matched_clean(api):
    data = _get(
        api,
        "/api/v1/datasets/eth/error-injections/detection_dropout?severity=high",
    ).json()

    assert data["global_clean_metrics"] != data["matched_clean_metrics"]
    assert data["windows"]["candidate_count"] == 5422
    assert "actual_missing_percentage_unique" in data["windows"]["affected_statistics"]


@pytest.mark.parametrize("severity", SEVERITIES)
def test_fragmentation_uses_exact_matched_survivors(api, severity):
    data = _get(
        api,
        f"/api/v1/datasets/eth/error-injections/track_fragmentation?severity={severity}",
    ).json()

    assert data["matched_clean_metrics"] == data["corrupted_metrics"]
    assert data["degradation"]["ade"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert data["degradation"]["fde"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert data["windows"]["evaluated_count"]["mean"] < 5422
    assert data["windows"]["retention_percentage"]["mean"] < 100


def test_identity_switch_returns_matched_clean_and_corrupted_comparison(api):
    data = _get(
        api,
        "/api/v1/datasets/eth/error-injections/identity_switch?severity=low",
    ).json()

    assert data["matched_clean_metrics"] is not None
    assert data["corrupted_metrics"] != data["matched_clean_metrics"]
    assert data["windows"]["affected_statistics"]["affected_window_count"]["mean"] > 0
    assert data["per_scene"]["HOTEL"]["matched_clean"] is not None


def test_seed_query_returns_single_seed_without_claiming_seed_sd(api):
    response = _get(
        api,
        "/api/v1/datasets/eth/error-injections/localization_noise"
        "?severity=low&seed=42",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["selected_seed"] == 42
    assert data["seed_ids"] == [42]
    assert data["corrupted_metrics"]["ade"]["population_std"] is None


def test_corrupted_metrics_are_population_statistics_of_saved_seed_reports(api):
    data = _get(
        api,
        "/api/v1/datasets/eth/error-injections/localization_noise?severity=low",
    ).json()
    raw = []
    for seed in (42, 43, 44):
        path = (
            PROJECT_ROOT
            / "runs"
            / "error_injection"
            / "localization_noise"
            / "per_seed"
            / f"seed-{seed}.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        raw.append(report["severities"]["low"]["metrics"]["lstm"]["ade"])

    assert data["corrupted_metrics"]["ade"]["mean"] == pytest.approx(
        statistics.mean(raw)
    )
    assert data["corrupted_metrics"]["ade"]["population_std"] == pytest.approx(
        statistics.pstdev(raw)
    )


def test_invalid_experiment_severity_and_seed_are_rejected(api):
    invalid_experiment = _get(
        api, "/api/v1/datasets/eth/error-injections/not-an-experiment"
    )
    invalid_severity = _get(
        api,
        "/api/v1/datasets/eth/error-injections/localization_noise?severity=extreme",
    )
    invalid_seed = _get(
        api,
        "/api/v1/datasets/eth/error-injections/localization_noise?seed=99",
    )

    assert invalid_experiment.status_code == 404
    assert invalid_severity.status_code == 404
    assert invalid_seed.status_code == 422


def test_missing_artifact_returns_404(tmp_path):
    app = create_app(tmp_path)

    response = _get(
        app,
        "/api/v1/datasets/eth/prediction/baseline?split=val&model=lstm",
    )

    assert response.status_code == 404
    assert "unavailable" in response.json()["detail"]


@pytest.mark.parametrize(
    ("experiment", "severity"),
    [(experiment, severity) for experiment in EXPERIMENT_IDS for severity in SEVERITIES],
)
def test_allowlisted_visualizations_are_served(api, experiment, severity):
    image = _get(
        api,
        f"/api/v1/datasets/eth/error-injections/{experiment}"
        f"/visualizations/{severity}",
    )

    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_visualization_endpoint_rejects_arbitrary_paths(api):
    invalid_experiment = _get(
        api,
        "/api/v1/datasets/eth/error-injections/..%2F..%2FREADME.md"
        "/visualizations/low",
    )

    assert invalid_experiment.status_code == 404


def test_repository_rejects_non_allowlisted_visualization_path(tmp_path):
    repository = ResultRepository(tmp_path)

    with pytest.raises(UnknownExperiment):
        repository.visualization_path("../../README.md", "low")


def test_cors_allows_only_the_local_frontend_origin(api):
    allowed = _options(
        api,
        "/api/v1/health",
        {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    rejected = _options(
        api,
        "/api/v1/health",
        {
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in rejected.headers

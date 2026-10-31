"""Read-only FastAPI app serving saved TrackVision reports.

Install the optional API dependencies in the supported Python 3.12 environment:
``python -m pip install -e ".[api]"``.
From the project root, start with:
``python -m uvicorn trackvision.api.app:app --app-dir src --reload``
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from trackvision.api import __version__
from trackvision.api.mot17_repository import Mot17Repository
from trackvision.api.result_repository import (
    ArtifactInvalid,
    ArtifactNotFound,
    ResultRepository,
    UnknownExperiment,
)
from trackvision.api.schemas import (
    BaselineResponse,
    DashboardSummaryResponse,
    ErrorExperimentsResponse,
    EthDatasetAuditResponse,
    EthFiguresResponse,
    ExperimentDetailResponse,
    HealthResponse,
    Mot17AblationResponse,
    Mot17ConfigurationResponse,
    Mot17ErrorProfileResponse,
    Mot17GmcResponse,
    Mot17SequencesResponse,
    Mot17SummaryResponse,
)

logger = logging.getLogger(__name__)


def create_app(results_root=None) -> FastAPI:
    """Create the API app; ``results_root`` is useful for isolated tests."""
    repository = ResultRepository(results_root)
    mot17 = Mot17Repository(repository.results_root)
    application = FastAPI(
        title="TrackVision Results API",
        version=__version__,
        description="Read-only access to saved validation and error-injection reports.",
    )
    application.state.result_repository = repository
    application.state.mot17_repository = mot17
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=[],
    )

    @application.exception_handler(ArtifactNotFound)
    async def artifact_not_found_handler(_request, _error):
        del _request, _error
        return JSONResponse(
            status_code=404,
            content={"detail": "A required saved result artifact is unavailable."},
        )

    @application.exception_handler(ArtifactInvalid)
    async def artifact_invalid_handler(_request, error):
        del _request
        logger.error("Saved result artifacts failed validation: %s", error)
        return JSONResponse(
            status_code=500,
            content={"detail": "Saved result artifacts are invalid."},
        )

    @application.exception_handler(UnknownExperiment)
    async def unknown_experiment_handler(_request, _error):
        del _request, _error
        return JSONResponse(
            status_code=404,
            content={"detail": "The requested experiment or severity is unavailable."},
        )

    @application.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @application.get(
        "/api/v1/dashboard/summary", response_model=DashboardSummaryResponse
    )
    def dashboard_summary(
        dataset: str = Query(default="eth"),
    ) -> DashboardSummaryResponse:
        return repository.dashboard_summary(dataset)

    @application.get(
        "/api/v1/datasets/eth/prediction/baseline",
        response_model=BaselineResponse,
    )
    def prediction_baseline(
        split: Literal["val"] = Query(default="val"),
        model: Literal["constant_velocity", "lstm"] = Query(default="lstm"),
    ) -> BaselineResponse:
        if split != "val":
            raise HTTPException(status_code=404, detail="Validation split unavailable.")
        return repository.baseline(model)

    @application.get(
        "/api/v1/datasets/eth/error-injections",
        response_model=ErrorExperimentsResponse,
    )
    def all_error_injections() -> ErrorExperimentsResponse:
        return repository.all_error_summaries("eth")

    @application.get(
        "/api/v1/datasets/eth/dataset-audit",
        response_model=EthDatasetAuditResponse,
    )
    def eth_dataset_audit() -> EthDatasetAuditResponse:
        return repository.eth_dataset_audit()

    # Declared before the {experiment} routes so "figures" is not read as an id.
    @application.get(
        "/api/v1/datasets/eth/error-injections/figures",
        response_model=EthFiguresResponse,
    )
    def eth_figures() -> EthFiguresResponse:
        return repository.eth_figures()

    @application.get(
        "/api/v1/datasets/eth/error-injections/figures/{figure_id}",
        response_class=FileResponse,
    )
    def eth_figure(figure_id: str) -> FileResponse:
        return FileResponse(repository.eth_figure_path(figure_id), media_type="image/png")

    @application.get(
        "/api/v1/datasets/eth/error-injections/{experiment}",
        response_model=ExperimentDetailResponse,
    )
    def error_injection_detail(
        experiment: str,
        severity: str = Query(default="low"),
        seed: int | None = Query(default=None),
    ) -> ExperimentDetailResponse:
        if seed is not None and seed not in (42, 43, 44):
            raise HTTPException(
                status_code=422,
                detail="seed must be one of 42, 43, or 44.",
            )
        return repository.experiment_detail(experiment, severity, seed)

    @application.get(
        "/api/v1/datasets/eth/error-injections/{experiment}/visualizations/{severity}",
        response_class=FileResponse,
    )
    def error_injection_visualization(experiment: str, severity: str) -> FileResponse:
        path = repository.visualization_path(experiment, severity)
        return FileResponse(path, media_type="image/png")

    @application.get(
        "/api/v1/datasets/mot17/tracking/summary",
        response_model=Mot17SummaryResponse,
    )
    def mot17_summary() -> Mot17SummaryResponse:
        return mot17.summary()

    @application.get(
        "/api/v1/datasets/mot17/tracking/sequences",
        response_model=Mot17SequencesResponse,
    )
    def mot17_sequences() -> Mot17SequencesResponse:
        return mot17.sequences()

    @application.get(
        "/api/v1/datasets/mot17/tracking/configuration",
        response_model=Mot17ConfigurationResponse,
    )
    def mot17_configuration() -> Mot17ConfigurationResponse:
        return mot17.configuration()

    @application.get(
        "/api/v1/datasets/mot17/tracking/gmc",
        response_model=Mot17GmcResponse,
    )
    def mot17_gmc() -> Mot17GmcResponse:
        return mot17.gmc()

    @application.get(
        "/api/v1/datasets/mot17/tracking/error-profile",
        response_model=Mot17ErrorProfileResponse,
    )
    def mot17_error_profile(
        experiment: Literal["gmc_enabled", "no_gmc"] = Query(default="gmc_enabled"),
    ) -> Mot17ErrorProfileResponse:
        return mot17.error_profile(experiment)

    @application.get(
        "/api/v1/datasets/mot17/tracking/ablation",
        response_model=Mot17AblationResponse,
    )
    def mot17_ablation() -> Mot17AblationResponse:
        return mot17.ablation()

    @application.get(
        "/api/v1/datasets/mot17/tracking/visualizations/{figure}",
        response_class=FileResponse,
    )
    def mot17_visualization(
        figure: str,
        experiment: Literal["gmc_enabled", "no_gmc"] = Query(default="gmc_enabled"),
    ) -> FileResponse:
        path = mot17.visualization_path(experiment, figure)
        return FileResponse(path, media_type="image/png")

    return application


app = create_app()

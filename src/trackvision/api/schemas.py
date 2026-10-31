"""Pydantic response models for the read-only results API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class MetricStats(BaseModel):
    mean: float
    population_std: float | None = None


class MetricPair(BaseModel):
    ade: MetricStats
    fde: MetricStats


class SceneMetrics(BaseModel):
    ade: MetricStats
    fde: MetricStats
    window_count: MetricStats | None = None


class BaselineResponse(BaseModel):
    dataset: str
    split: str
    model: str
    metrics: dict[str, float]
    window_count: int
    per_scene: dict[str, SceneMetrics]
    ade_by_timestep: list[dict[str, float | int]]
    coordinate_units: str


class DashboardSummaryResponse(BaseModel):
    dataset: str
    split: str
    candidate_window_count: int
    validation_scene_count: int
    validation_scenes: list[str]
    clean_baseline: dict[str, dict[str, float]]
    available_error_experiments: list[str]
    lstm_model_configuration: dict[str, Any]
    coordinate_units: str


class WindowStatistics(BaseModel):
    candidate_count: int
    evaluated_count: MetricStats
    retained_count: MetricStats
    unusable_count: MetricStats | None = None
    retention_percentage: MetricStats
    affected_statistics: dict[str, MetricStats]


class SceneComparison(BaseModel):
    matched_clean: SceneMetrics | None = None
    corrupted: SceneMetrics


class TimestepComparison(BaseModel):
    timestep: int
    matched_clean_ade: MetricStats | None = None
    corrupted_ade: MetricStats
    degradation_ade: MetricStats | None = None


class SeveritySummary(BaseModel):
    severity: str
    parameters: dict[str, Any]
    global_clean_metrics: MetricPair
    matched_clean_metrics: MetricPair | None = None
    corrupted_metrics: MetricPair
    degradation: MetricPair
    percentage_degradation: MetricPair
    windows: WindowStatistics
    seed_ids: list[int]
    coordinate_units: str


class ErrorExperimentSummary(BaseModel):
    experiment_id: str
    severities: list[SeveritySummary]


class ErrorExperimentsResponse(BaseModel):
    dataset: str
    split: str
    experiments: list[ErrorExperimentSummary]


class ExperimentMetadata(BaseModel):
    protocol: dict[str, Any]
    description: str | None = None


class ExperimentDetailResponse(SeveritySummary):
    experiment_id: str
    metadata: ExperimentMetadata
    per_scene: dict[str, SceneComparison]
    per_timestep: list[TimestepComparison]
    selected_seed: int | None = None


class EthFigure(BaseModel):
    figure_id: str
    title: str
    group: str


class EthFiguresResponse(BaseModel):
    dataset: str
    figures: list[EthFigure]


class EthDatasetAuditResponse(BaseModel):
    dataset: str
    configuration: dict[str, Any]
    split_summaries: list[dict[str, Any]]
    issue_count: int
    scene_overlap_checks: int
    identical_observation_rows_across_splits: int
    source: str


class Mot17SummaryResponse(BaseModel):
    dataset: str
    split: str
    role: str
    primary_experiment: str
    comparison_experiment: str
    sequence_count: int
    total_frames: int
    metrics: dict[str, float]
    comparison_metrics: dict[str, float]
    all_sequences_complete: bool
    gmc_failed_calls: int
    gmc_calls: int
    coordinate_units: str
    scope_note: str
    provenance: list[str]


class Mot17SequencesResponse(BaseModel):
    dataset: str
    split: str
    primary_experiment: str
    comparison_experiment: str
    canonical_variant: str
    variant_policy: str
    total_frames: int
    camera_class_rule: str
    sequences: list[dict[str, Any]]
    provenance: list[str]


class Mot17ConfigurationResponse(BaseModel):
    experiment: str
    training_performed: bool
    detector: dict[str, Any]
    weights_sha256: str
    person_class_id: int
    tracker_implementation: str
    tracker_settings: dict[str, Any]
    settings_changed_from_comparison: bool
    gmc: dict[str, Any]
    evaluation: dict[str, Any]
    provenance: list[str]


class Mot17GmcResponse(BaseModel):
    experiment: str
    method: str
    totals: dict[str, Any]
    complete_without_failure: bool
    control_identical_to_no_gmc: bool
    sequences: list[dict[str, Any]]
    provenance: list[str]


class Mot17ErrorProfileResponse(BaseModel):
    dataset: str
    experiment: str
    kind: str
    coordinate_units: str
    matching: dict[str, Any]
    definitions: dict[str, str]
    limitations: list[str]
    overall: dict[str, Any]
    coverage_counts: dict[str, int]
    sequences: list[dict[str, Any]]
    iou_histogram: list[dict[str, Any]]
    center_error_histogram: list[dict[str, Any]]
    coverage_histogram: list[dict[str, Any]]
    gap_histogram: list[dict[str, Any]]
    track_duration_histogram: list[dict[str, Any]]
    track_duration_seconds: dict[str, Any]
    availability: list[dict[str, Any]]
    availability_by_sequence: list[dict[str, Any]]
    survival: list[dict[str, Any]]
    provenance: list[str]


class Mot17AblationResponse(BaseModel):
    primary_experiment: str
    comparison_experiment: str
    control_identical_to_no_gmc: bool
    rows: list[dict[str, Any]]
    regression: dict[str, Any]
    provenance: list[str]

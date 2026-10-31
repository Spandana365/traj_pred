'use client'

import { useEffect, useState } from 'react'

export type MetricStats = {
  mean: number
  population_std?: number | null
}

export type MetricPair = {
  ade: MetricStats
  fde: MetricStats
}

export type SceneMetrics = {
  ade: MetricStats
  fde: MetricStats
  window_count?: MetricStats | null
}

export type BaselineResponse = {
  dataset: string
  split: string
  model: string
  metrics: { ade: number; fde: number }
  window_count: number
  per_scene: Record<string, SceneMetrics>
  ade_by_timestep: Array<{ timestep: number; ade: number }>
  coordinate_units: string
}

export type DashboardSummaryResponse = {
  dataset: string
  split: string
  candidate_window_count: number
  validation_scene_count: number
  validation_scenes: string[]
  clean_baseline: Record<string, { ade: number; fde: number }>
  available_error_experiments: string[]
  lstm_model_configuration: Record<string, unknown>
  coordinate_units: string
}

export type WindowStatistics = {
  candidate_count: number
  evaluated_count: MetricStats
  retained_count: MetricStats
  unusable_count?: MetricStats | null
  retention_percentage: MetricStats
  affected_statistics: Record<string, MetricStats>
}

export type SeveritySummary = {
  severity: string
  parameters: Record<string, unknown>
  global_clean_metrics: MetricPair
  matched_clean_metrics?: MetricPair | null
  corrupted_metrics: MetricPair
  degradation: MetricPair
  percentage_degradation: MetricPair
  windows: WindowStatistics
  seed_ids: number[]
  coordinate_units: string
}

export type ErrorExperimentSummary = {
  experiment_id: string
  severities: SeveritySummary[]
}

export type ErrorExperimentsResponse = {
  dataset: string
  split: string
  experiments: ErrorExperimentSummary[]
}

export type SceneComparison = {
  matched_clean?: SceneMetrics | null
  corrupted: SceneMetrics
}

export type TimestepComparison = {
  timestep: number
  matched_clean_ade?: MetricStats | null
  corrupted_ade: MetricStats
  degradation_ade?: MetricStats | null
}

export type ExperimentDetailResponse = SeveritySummary & {
  experiment_id: ExperimentId
  metadata: {
    protocol: Record<string, unknown>
    description?: string | null
  }
  per_scene: Record<string, SceneComparison>
  per_timestep: TimestepComparison[]
  selected_seed?: number | null
}

export type ExperimentId =
  | 'localization_noise'
  | 'detection_dropout'
  | 'track_fragmentation'
  | 'identity_switch'

export type SeverityId = 'low' | 'medium' | 'high'
export type BaselineModel = 'lstm' | 'constant_velocity'

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_TRACKVISION_API_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  'http://localhost:8000'
).replace(/\/+$/, '')

export function getApiBaseUrl(): string {
  return API_BASE_URL
}

export function getApiUrl(path: string): string {
  return `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`
}

export async function fetchApi<T>(path: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(getApiUrl(path), { cache: 'no-store' })
  } catch (error) {
    throw new Error(
      `Could not connect to the TrackVision API at ${API_BASE_URL}.`,
      { cause: error },
    )
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // Keep the HTTP status when an error response is not JSON.
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export function getDashboardSummary(): Promise<DashboardSummaryResponse> {
  return fetchApi('/api/v1/dashboard/summary?dataset=eth')
}

export function getBaseline(
  model: BaselineModel,
): Promise<BaselineResponse> {
  return fetchApi(
    `/api/v1/datasets/eth/prediction/baseline?split=val&model=${model}`,
  )
}

export function getErrorExperiments(): Promise<ErrorExperimentsResponse> {
  return fetchApi('/api/v1/datasets/eth/error-injections')
}

export function getExperimentDetail(
  experiment: ExperimentId,
  severity: SeverityId,
): Promise<ExperimentDetailResponse> {
  return fetchApi(
    `/api/v1/datasets/eth/error-injections/${experiment}?severity=${severity}`,
  )
}

export function getVisualizationUrl(
  experiment: ExperimentId,
  severity: SeverityId,
): string {
  return getApiUrl(
    `/api/v1/datasets/eth/error-injections/${experiment}/visualizations/${severity}`,
  )
}

export function getEthFigureUrl(figureId: string): string {
  return getApiUrl(`/api/v1/datasets/eth/error-injections/figures/${figureId}`)
}

export type EthFiguresResponse = {
  dataset: string
  figures: Array<{ figure_id: string; title: string; group: string }>
}

export type EthDatasetAuditResponse = {
  dataset: string
  configuration: Record<string, number>
  split_summaries: Array<{
    split: string
    candidate_window_count: number
    observations: number
    files_expected: number
    files_present: number
  }>
  issue_count: number
  scene_overlap_checks: number
  identical_observation_rows_across_splits: number
  source: string
}

export type Mot17ExperimentKey = 'gmc_enabled' | 'no_gmc'
export type Mot17Metrics = Record<string, number>

export type Mot17SummaryResponse = {
  dataset: string
  split: string
  role: string
  primary_experiment: string
  comparison_experiment: string
  sequence_count: number
  total_frames: number
  metrics: Mot17Metrics
  comparison_metrics: Mot17Metrics
  all_sequences_complete: boolean
  gmc_failed_calls: number
  gmc_calls: number
  coordinate_units: string
  scope_note: string
  provenance: string[]
}

export type Mot17Sequence = {
  sequence: string
  canonical_directory: string
  frames: number
  frame_rate: number
  width: number
  height: number
  gt_rows: number
  evaluated_gt_boxes: number
  evaluated_gt_identities: number
  variants_agree: boolean
  camera_class: string
  metrics: Mot17Metrics
  comparison_metrics: Mot17Metrics
}

export type Mot17SequencesResponse = {
  dataset: string
  split: string
  primary_experiment: string
  comparison_experiment: string
  canonical_variant: string
  variant_policy: string
  total_frames: number
  camera_class_rule: string
  sequences: Mot17Sequence[]
  provenance: string[]
}

export type Mot17ConfigurationResponse = {
  experiment: string
  training_performed: boolean
  detector: Record<string, string | number | boolean>
  weights_sha256: string
  person_class_id: number
  tracker_implementation: string
  tracker_settings: Record<string, string | number | boolean>
  settings_changed_from_comparison: boolean
  gmc: Record<string, string>
  evaluation: Record<string, string | number | boolean>
  provenance: string[]
}

export type Mot17GmcSequence = {
  sequence: string
  calls: number
  successful_calls: number
  failed_calls: number
  estimated_warps: number
  first_frame_identity_warps: number
  non_first_identity_warps: number
  camera_class: string
  median_warp_center_motion_pixels: number
  median_phase_motion_pixels: number
  correlation_x: number | null
  correlation_y: number | null
  share_of_frame_pairs_improved: number
  static_objects: {
    static_objects: number
    anchored_motion_pixels: Record<
      string,
      {
        pairs: number
        median_raw: number | null
        median_compensated: number | null
      }
    >
  }
}

export type Mot17GmcResponse = {
  experiment: string
  method: string
  totals: {
    calls_expected: number
    calls: number
    successful_calls: number
    failed_calls: number
    estimated_warps: number
    first_frame_identity_warps: number
    non_first_identity_warps: number
    ultralytics_gmc_warnings: Record<string, number>
  }
  complete_without_failure: boolean
  control_identical_to_no_gmc: boolean
  sequences: Mot17GmcSequence[]
  provenance: string[]
}

export type HistogramBin = {
  lower: number
  upper: number | null
  count: number
}

export type DistributionSummary = {
  count: number
  mean?: number | null
  std?: number | null
  median: number | null
  p5?: number | null
  p95: number | null
  max?: number | null
}

export type DetectorCounts = {
  true_positives: number
  false_positives: number
  false_negatives: number
  recall: number
  precision: number
}

export type Mot17ErrorProfileResponse = {
  dataset: string
  experiment: string
  kind: string
  coordinate_units: string
  matching: Record<string, unknown>
  definitions: Record<string, string>
  limitations: string[]
  overall: {
    true_positives: number
    false_positives: number
    false_negatives: number
    identity_switches: number
    fragmentations: number
    recall: number
    precision: number
    'detector_conf_0.25': DetectorCounts
    'detector_conf_0.10': DetectorCounts
    iou: DistributionSummary
    center_error_px: DistributionSummary
    center_error_over_height: DistributionSummary
    interior_gap_frames: DistributionSummary
    interior_gap_seconds: DistributionSummary
    per_pedestrian_coverage: DistributionSummary
  }
  coverage_counts: Record<string, number>
  sequences: Array<{
    sequence: string
    recall: number
    precision: number
    median_iou: number
    median_center_error_px: number
    p95_center_error_px: number
    median_center_error_over_height: number
    median_track_span_seconds: number
  }>
  iou_histogram: HistogramBin[]
  center_error_histogram: HistogramBin[]
  coverage_histogram: HistogramBin[]
  gap_histogram: HistogramBin[]
  track_duration_histogram: HistogramBin[]
  track_duration_seconds: DistributionSummary
  availability: Array<{
    window_seconds: number
    possible_windows: number
    available_windows: number
    availability: number | null
  }>
  availability_by_sequence: Array<{
    sequence: string
    window_seconds: number
    possible_windows: number
    available_windows: number
    availability: number | null
  }>
  survival: Array<{
    sequence: string
    horizon_seconds: number
    survival: number
  }>
  provenance: string[]
}

export type Mot17ComparedValue = {
  no_gmc: number
  gmc_enabled: number
  delta: number
}

export type Mot17AblationResponse = {
  primary_experiment: string
  comparison_experiment: string
  control_identical_to_no_gmc: boolean
  rows: Array<{
    sequence: string
    camera_class: string | null
    metrics: Record<string, Mot17ComparedValue>
  }>
  regression: {
    sequence: string
    output_boxes: Record<Mot17ExperimentKey, number>
    median_track_length_frames: Record<Mot17ExperimentKey, number>
    delta: Record<string, number>
    warp_correlation_x: number | null
    warp_correlation_y: number | null
    share_of_frame_pairs_improved: number
    false_positive_tracks: Array<{
      track_id: number
      false_positive_frames: number
      total_frames: number
      median_confidence: number
      raw_travel_pixels: number
      stabilized_travel_pixels: number
    }>
  }
  provenance: string[]
}

export function getMot17FigureUrl(
  figure: string,
  experiment: Mot17ExperimentKey = 'gmc_enabled',
): string {
  return getApiUrl(
    `/api/v1/datasets/mot17/tracking/visualizations/${figure}?experiment=${experiment}`,
  )
}

export type ApiResource<T> = {
  data: T | null
  isLoading: boolean
  error: string | null
}

type KeyedApiResource<T> = ApiResource<T> & { path: string }

export function useApiResource<T>(
  path: string,
): ApiResource<T> {
  const [resource, setResource] = useState<KeyedApiResource<T>>({
    path,
    data: null,
    isLoading: true,
    error: null,
  })

  useEffect(() => {
    let active = true
    setResource({ path, data: null, isLoading: true, error: null })
    fetchApi<T>(path)
      .then((data) => {
        if (active) setResource({ path, data, isLoading: false, error: null })
      })
      .catch((error: unknown) => {
        if (!active) return
        setResource({
          path,
          data: null,
          isLoading: false,
          error:
            error instanceof Error
              ? error.message
              : 'The TrackVision API request failed.',
        })
      })
    return () => {
      active = false
    }
  }, [path])

  return resource.path === path
    ? resource
    : { data: null, isLoading: true, error: null }
}

export function useApiHealth(): ApiResource<{ status: string; version: string }> {
  return useApiResource('/api/v1/health')
}

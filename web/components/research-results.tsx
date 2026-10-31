'use client'

import { useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  AppShell,
  EmptyState,
  MetricGrid,
  PageHeader,
  Panel,
  Status,
  WorkflowFooter,
} from '@/components/trackvision-app'
import {
  Mot17AnalysisSection,
  Mot17DatasetPanels,
  Mot17ExperimentsSection,
  Mot17OverviewSummary,
  Mot17TrackingResults,
  Mot17VisualGallery,
} from '@/components/mot17-results'
import {
  Cell,
  DataTable,
  DatasetSection,
  ResourceState,
  SavedFigure,
  ScienceNote,
  UnitNote,
  fixed,
  stat,
} from '@/components/result-primitives'
import {
  type BaselineModel,
  type BaselineResponse,
  type DashboardSummaryResponse,
  type ErrorExperimentsResponse,
  type EthDatasetAuditResponse,
  type EthFiguresResponse,
  type ExperimentDetailResponse,
  type ExperimentId,
  type SeverityId,
  type SeveritySummary,
  getEthFigureUrl,
  getVisualizationUrl,
  useApiResource,
} from '@/lib/trackvision-api'

const EXPERIMENTS: Array<{ id: ExperimentId; label: string }> = [
  { id: 'localization_noise', label: 'Localization noise' },
  { id: 'detection_dropout', label: 'Detection dropout' },
  { id: 'track_fragmentation', label: 'Track fragmentation' },
  { id: 'identity_switch', label: 'Identity switches' },
]
const SEVERITIES: SeverityId[] = ['low', 'medium', 'high']

function severityById(
  response: ErrorExperimentsResponse,
  experiment: ExperimentId,
  severity: SeverityId,
): SeveritySummary | undefined {
  return response.experiments
    .find((item) => item.experiment_id === experiment)
    ?.severities.find((item) => item.severity === severity)
}

function experimentLabel(id: string): string {
  return EXPERIMENTS.find((item) => item.id === id)?.label ?? id
}

function ModelBaselinePanel({
  model,
  title,
}: {
  model: BaselineModel
  title: string
}) {
  const baseline = useApiResource<BaselineResponse>(
    `/api/v1/datasets/eth/prediction/baseline?split=val&model=${model}`,
  )
  return (
    <ResourceState resource={baseline} label={`${title} validation baseline`}>
      {(data) => (
        <Panel title={title} eyebrow="Saved ETH validation report">
          <MetricGrid
            items={[
              { label: 'ADE', value: fixed(data.metrics.ade) },
              { label: 'FDE', value: fixed(data.metrics.fde) },
              { label: 'Windows', value: data.window_count.toLocaleString() },
            ]}
          />
          <UnitNote units={data.coordinate_units} />
        </Panel>
      )}
    </ResourceState>
  )
}

export function OverviewResultsPage() {
  const summary = useApiResource<DashboardSummaryResponse>(
    '/api/v1/dashboard/summary?dataset=eth',
  )
  const experiments = useApiResource<ErrorExperimentsResponse>(
    '/api/v1/datasets/eth/error-injections',
  )

  return (
    <AppShell stage="overview">
      <PageHeader
        eyebrow="Research platform / saved results"
        title="TrackVision results overview"
        description="Read-only ETH prediction results and MOT17 tracking results, kept as separate studies."
        stage="01"
      />
      <DatasetSection
        dataset="ETH"
        title="Trajectory prediction"
        detail="Validation baselines and controlled error-injection experiments."
      />
      <ResourceState resource={summary} label="dashboard summary">
        {(data) => (
          <>
            <div className="mb-5">
              <MetricGrid
                items={[
                  {
                    label: 'Validation candidate windows',
                    value: data.candidate_window_count.toLocaleString(),
                    detail: `${data.split.toUpperCase()} split · ${data.dataset.toUpperCase()}`,
                  },
                  {
                    label: 'Validation scenes',
                    value: String(data.validation_scene_count),
                    detail: data.validation_scenes.join(', '),
                  },
                  {
                    label: 'LSTM ADE / FDE',
                    value: `${fixed(data.clean_baseline.lstm?.ade)} / ${fixed(data.clean_baseline.lstm?.fde)}`,
                    detail: 'Clean validation baseline',
                  },
                  {
                    label: 'Constant velocity ADE / FDE',
                    value: `${fixed(data.clean_baseline.constant_velocity?.ade)} / ${fixed(data.clean_baseline.constant_velocity?.fde)}`,
                    detail: 'Clean validation baseline',
                  },
                ]}
              />
            </div>
            <Panel title="Validation dataset" eyebrow="Backend-reported information">
              <div className="flex flex-wrap gap-2">
                <Status label={data.dataset.toUpperCase()} kind="success" />
                <Status label={`${data.split.toUpperCase()} split`} />
                <Status
                  label={`${data.available_error_experiments.length} error experiments`}
                />
              </div>
              <UnitNote units={data.coordinate_units} />
            </Panel>
          </>
        )}
      </ResourceState>
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <ModelBaselinePanel model="lstm" title="LSTM baseline" />
        <ModelBaselinePanel
          model="constant_velocity"
          title="Constant-velocity baseline"
        />
      </div>
      <div className="mt-5">
        <ResourceState resource={experiments} label="error-experiment summaries">
          {(data) => <ExperimentSummaryTable data={data} />}
        </ResourceState>
      </div>
      <Mot17OverviewSummary />
      <WorkflowFooter stage="overview" />
    </AppShell>
  )
}

function ExperimentSummaryTable({
  data,
}: {
  data: ErrorExperimentsResponse
}) {
  return (
    <Panel
      title="Saved error-injection results"
      eyebrow="Per-severity seed means and population standard deviations"
    >
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-left text-xs">
          <thead className="border-b border-white/[.08] font-mono text-[10px] uppercase tracking-wider text-slate-600">
            <tr>
              {[
                'Experiment',
                'Severity',
                'Parameters',
                'Matched-clean ADE',
                'Corrupted ADE',
                'Matched-clean FDE',
                'Corrupted FDE',
                'Retained windows',
              ].map((label) => (
                <th key={label} className="pb-3 pr-4 font-normal">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.experiments.flatMap((experiment) =>
              experiment.severities.map((result) => (
                <tr
                  key={`${experiment.experiment_id}-${result.severity}`}
                  className="border-b border-white/[.05]"
                >
                  <td className="py-3 pr-4 text-slate-300">
                    {experimentLabel(experiment.experiment_id)}
                  </td>
                  <td className="py-3 pr-4 capitalize text-cyan-300">
                    {result.severity}
                  </td>
                  <td className="py-3 pr-4 text-slate-500">
                    {Object.entries(result.parameters)
                      .map(([key, value]) => `${key}: ${String(value)}`)
                      .join(', ') || 'Not reported'}
                  </td>
                  <td className="py-3 pr-4 font-mono text-slate-400">
                    {stat(result.matched_clean_metrics?.ade)}
                  </td>
                  <td className="py-3 pr-4 font-mono text-slate-300">
                    {stat(result.corrupted_metrics.ade)}
                  </td>
                  <td className="py-3 pr-4 font-mono text-slate-400">
                    {stat(result.matched_clean_metrics?.fde)}
                  </td>
                  <td className="py-3 pr-4 font-mono text-slate-300">
                    {stat(result.corrupted_metrics.fde)}
                  </td>
                  <td className="py-3 pr-4 font-mono text-slate-400">
                    {stat(result.windows.retained_count)}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

export function DataResultsPage() {
  const summary = useApiResource<DashboardSummaryResponse>(
    '/api/v1/dashboard/summary?dataset=eth',
  )
  const baseline = useApiResource<BaselineResponse>(
    '/api/v1/datasets/eth/prediction/baseline?split=val&model=lstm',
  )
  const audit = useApiResource<EthDatasetAuditResponse>(
    '/api/v1/datasets/eth/dataset-audit',
  )

  return (
    <AppShell stage="data">
      <PageHeader
        eyebrow="Stage 02 / Datasets"
        title="Dataset results"
        description="Dataset facts reported by the saved research artifacts. ETH supports prediction experiments and MOT17 supports tracking evaluation."
        stage="02"
      />
      <DatasetSection
        dataset="ETH"
        title="ETH prediction dataset"
        detail="Used for trajectory-prediction baselines and controlled error injection."
      />
      <ResourceState resource={summary} label="ETH validation dataset">
        {(data) => (
          <>
            <MetricGrid
              items={[
                { label: 'Dataset', value: data.dataset.toUpperCase() },
                { label: 'Split', value: data.split.toUpperCase() },
                {
                  label: 'Candidate windows',
                  value: data.candidate_window_count.toLocaleString(),
                },
                {
                  label: 'Validation scenes',
                  value: String(data.validation_scene_count),
                },
              ]}
            />
            <Panel
              className="mt-5"
              title="Validation scenes"
              eyebrow="Scene identifiers from the validation baseline"
            >
              <div className="flex flex-wrap gap-2">
                {data.validation_scenes.map((scene) => (
                  <Status key={scene} label={scene} />
                ))}
              </div>
              <UnitNote units={data.coordinate_units} />
            </Panel>
          </>
        )}
      </ResourceState>
      <div className="mt-5">
        <ResourceState resource={baseline} label="per-scene validation metrics">
          {(data) => (
            <Panel
              title="LSTM validation windows by scene"
              eyebrow={`${data.window_count.toLocaleString()} total candidate windows`}
            >
              <div className="overflow-x-auto">
                <table className="w-full min-w-[600px] text-left text-xs">
                  <thead className="border-b border-white/[.08] font-mono text-[10px] uppercase tracking-wider text-slate-600">
                    <tr>
                      {['Scene', 'Windows', 'ADE', 'FDE'].map((label) => (
                        <th key={label} className="pb-3 pr-4 font-normal">
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.per_scene).map(([scene, metrics]) => (
                      <tr
                        key={scene}
                        className="border-b border-white/[.05] text-slate-400"
                      >
                        <td className="py-3 pr-4 text-slate-200">{scene}</td>
                        <td className="py-3 pr-4 font-mono">
                          {metrics.window_count
                            ? Math.round(metrics.window_count.mean).toLocaleString()
                            : '—'}
                        </td>
                        <td className="py-3 pr-4 font-mono">
                          {fixed(metrics.ade.mean)}
                        </td>
                        <td className="py-3 pr-4 font-mono">
                          {fixed(metrics.fde.mean)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <UnitNote units={data.coordinate_units} />
            </Panel>
          )}
        </ResourceState>
      </div>
      <div className="mt-5">
        <ResourceState resource={audit} label="ETH dataset audit">
          {(data) => <EthAuditPanel data={data} />}
        </ResourceState>
      </div>
      <Mot17DatasetPanels />
      <WorkflowFooter stage="data" />
    </AppShell>
  )
}

function EthAuditPanel({ data }: { data: EthDatasetAuditResponse }) {
  return (
    <Panel title="ETH dataset audit" eyebrow="Split structure and integrity">
      <DataTable
        minWidthClass="min-w-[520px]"
        headers={['Split', 'Candidate windows', 'Observations', 'Files present']}
      >
        {data.split_summaries.map((row) => (
          <tr key={row.split} className="border-b border-white/[.05]">
            <Cell strong>{row.split}</Cell>
            <Cell>{row.candidate_window_count.toLocaleString()}</Cell>
            <Cell>{row.observations.toLocaleString()}</Cell>
            <Cell>{`${row.files_present} / ${row.files_expected}`}</Cell>
          </tr>
        ))}
      </DataTable>
      <ScienceNote>
        {data.issue_count} audit issues; {data.identical_observation_rows_across_splits}{' '}
        identical observation rows across {data.scene_overlap_checks} scene
        overlap checks. These are structural counts only, not test metrics.
      </ScienceNote>
      <p className="mt-4 break-words font-mono text-[10px] leading-5 text-slate-600">
        Source: runs/{data.source}
      </p>
    </Panel>
  )
}

export function AnalysisResultsPage() {
  const experiments = useApiResource<ErrorExperimentsResponse>(
    '/api/v1/datasets/eth/error-injections',
  )
  const [selectedExperiment, setSelectedExperiment] =
    useState<ExperimentId>('localization_noise')
  const [selectedSeverity, setSelectedSeverity] =
    useState<SeverityId>('low')
  const detail = useApiResource<ExperimentDetailResponse>(
    `/api/v1/datasets/eth/error-injections/${selectedExperiment}?severity=${selectedSeverity}`,
  )

  return (
    <AppShell stage="analysis">
      <PageHeader
        eyebrow="Stage 07 / Saved experiment analysis"
        title="Research analysis"
        description="ETH prediction analysis and MOT17 tracking analysis, shown as separate studies."
        stage="07"
      />
      <DatasetSection
        dataset="ETH"
        title="Error propagation analysis"
        detail="Clean and corrupted LSTM results across the saved validation experiments."
      />
      <ResourceState resource={experiments} label="error-injection analysis">
        {(data) => (
          <>
            <div className="mb-5 grid gap-2 border border-white/[.08] bg-[#0b1017] p-4 sm:grid-cols-2">
              <label className="text-xs text-slate-500">
                Experiment
                <select
                  value={selectedExperiment}
                  onChange={(event) =>
                    setSelectedExperiment(event.target.value as ExperimentId)
                  }
                  className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
                >
                  {EXPERIMENTS.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-xs text-slate-500">
                Severity
                <select
                  value={selectedSeverity}
                  onChange={(event) =>
                    setSelectedSeverity(event.target.value as SeverityId)
                  }
                  className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
                >
                  {SEVERITIES.map((severity) => (
                    <option key={severity} value={severity}>
                      {severity}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="grid gap-5 xl:grid-cols-2">
              {(['ade', 'fde'] as const).map((metric) => (
                <SeverityChart
                  key={metric}
                  data={data}
                  experiment={selectedExperiment}
                  metric={metric}
                />
              ))}
            </div>
          </>
        )}
      </ResourceState>
      <div className="mt-5">
        <ResourceState resource={detail} label="selected experiment details">
          {(data) => <DetailAnalysis data={data} />}
        </ResourceState>
      </div>
      <Mot17AnalysisSection />
      <WorkflowFooter stage="analysis" />
    </AppShell>
  )
}

function SeverityChart({
  data,
  experiment,
  metric,
}: {
  data: ErrorExperimentsResponse
  experiment: ExperimentId
  metric: 'ade' | 'fde'
}) {
  const entry = data.experiments.find(
    (item) => item.experiment_id === experiment,
  )
  const chartData = (entry?.severities ?? []).map((item) => ({
    severity: item.severity,
    globalClean: item.global_clean_metrics[metric].mean,
    matchedClean: item.matched_clean_metrics?.[metric].mean ?? null,
    corrupted: item.corrupted_metrics[metric].mean,
  }))

  return (
    <Panel
      title={`${metric.toUpperCase()} by severity`}
      eyebrow={`${experimentLabel(experiment)} · saved API means`}
    >
      <div className="h-[280px]">
        {chartData.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid stroke="rgba(255,255,255,.08)" />
              <XAxis dataKey="severity" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip
                contentStyle={{
                  background: '#0b1017',
                  border: '1px solid rgba(255,255,255,.12)',
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="globalClean"
                name="Global clean baseline"
                stroke="#94a3b8"
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="matchedClean"
                name="Matched-clean windows"
                stroke="#22d3ee"
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="corrupted"
                name="Corrupted"
                stroke="#fb7185"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <EmptyState title="No severity data" />
        )}
      </div>
    </Panel>
  )
}

function DetailAnalysis({ data }: { data: ExperimentDetailResponse }) {
  return (
    <>
      <Panel
        title={`${experimentLabel(data.experiment_id)} · ${data.severity}`}
        eyebrow="Backend-provided matched and corrupted comparisons"
      >
        <MetricGrid
          items={[
            {
              label: 'Global clean ADE',
              value: stat(data.global_clean_metrics.ade),
            },
            {
              label: 'Matched clean ADE',
              value: stat(data.matched_clean_metrics?.ade),
            },
            {
              label: 'Corrupted ADE',
              value: stat(data.corrupted_metrics.ade),
            },
            {
              label: 'ADE degradation',
              value: stat(data.degradation.ade),
              detail: `${stat(data.percentage_degradation.ade)}%`,
            },
          ]}
        />
        {data.experiment_id === 'detection_dropout' && (
          <p className="mt-4 text-xs leading-5 text-amber-200/80">
            Dropout results distinguish the global clean baseline from
            matched-clean performance on the evaluated windows.
          </p>
        )}
        {data.experiment_id === 'track_fragmentation' && (
          <p className="mt-4 text-xs leading-5 text-amber-200/80">
            Fragmentation degradation compares corrupted predictions against
            matched-clean metrics on the same retained windows.
          </p>
        )}
        {data.experiment_id === 'identity_switch' && (
          <p className="mt-4 text-xs leading-5 text-amber-200/80">
            Identity-switch clean and corrupted metrics use identical windows.
          </p>
        )}
        <UnitNote units={data.coordinate_units} />
      </Panel>
      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <SceneTable data={data} />
        <TimestepTable data={data} />
      </div>
    </>
  )
}

function SceneTable({ data }: { data: ExperimentDetailResponse }) {
  return (
    <Panel title="Per-scene results" eyebrow="Metrics as returned by the API">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-left text-xs">
          <thead className="border-b border-white/[.08] font-mono text-[10px] uppercase tracking-wider text-slate-600">
            <tr>
              {['Scene', 'Matched ADE', 'Corrupted ADE', 'Matched FDE', 'Corrupted FDE'].map(
                (label) => (
                  <th key={label} className="pb-3 pr-3 font-normal">
                    {label}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.per_scene).map(([scene, values]) => (
              <tr key={scene} className="border-b border-white/[.05]">
                <td className="py-3 pr-3 text-slate-200">{scene}</td>
                <td className="py-3 pr-3 font-mono text-slate-400">
                  {stat(values.matched_clean?.ade)}
                </td>
                <td className="py-3 pr-3 font-mono text-slate-300">
                  {stat(values.corrupted.ade)}
                </td>
                <td className="py-3 pr-3 font-mono text-slate-400">
                  {stat(values.matched_clean?.fde)}
                </td>
                <td className="py-3 pr-3 font-mono text-slate-300">
                  {stat(values.corrupted.fde)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

function TimestepTable({ data }: { data: ExperimentDetailResponse }) {
  return (
    <Panel
      title="Per-timestep ADE"
      eyebrow="Prediction horizon · saved report values"
    >
      {data.per_timestep.length ? (
        <div className="max-h-80 overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 border-b border-white/[.08] bg-[#0b1017] font-mono text-[10px] uppercase tracking-wider text-slate-600">
              <tr>
                {['Timestep', 'Matched clean', 'Corrupted', 'Difference'].map(
                  (label) => (
                    <th key={label} className="pb-3 pr-3 font-normal">
                      {label}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {data.per_timestep.map((item) => (
                <tr
                  key={item.timestep}
                  className="border-b border-white/[.05] text-slate-400"
                >
                  <td className="py-2 pr-3">{item.timestep}</td>
                  <td className="py-2 pr-3 font-mono">
                    {stat(item.matched_clean_ade)}
                  </td>
                  <td className="py-2 pr-3 font-mono">
                    {stat(item.corrupted_ade)}
                  </td>
                  <td className="py-2 pr-3 font-mono">
                    {stat(item.degradation_ade)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState
          title="Per-timestep metrics unavailable"
          message="The saved report does not include per-timestep errors."
        />
      )}
    </Panel>
  )
}

export function ExperimentsResultsPage() {
  const summaries = useApiResource<ErrorExperimentsResponse>(
    '/api/v1/datasets/eth/error-injections',
  )
  const [selectedExperiment, setSelectedExperiment] =
    useState<ExperimentId>('localization_noise')
  const [selectedSeverity, setSelectedSeverity] =
    useState<SeverityId>('low')
  const [selectedSeed, setSelectedSeed] = useState('')
  const detail = useApiResource<ExperimentDetailResponse>(
    `/api/v1/datasets/eth/error-injections/${selectedExperiment}?severity=${selectedSeverity}${selectedSeed ? `&seed=${selectedSeed}` : ''}`,
  )

  return (
    <AppShell>
      <PageHeader
        eyebrow="Saved experiments"
        title="Research experiments"
        description="The four saved ETH validation error-injection experiments, followed by the separate MOT17 tracking experiments."
      />
      <DatasetSection
        dataset="ETH"
        title="Error-injection experiments"
        detail="Configured severity parameters, matched-window metrics and saved visual artifacts."
      />
      <ResourceState resource={summaries} label="experiment summaries">
        {(data) => <ExperimentSummaryTable data={data} />}
      </ResourceState>
      <div className="mt-5 grid gap-5 md:grid-cols-3">
        <label className="text-xs text-slate-500">
          Experiment
          <select
            value={selectedExperiment}
            onChange={(event) =>
              setSelectedExperiment(event.target.value as ExperimentId)
            }
            className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
          >
            {EXPERIMENTS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500">
          Severity
          <select
            value={selectedSeverity}
            onChange={(event) =>
              setSelectedSeverity(event.target.value as SeverityId)
            }
            className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
          >
            {SEVERITIES.map((severity) => (
              <option key={severity} value={severity}>
                {severity}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500">
          Seed
          <select
            value={selectedSeed}
            onChange={(event) => setSelectedSeed(event.target.value)}
            className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
          >
            <option value="">All seeds (mean ± SD)</option>
            {['42', '43', '44'].map((seed) => (
              <option key={seed} value={seed}>
                Seed {seed}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="mt-5">
        <ResourceState resource={detail} label="selected experiment details">
          {(data) => (
            <ExperimentDetails data={data} severity={selectedSeverity} />
          )}
        </ResourceState>
      </div>
      <Mot17ExperimentsSection />
    </AppShell>
  )
}

function ExperimentDetails({
  data,
  severity,
}: {
  data: ExperimentDetailResponse
  severity: SeverityId
}) {
  const [failedVisualization, setFailedVisualization] = useState<string | null>(
    null,
  )
  const windows = data.windows
  const visualizationUrl = getVisualizationUrl(data.experiment_id, severity)

  return (
    <>
      <div className="grid gap-5 xl:grid-cols-[1fr_.8fr]">
        <Panel
          title={`${experimentLabel(data.experiment_id)} · ${severity}`}
          eyebrow={`Seeds ${data.seed_ids.join(', ')}`}
        >
          <div className="grid grid-cols-2 gap-3 text-xs">
            {Object.entries(data.parameters).map(([key, value]) => (
              <div key={key} className="border border-white/[.07] p-3">
                <div className="font-mono text-[9px] uppercase tracking-wider text-slate-600">
                  {key.replaceAll('_', ' ')}
                </div>
                <div className="mt-1 break-words text-slate-300">
                  {String(value)}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5">
            <MetricGrid
              items={[
                {
                  label: 'Global clean ADE',
                  value: stat(data.global_clean_metrics.ade),
                },
                {
                  label: 'Matched clean ADE',
                  value: stat(data.matched_clean_metrics?.ade),
                },
                {
                  label: 'Corrupted ADE',
                  value: stat(data.corrupted_metrics.ade),
                },
                {
                  label: 'ADE degradation',
                  value: stat(data.degradation.ade),
                  detail: `${stat(data.percentage_degradation.ade)}%`,
                },
              ]}
            />
          </div>
          <MetricGrid
            items={[
              {
                label: 'Global clean FDE',
                value: stat(data.global_clean_metrics.fde),
              },
              {
                label: 'Matched clean FDE',
                value: stat(data.matched_clean_metrics?.fde),
              },
              {
                label: 'Corrupted FDE',
                value: stat(data.corrupted_metrics.fde),
              },
              {
                label: 'FDE degradation',
                value: stat(data.degradation.fde),
                detail: `${stat(data.percentage_degradation.fde)}%`,
              },
            ]}
          />
          <UnitNote units={data.coordinate_units} />
        </Panel>
        <Panel title="Window and affected statistics" eyebrow="Backend summary">
          <div className="space-y-3">
            {[
              ['Candidate windows', String(windows.candidate_count)],
              ['Retained / evaluated', stat(windows.retained_count)],
              ['Unusable windows', stat(windows.unusable_count)],
              ['Retention', `${stat(windows.retention_percentage)}%`],
              ...Object.entries(windows.affected_statistics).map(
                ([key, value]) => [key.replaceAll('_', ' '), stat(value)],
              ),
            ].map(([label, value]) => (
              <div
                key={label}
                className="flex justify-between gap-4 border-b border-white/[.06] pb-3 text-xs"
              >
                <span className="text-slate-500">{label}</span>
                <span className="text-right font-mono text-slate-300">
                  {value}
                </span>
              </div>
            ))}
          </div>
          {data.experiment_id === 'detection_dropout' && (
            <p className="mt-4 text-xs leading-5 text-amber-200/80">
              Global clean metrics are shown separately from matched-clean
              performance on the evaluated dropout windows.
            </p>
          )}
          {data.experiment_id === 'track_fragmentation' && (
            <p className="mt-4 text-xs leading-5 text-amber-200/80">
              Prediction degradation is computed against clean metrics on the
              same retained windows; retention is reported separately.
            </p>
          )}
          {data.experiment_id === 'identity_switch' && (
            <p className="mt-4 text-xs leading-5 text-amber-200/80">
              Clean and corrupted metrics use the same evaluated windows.
            </p>
          )}
        </Panel>
      </div>
      <div className="mt-5">
        <Panel title="Saved experiment visualization" eyebrow={severity}>
          {failedVisualization !== visualizationUrl ? (
            // These fixed experiment/severity values map to the API's filename allowlist.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={`${data.experiment_id}-${severity}`}
              src={visualizationUrl}
              alt={`${experimentLabel(data.experiment_id)} ${severity} visualization`}
              onError={() => setFailedVisualization(visualizationUrl)}
              className="mx-auto max-h-[560px] max-w-full object-contain"
            />
          ) : (
            <EmptyState
              title="Visualization unavailable"
              message="The API could not serve this allowlisted image."
            />
          )}
        </Panel>
      </div>
      <div className="mt-5">
        <DetailAnalysis data={data} />
      </div>
    </>
  )
}

export function TrackingResultsPage() {
  return (
    <AppShell stage="tracking">
      <PageHeader
        eyebrow="Stage 03 / MOT17 tracking evaluation"
        title="Tracking evaluation"
        description="Real YOLOv8s and BoT-SORT results on MOT17, with the GMC-enabled run as the primary result and the no-GMC run as an ablation comparison."
        stage="03"
      />
      <Mot17TrackingResults />
      <WorkflowFooter stage="tracking" />
    </AppShell>
  )
}

const EXPERIMENT_DEFINITIONS: Record<ExperimentId, string> = {
  localization_noise: 'Random noise added to observed positions.',
  detection_dropout: 'Observations removed at a configured probability.',
  track_fragmentation: 'Sampled observations removed to break tracks.',
  identity_switch: 'Observations swapped between nearby pedestrians.',
}

export function ErrorInjectionResultsPage() {
  const experiments = useApiResource<ErrorExperimentsResponse>(
    '/api/v1/datasets/eth/error-injections',
  )
  const [selected, setSelected] = useState<ExperimentId>('localization_noise')

  return (
    <AppShell stage="error-injection">
      <PageHeader
        eyebrow="Stage 04 / Controlled corruption"
        title="Error injection results"
        description="The four saved controlled error experiments on ETH validation windows. Results are read-only; nothing is run from this page."
        stage="04"
      />
      <DatasetSection
        dataset="ETH"
        title="Controlled error injection"
        detail="Clean future targets stay unchanged; only observed history is corrupted."
      />
      <ResourceState resource={experiments} label="error-injection results">
        {(data) => {
          const entry = data.experiments.find(
            (item) => item.experiment_id === selected,
          )
          return (
            <>
              <div className="mb-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                {EXPERIMENTS.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setSelected(item.id)}
                    className={
                      selected === item.id
                        ? 'border border-cyan-400/40 bg-cyan-400/[.06] p-4 text-left'
                        : 'border border-white/[.08] bg-[#0b1017] p-4 text-left hover:bg-white/[.03]'
                    }
                  >
                    <div className="text-sm text-slate-200">{item.label}</div>
                    <div className="mt-2 text-xs leading-5 text-slate-500">
                      {EXPERIMENT_DEFINITIONS[item.id]}
                    </div>
                  </button>
                ))}
              </div>
              {entry ? (
                <SeverityResults experiment={selected} summaries={entry.severities} />
              ) : (
                <EmptyState
                  title="Experiment unavailable"
                  message="The API did not return this experiment."
                />
              )}
            </>
          )
        }}
      </ResourceState>
      <WorkflowFooter stage="error-injection" />
    </AppShell>
  )
}

function SeverityResults({
  experiment,
  summaries,
}: {
  experiment: ExperimentId
  summaries: SeveritySummary[]
}) {
  const affected = Object.keys(summaries[0]?.windows.affected_statistics ?? {})
  return (
    <>
      <Panel
        title={`${experimentLabel(experiment)} by severity`}
        eyebrow={`Seeds ${summaries[0]?.seed_ids.join(', ') ?? '—'} · LSTM`}
      >
        <DataTable
          minWidthClass="min-w-[900px]"
          headers={['Severity', 'Parameters', 'Evaluated windows', 'Retention', 'ADE degradation', 'FDE degradation']}
        >
          {summaries.map((row) => (
            <tr key={row.severity} className="border-b border-white/[.05]">
              <td className="py-3 pr-4 capitalize text-cyan-300">{row.severity}</td>
              <td className="py-3 pr-4 text-slate-500">
                {Object.entries(row.parameters)
                  .filter(([key]) => key !== 'interpretation' && key !== 'units')
                  .map(([key, value]) => `${key.replaceAll('_', ' ')}: ${String(value)}`)
                  .join(', ') || 'Not reported'}
              </td>
              <Cell>{stat(row.windows.evaluated_count)}</Cell>
              <Cell>{`${stat(row.windows.retention_percentage)}%`}</Cell>
              <Cell>{`${stat(row.degradation.ade)} (${stat(row.percentage_degradation.ade)}%)`}</Cell>
              <Cell>{`${stat(row.degradation.fde)} (${stat(row.percentage_degradation.fde)}%)`}</Cell>
            </tr>
          ))}
        </DataTable>
        <ScienceNote>
          Severity levels are exploratory. They are not calibrated to MOT17
          tracking errors, because no calibration artifact exists.
        </ScienceNote>
        <UnitNote units={summaries[0]?.coordinate_units ?? 'not reported'} />
      </Panel>
      {affected.length > 0 && (
        <div className="mt-5">
          <Panel title="Measured corruption" eyebrow="Saved per-severity statistics">
            <DataTable
              minWidthClass="min-w-[600px]"
              headers={['Measure', ...summaries.map((row) => row.severity)]}
            >
              {affected.map((name) => (
                <tr key={name} className="border-b border-white/[.05]">
                  <td className="py-3 pr-4 text-slate-300">
                    {name.replaceAll('_', ' ')}
                  </td>
                  {summaries.map((row) => (
                    <Cell key={row.severity}>
                      {stat(row.windows.affected_statistics[name])}
                    </Cell>
                  ))}
                </tr>
              ))}
            </DataTable>
          </Panel>
        </div>
      )}
    </>
  )
}

export function ErrorVisualizationResultsPage() {
  const figures = useApiResource<EthFiguresResponse>(
    '/api/v1/datasets/eth/error-injections/figures',
  )
  const [experiment, setExperiment] = useState<ExperimentId>('localization_noise')
  const [severity, setSeverity] = useState<SeverityId>('low')
  return (
    <AppShell stage="error-visualization">
      <PageHeader
        eyebrow="Stage 05 / Visual comparison"
        title="Error visualization"
        description="Saved figures from the controlled ETH experiments and from real MOT17 tracking. No video viewer is shown because no videos were saved."
        stage="05"
      />
      <DatasetSection
        dataset="ETH"
        title="Controlled error examples"
        detail="Saved figures from the ETH error-injection experiments."
      />
      <div className="mb-5 grid gap-2 border border-white/[.08] bg-[#0b1017] p-4 sm:grid-cols-2">
        <label className="text-xs text-slate-500">
          Experiment
          <select
            value={experiment}
            onChange={(event) => setExperiment(event.target.value as ExperimentId)}
            className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
          >
            {EXPERIMENTS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500">
          Severity
          <select
            value={severity}
            onChange={(event) => setSeverity(event.target.value as SeverityId)}
            className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
          >
            {SEVERITIES.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Panel
        title={`${experimentLabel(experiment)} example`}
        eyebrow={`${severity} · saved ETH figure`}
      >
        <SavedFigure
          src={getVisualizationUrl(experiment, severity)}
          alt={`${experimentLabel(experiment)} ${severity} example`}
        />
      </Panel>
      <div className="mt-5">
        <ResourceState resource={figures} label="ETH summary figures">
          {(data) => (
            <div className="grid gap-5 xl:grid-cols-2">
              {data.figures.map((figure) => (
                <Panel
                  key={figure.figure_id}
                  title={figure.title}
                  eyebrow={`ETH · ${figure.group}`}
                >
                  <SavedFigure
                    src={getEthFigureUrl(figure.figure_id)}
                    alt={figure.title}
                  />
                </Panel>
              ))}
            </div>
          )}
        </ResourceState>
      </div>
      <Mot17VisualGallery />
      <WorkflowFooter stage="error-visualization" />
    </AppShell>
  )
}

function TimestepChart({
  lstm,
  constantVelocity,
}: {
  lstm: BaselineResponse
  constantVelocity: BaselineResponse
}) {
  const rows = lstm.ade_by_timestep.map((item, index) => ({
    timestep: item.timestep,
    lstm: item.ade,
    constantVelocity: constantVelocity.ade_by_timestep[index]?.ade ?? null,
  }))
  return (
    <Panel title="ADE by prediction timestep" eyebrow="ETH validation · clean input">
      {rows.length ? (
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows}>
              <CartesianGrid stroke="rgba(255,255,255,.08)" />
              <XAxis dataKey="timestep" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip
                contentStyle={{
                  background: '#0b1017',
                  border: '1px solid rgba(255,255,255,.12)',
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="lstm" name="LSTM" stroke="#22d3ee" dot={false} />
              <Line
                type="monotone"
                dataKey="constantVelocity"
                name="Constant velocity"
                stroke="#94a3b8"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <EmptyState
          title="Per-timestep metrics unavailable"
          message="The saved report does not include per-timestep errors."
        />
      )}
    </Panel>
  )
}

export function PredictionResultsPage() {
  const summary = useApiResource<DashboardSummaryResponse>(
    '/api/v1/dashboard/summary?dataset=eth',
  )
  const lstm = useApiResource<BaselineResponse>(
    '/api/v1/datasets/eth/prediction/baseline?split=val&model=lstm',
  )
  const constantVelocity = useApiResource<BaselineResponse>(
    '/api/v1/datasets/eth/prediction/baseline?split=val&model=constant_velocity',
  )
  return (
    <AppShell stage="prediction">
      <PageHeader
        eyebrow="Stage 06 / Future trajectory"
        title="Trajectory prediction"
        description="Saved ETH validation baselines for the two implemented predictors. Nothing is trained or run from this page."
        stage="06"
      />
      <DatasetSection
        dataset="ETH"
        title="Clean validation baselines"
        detail="Validation split only. The sealed test evaluation is not displayed."
      />
      <div className="grid gap-5 lg:grid-cols-2">
        <ModelBaselinePanel model="lstm" title="LSTM baseline" />
        <ModelBaselinePanel
          model="constant_velocity"
          title="Constant-velocity baseline"
        />
      </div>
      <div className="mt-5">
        <ResourceState resource={summary} label="LSTM configuration">
          {(data) => (
            <Panel title="LSTM configuration" eyebrow="Saved experiment record">
              <div className="grid grid-cols-2 gap-3 text-xs md:grid-cols-4">
                {Object.entries(data.lstm_model_configuration).map(([key, value]) => (
                  <div key={key} className="border border-white/[.07] p-3">
                    <div className="font-mono text-[9px] uppercase tracking-wider text-slate-600">
                      {key.replaceAll('_', ' ')}
                    </div>
                    <div className="mt-1 break-words text-slate-300">
                      {String(value)}
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </ResourceState>
      </div>
      <div className="mt-5">
        <ResourceState resource={lstm} label="LSTM baseline">
          {(lstmData) => (
            <ResourceState
              resource={constantVelocity}
              label="constant-velocity baseline"
            >
              {(constantData) => (
                <>
                  <TimestepChart lstm={lstmData} constantVelocity={constantData} />
                  <Panel
                    className="mt-5"
                    title="Per-scene comparison"
                    eyebrow="ETH validation · clean input"
                  >
                    <DataTable
                      headers={['Scene', 'Windows', 'LSTM ADE', 'LSTM FDE', 'Constant-velocity ADE', 'Constant-velocity FDE']}
                    >
                      {Object.entries(lstmData.per_scene).map(([scene, metrics]) => (
                        <tr key={scene} className="border-b border-white/[.05]">
                          <Cell strong>{scene}</Cell>
                          <Cell>
                            {metrics.window_count
                              ? Math.round(metrics.window_count.mean).toLocaleString()
                              : '—'}
                          </Cell>
                          <Cell>{fixed(metrics.ade.mean)}</Cell>
                          <Cell>{fixed(metrics.fde.mean)}</Cell>
                          <Cell>{fixed(constantData.per_scene[scene]?.ade.mean)}</Cell>
                          <Cell>{fixed(constantData.per_scene[scene]?.fde.mean)}</Cell>
                        </tr>
                      ))}
                    </DataTable>
                    <UnitNote units={lstmData.coordinate_units} />
                  </Panel>
                </>
              )}
            </ResourceState>
          )}
        </ResourceState>
      </div>
      <DatasetSection
        dataset="MOT17"
        title="MOT17 trajectory prediction"
        detail="Not implemented."
      />
      <EmptyState
        title="No MOT17 prediction results"
        message="No predictor has been trained or evaluated on MOT17. MOT17 is used for tracking evaluation only."
      />
      <WorkflowFooter stage="prediction" />
    </AppShell>
  )
}

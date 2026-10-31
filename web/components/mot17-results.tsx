'use client'

import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  EmptyState,
  MetricGrid,
  Panel,
  Status,
} from '@/components/trackvision-app'
import {
  Cell,
  DataTable,
  DatasetSection,
  ResourceState,
  SavedFigure,
  ScienceNote,
  fixed,
} from '@/components/result-primitives'
import {
  type HistogramBin,
  type Mot17AblationResponse,
  type Mot17ConfigurationResponse,
  type Mot17ErrorProfileResponse,
  type Mot17ExperimentKey,
  type Mot17GmcResponse,
  type Mot17Metrics,
  type Mot17SequencesResponse,
  type Mot17SummaryResponse,
  getMot17FigureUrl,
  useApiResource,
} from '@/lib/trackvision-api'

const API = '/api/v1/datasets/mot17/tracking'
const EXPERIMENT_OPTIONS: Array<{ id: Mot17ExperimentKey; label: string }> = [
  { id: 'gmc_enabled', label: 'BoT-SORT_GMC_enabled (primary)' },
  { id: 'no_gmc', label: 'BoT-SORT_no_GMC (ablation comparison)' },
]
const FIGURES = [
  { id: 'per_sequence_metrics', title: 'Per-sequence tracking metrics' },
  { id: 'localization_errors', title: 'Localization error distributions' },
  { id: 'track_lengths_and_missing_frames', title: 'Track lengths and missing frames' },
]
const PIXEL_NOTE =
  'Coordinate units: image pixels. Nothing here is expressed in metres or world coordinates.'
const CHART_TOOLTIP = {
  background: '#0b1017',
  border: '1px solid rgba(255,255,255,.12)',
}

function percent(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}`
    : '—'
}

function count(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.round(value).toLocaleString()
    : '—'
}

function signed(value: number, digits: number): string {
  const text = value.toFixed(digits)
  return value > 0 ? `+${text}` : text
}

function sequenceShort(name: string): string {
  return name.replace('MOT17-', '')
}

function binLabel(bin: HistogramBin): string {
  return bin.upper === null ? `≥ ${bin.lower}` : `${bin.lower}–${bin.upper}`
}

function ExperimentLabel({ label }: { label: string }) {
  return <Status label={label} kind="success" />
}

function ProvenanceNote({ paths }: { paths: string[] }) {
  return (
    <p className="mt-4 break-words font-mono text-[10px] leading-5 text-slate-600">
      Source: {paths.join(' · ')}
    </p>
  )
}

function HistogramChart({
  bins,
  xLabel,
}: {
  bins: HistogramBin[]
  xLabel: string
}) {
  if (!bins.some((bin) => bin.count > 0)) {
    return <EmptyState title="No saved values" message="This distribution is empty." />
  }
  const data = bins.map((bin) => ({ bin: binLabel(bin), count: bin.count }))
  return (
    <div className="h-[280px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,.08)" />
          <XAxis
            dataKey="bin"
            stroke="#64748b"
            tick={{ fontSize: 9 }}
            label={{ value: xLabel, position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 10 }}
            height={44}
          />
          <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
          <Tooltip contentStyle={CHART_TOOLTIP} />
          <Bar dataKey="count" name="Count" fill="#22d3ee" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function ExperimentFilter({
  value,
  onChange,
}: {
  value: Mot17ExperimentKey
  onChange: (value: Mot17ExperimentKey) => void
}) {
  return (
    <label className="text-xs text-slate-500">
      MOT17 experiment
      <select
        aria-label="MOT17 experiment"
        value={value}
        onChange={(event) => onChange(event.target.value as Mot17ExperimentKey)}
        className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
      >
        {EXPERIMENT_OPTIONS.map((item) => (
          <option key={item.id} value={item.id}>
            {item.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function CombinedMetricCards({
  metrics,
  extended = false,
}: {
  metrics: Mot17Metrics
  extended?: boolean
}) {
  const primary = [
    { label: 'HOTA', value: percent(metrics.HOTA) },
    { label: 'DetA', value: percent(metrics.DetA) },
    { label: 'AssA', value: percent(metrics.AssA) },
    { label: 'LocA', value: percent(metrics.LocA) },
    { label: 'MOTA', value: percent(metrics.MOTA) },
    { label: 'IDF1', value: percent(metrics.IDF1) },
    { label: 'ID switches', value: count(metrics.IDSW) },
    { label: 'Fragmentation', value: count(metrics.Frag) },
  ]
  const detail = [
    { label: 'True positives', value: count(metrics.CLR_TP) },
    { label: 'False positives', value: count(metrics.CLR_FP) },
    { label: 'False negatives', value: count(metrics.CLR_FN) },
    {
      label: 'Mostly tracked / lost',
      value: `${count(metrics.MT)} / ${count(metrics.ML)}`,
    },
  ]
  return <MetricGrid items={extended ? [...primary, ...detail] : primary} />
}

export function Mot17OverviewSummary() {
  const summary = useApiResource<Mot17SummaryResponse>(`${API}/summary`)
  return (
    <>
      <DatasetSection
        dataset="MOT17"
        title="Real tracking evaluation"
        detail="YOLOv8s detections with BoT-SORT on seven MOT17 training sequences. Separate from the ETH prediction results above."
      />
      <ResourceState resource={summary} label="MOT17 tracking summary">
        {(data) => (
          <Panel
            title="MOT17 tracking summary"
            eyebrow={`Combined TrackEval results · ${data.sequence_count} sequences · ${count(data.total_frames)} frames`}
          >
            <div className="mb-5 flex flex-wrap gap-2">
              <ExperimentLabel label={data.primary_experiment} />
              <Status label="Primary result" />
              <Status label={`${count(data.gmc_calls)} GMC calls · ${data.gmc_failed_calls} failed`} />
            </div>
            <CombinedMetricCards metrics={data.metrics} />
            <ScienceNote>{data.scope_note}</ScienceNote>
          </Panel>
        )}
      </ResourceState>
    </>
  )
}

export function Mot17DatasetPanels() {
  const sequences = useApiResource<Mot17SequencesResponse>(`${API}/sequences`)
  return (
    <>
      <DatasetSection
        dataset="MOT17"
        title="MOT17 tracking dataset"
        detail="Used for real tracking evaluation and error profiles. No trajectory prediction is run on MOT17."
      />
      <ResourceState resource={sequences} label="MOT17 sequence inventory">
        {(data) => (
          <Panel
            title="MOT17 sequence inventory"
            eyebrow={`${data.sequences.length} training sequences · ${count(data.total_frames)} unique frames`}
          >
            <DataTable
              minWidthClass="min-w-[900px]"
              headers={['Sequence', 'Canonical folder', 'Frames', 'FPS', 'Resolution', 'Camera', 'Evaluated boxes', 'Pedestrians']}
            >
              {data.sequences.map((row) => (
                <tr key={row.sequence} className="border-b border-white/[.05]">
                  <Cell strong>{row.sequence}</Cell>
                  <Cell>{row.canonical_directory}</Cell>
                  <Cell>{count(row.frames)}</Cell>
                  <Cell>{row.frame_rate}</Cell>
                  <Cell>{`${row.width} × ${row.height}`}</Cell>
                  <Cell>{row.camera_class}</Cell>
                  <Cell>{count(row.evaluated_gt_boxes)}</Cell>
                  <Cell>{count(row.evaluated_gt_identities)}</Cell>
                </tr>
              ))}
            </DataTable>
            <ScienceNote>{data.variant_policy}</ScienceNote>
            <ScienceNote>{data.camera_class_rule}</ScienceNote>
            <ProvenanceNote paths={data.provenance} />
          </Panel>
        )}
      </ResourceState>
    </>
  )
}

function SequenceMetricsTable({ data }: { data: Mot17SequencesResponse }) {
  return (
    <Panel
      title="Per-sequence TrackEval results"
      eyebrow={`${data.primary_experiment} · primary result`}
    >
      <DataTable
        minWidthClass="min-w-[1040px]"
        headers={['Sequence', 'Camera', 'HOTA', 'DetA', 'AssA', 'LocA', 'MOTA', 'IDF1', 'IDSW', 'TP', 'FP', 'FN', 'Frag']}
      >
        {data.sequences.map((row) => (
          <tr key={row.sequence} className="border-b border-white/[.05]">
            <Cell strong>{row.sequence}</Cell>
            <Cell>{row.camera_class}</Cell>
            <Cell strong>{percent(row.metrics.HOTA)}</Cell>
            <Cell>{percent(row.metrics.DetA)}</Cell>
            <Cell>{percent(row.metrics.AssA)}</Cell>
            <Cell>{percent(row.metrics.LocA)}</Cell>
            <Cell>{percent(row.metrics.MOTA)}</Cell>
            <Cell>{percent(row.metrics.IDF1)}</Cell>
            <Cell>{count(row.metrics.IDSW)}</Cell>
            <Cell>{count(row.metrics.CLR_TP)}</Cell>
            <Cell>{count(row.metrics.CLR_FP)}</Cell>
            <Cell>{count(row.metrics.CLR_FN)}</Cell>
            <Cell>{count(row.metrics.Frag)}</Cell>
          </tr>
        ))}
      </DataTable>
      <ProvenanceNote paths={data.provenance} />
    </Panel>
  )
}

function ConfigurationPanel({ data }: { data: Mot17ConfigurationResponse }) {
  const rows: Array<[string, string]> = [
    ['Detector', `${data.detector.model_name} (pretrained COCO, person class ${data.person_class_id})`],
    ['Image size', String(data.detector.image_size)],
    ['Detector confidence', String(data.detector.confidence_threshold)],
    ['Detector NMS IoU', String(data.detector.iou_threshold)],
    ['Training performed', data.training_performed ? 'Yes' : 'No'],
    ['Tracker', data.tracker_implementation],
    ['Track high / low / new', `${data.tracker_settings.track_high_thresh} / ${data.tracker_settings.track_low_thresh} / ${data.tracker_settings.new_track_thresh}`],
    ['Track buffer (frames)', String(data.tracker_settings.track_buffer)],
    ['Match threshold', String(data.tracker_settings.match_thresh)],
    ['GMC method', String(data.tracker_settings.gmc_method)],
    ['ReID', data.tracker_settings.with_reid ? 'Enabled' : 'Disabled'],
    ['Settings changed from comparison', data.settings_changed_from_comparison ? 'Yes' : 'No'],
    ['Evaluation', `${data.evaluation.toolkit}, IoU ${data.evaluation.iou_threshold}`],
  ]
  return (
    <Panel
      title="Detector and tracker configuration"
      eyebrow={`MOT17 · ${data.experiment}`}
    >
      <div className="space-y-3">
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="flex justify-between gap-4 border-b border-white/[.06] pb-3 text-xs"
          >
            <span className="text-slate-500">{label}</span>
            <span className="text-right font-mono text-slate-300">{value}</span>
          </div>
        ))}
      </div>
      <ScienceNote>{data.gmc.boxes_note}</ScienceNote>
      <p className="mt-4 break-words font-mono text-[10px] leading-5 text-slate-600">
        Weights SHA-256: {data.weights_sha256}
      </p>
      <ProvenanceNote paths={data.provenance} />
    </Panel>
  )
}

export function Mot17ConfigurationPanel() {
  const configuration = useApiResource<Mot17ConfigurationResponse>(
    `${API}/configuration`,
  )
  return (
    <ResourceState resource={configuration} label="MOT17 configuration">
      {(data) => <ConfigurationPanel data={data} />}
    </ResourceState>
  )
}

function GmcPanel({ data }: { data: Mot17GmcResponse }) {
  const warnings = Object.values(data.totals.ultralytics_gmc_warnings).reduce(
    (total, value) => total + value,
    0,
  )
  return (
    <Panel
      title="GMC diagnostics"
      eyebrow={`${data.experiment} · ${data.method}`}
    >
      <MetricGrid
        items={[
          { label: 'Calls / expected', value: `${count(data.totals.calls)} / ${count(data.totals.calls_expected)}` },
          { label: 'Failed calls', value: count(data.totals.failed_calls) },
          { label: 'Estimated warps', value: count(data.totals.estimated_warps) },
          {
            label: 'First-frame identity',
            value: count(data.totals.first_frame_identity_warps),
            detail: 'A first frame has no previous frame',
          },
          { label: 'Other identity warps', value: count(data.totals.non_first_identity_warps) },
          { label: 'GMC warnings logged', value: count(warnings) },
          {
            label: 'GMC-off control',
            value: data.control_identical_to_no_gmc ? 'Identical' : 'Differs',
            detail: 'Reproduces the no-GMC tracker files',
          },
          { label: 'Sequences', value: count(data.sequences.length) },
        ]}
      />
      <div className="mt-5">
        <DataTable
          minWidthClass="min-w-[900px]"
          headers={['Sequence', 'Camera', 'Calls', 'Failed', 'Warp motion (px)', 'Phase motion (px)', 'Correlation x / y', 'Image error improved']}
        >
          {data.sequences.map((row) => (
            <tr key={row.sequence} className="border-b border-white/[.05]">
              <Cell strong>{row.sequence}</Cell>
              <Cell>{row.camera_class}</Cell>
              <Cell>{count(row.calls)}</Cell>
              <Cell>{count(row.failed_calls)}</Cell>
              <Cell>{fixed(row.median_warp_center_motion_pixels, 2)}</Cell>
              <Cell>{fixed(row.median_phase_motion_pixels, 2)}</Cell>
              <Cell>{`${fixed(row.correlation_x, 2)} / ${fixed(row.correlation_y, 2)}`}</Cell>
              <Cell>{`${percent(row.share_of_frame_pairs_improved, 1)}%`}</Cell>
            </tr>
          ))}
        </DataTable>
      </div>
      <ScienceNote>
        Warps match independent image evidence on moving cameras. On
        static-looking cameras they are small noise, not useful compensation,
        and they can add false motion. Warps do not model parallax.
      </ScienceNote>
      <ProvenanceNote paths={data.provenance} />
    </Panel>
  )
}

function AblationChart({
  data,
  metric,
}: {
  data: Mot17AblationResponse
  metric: 'HOTA' | 'IDF1' | 'MOTA'
}) {
  const rows = data.rows
    .filter((row) => row.camera_class !== null)
    .map((row) => ({
      sequence: sequenceShort(row.sequence),
      [data.comparison_experiment]: Number(percent(row.metrics[metric].no_gmc)),
      [data.primary_experiment]: Number(percent(row.metrics[metric].gmc_enabled)),
    }))
  return (
    <Panel title={`${metric} by sequence`} eyebrow="Ablation comparison · percent">
      <div className="h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows}>
            <CartesianGrid stroke="rgba(255,255,255,.08)" />
            <XAxis dataKey="sequence" stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, 'auto']} />
            <Tooltip contentStyle={CHART_TOOLTIP} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar
              dataKey={data.comparison_experiment}
              fill="#64748b"
              isAnimationActive={false}
            />
            <Bar
              dataKey={data.primary_experiment}
              fill="#22d3ee"
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function ComparedCell({
  value,
  digits = 2,
  scale = 100,
}: {
  value: { no_gmc: number; gmc_enabled: number; delta: number }
  digits?: number
  scale?: number
}) {
  const worse = scale === 100 ? value.delta < 0 : false
  return (
    <td className={`py-3 pr-4 font-mono ${worse ? 'text-amber-300' : 'text-slate-400'}`}>
      {(value.no_gmc * scale).toFixed(digits)} → {(value.gmc_enabled * scale).toFixed(digits)}{' '}
      ({signed(value.delta * scale, digits)})
    </td>
  )
}

function CountChange({ value }: { value: { no_gmc: number; gmc_enabled: number } }) {
  return (
    <Cell>
      {count(value.no_gmc)} → {count(value.gmc_enabled)}
    </Cell>
  )
}

function AblationTable({ data }: { data: Mot17AblationResponse }) {
  return (
    <Panel
      title="No-GMC versus GMC-enabled"
      eyebrow={`${data.comparison_experiment} → ${data.primary_experiment} · ablation comparison`}
    >
      <DataTable
        minWidthClass="min-w-[1040px]"
        headers={['Sequence', 'Camera', 'HOTA', 'AssA', 'DetA', 'MOTA', 'IDF1', 'IDSW', 'Frag', 'FP', 'FN']}
      >
        {data.rows.map((row) => (
          <tr key={row.sequence} className="border-b border-white/[.05]">
            <Cell strong>{row.sequence === 'COMBINED' ? 'Combined' : row.sequence}</Cell>
            <Cell>{row.camera_class ?? 'All'}</Cell>
            <ComparedCell value={row.metrics.HOTA} />
            <ComparedCell value={row.metrics.AssA} />
            <ComparedCell value={row.metrics.DetA} />
            <ComparedCell value={row.metrics.MOTA} />
            <ComparedCell value={row.metrics.IDF1} />
            <CountChange value={row.metrics.IDSW} />
            <CountChange value={row.metrics.Frag} />
            <CountChange value={row.metrics.CLR_FP} />
            <CountChange value={row.metrics.CLR_FN} />
          </tr>
        ))}
      </DataTable>
      <ScienceNote>
        GMC did not benefit every sequence. Gains are concentrated on moving
        cameras, MOT17-11 got worse, and changes on static-looking cameras
        should not be read as a benefit. The two experiments are shown side by
        side and their metrics are never pooled.
      </ScienceNote>
      <ScienceNote>
        {data.control_identical_to_no_gmc
          ? 'A GMC-off control reproduced all seven no-GMC tracker files byte for byte, so the differences come from enabling GMC on this input. With seven sequences, they may not generalise.'
          : 'The GMC-off control did not match the no-GMC tracker files, so differences cannot be attributed to GMC.'}
      </ScienceNote>
      <ProvenanceNote paths={data.provenance} />
    </Panel>
  )
}

function RegressionPanel({ data }: { data: Mot17AblationResponse }) {
  const evidence = data.regression
  return (
    <Panel
      title={`${evidence.sequence} regression`}
      eyebrow="Where enabling GMC made results worse"
    >
      <MetricGrid
        items={[
          { label: 'HOTA change', value: signed(evidence.delta.HOTA * 100, 2) },
          { label: 'MOTA change', value: signed(evidence.delta.MOTA * 100, 2) },
          { label: 'IDF1 change', value: signed(evidence.delta.IDF1 * 100, 2) },
          { label: 'ID switch change', value: signed(evidence.delta.IDSW, 0) },
          {
            label: 'Output boxes',
            value: `${count(evidence.output_boxes.no_gmc)} → ${count(evidence.output_boxes.gmc_enabled)}`,
          },
          { label: 'Extra true positives', value: signed(evidence.delta.CLR_TP, 0) },
          { label: 'Extra false positives', value: signed(evidence.delta.CLR_FP, 0) },
          {
            label: 'Median track length',
            value: `${evidence.median_track_length_frames.no_gmc} → ${evidence.median_track_length_frames.gmc_enabled}`,
            detail: 'Frames',
          },
        ]}
      />
      <ScienceNote>
        Nearly all the extra output boxes are false positives. The warps
        themselves look sound here: phase-correlation agreement is{' '}
        {fixed(evidence.warp_correlation_x, 3)} and{' '}
        {fixed(evidence.warp_correlation_y, 3)}, and image error improved on{' '}
        {percent(evidence.share_of_frame_pairs_improved, 1)}% of frame pairs.
      </ScienceNote>
      <ScienceNote>
        Likely explanation, not proven: with GMC, tracks survive camera motion
        better, including tracks following boxes that do not match evaluated
        pedestrians. Some false tracks barely move once stabilized, which suits
        a fixed object that YOLO repeatedly detects. Whether these boxes are
        non-pedestrian objects or unannotated people was not checked by hand.
      </ScienceNote>
      <div className="mt-5">
        <DataTable
          headers={['Track', 'False-positive frames', 'Total frames', 'Median confidence', 'Raw travel (px)', 'Stabilized travel (px)']}
        >
          {evidence.false_positive_tracks.map((track) => (
            <tr key={track.track_id} className="border-b border-white/[.05]">
              <Cell strong>{track.track_id}</Cell>
              <Cell>{count(track.false_positive_frames)}</Cell>
              <Cell>{count(track.total_frames)}</Cell>
              <Cell>{fixed(track.median_confidence, 2)}</Cell>
              <Cell>{fixed(track.raw_travel_pixels, 0)}</Cell>
              <Cell>{fixed(track.stabilized_travel_pixels, 0)}</Cell>
            </tr>
          ))}
        </DataTable>
      </div>
      <ScienceNote>
        Travel is the distance between a track&apos;s first and last foot points, in
        image pixels. Stabilized positions drift over long spans and are not
        world coordinates.
      </ScienceNote>
    </Panel>
  )
}

export function Mot17TrackingResults() {
  const summary = useApiResource<Mot17SummaryResponse>(`${API}/summary`)
  const sequences = useApiResource<Mot17SequencesResponse>(`${API}/sequences`)
  const gmc = useApiResource<Mot17GmcResponse>(`${API}/gmc`)
  const ablation = useApiResource<Mot17AblationResponse>(`${API}/ablation`)
  const [chartMetric, setChartMetric] = useState<'HOTA' | 'IDF1' | 'MOTA'>('HOTA')

  return (
    <>
      <ResourceState resource={summary} label="MOT17 tracking summary">
        {(data) => (
          <Panel
            title="Combined MOT17 tracking results"
            eyebrow="Official TrackEval · primary result"
          >
            <div className="mb-5 flex flex-wrap gap-2">
              <ExperimentLabel label={data.primary_experiment} />
              <Status label={`${data.sequence_count} sequences`} />
              <Status label={`${count(data.total_frames)} frames`} />
              <Status label={`${data.split} split`} />
            </div>
            <CombinedMetricCards metrics={data.metrics} extended />
            <ScienceNote>{data.scope_note}</ScienceNote>
            <ScienceNote>
              Combined values pool counts across sequences; they are not
              averages of the per-sequence percentages. Metrics are percentages
              except counts.
            </ScienceNote>
            <ProvenanceNote paths={data.provenance} />
          </Panel>
        )}
      </ResourceState>
      <div className="mt-5">
        <ResourceState resource={sequences} label="per-sequence metrics">
          {(data) => <SequenceMetricsTable data={data} />}
        </ResourceState>
      </div>
      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Mot17ConfigurationPanel />
        <ResourceState resource={gmc} label="GMC diagnostics">
          {(data) => <GmcPanel data={data} />}
        </ResourceState>
      </div>
      <DatasetSection
        dataset="MOT17"
        title="GMC ablation comparison"
        detail="BoT-SORT_no_GMC is the archived comparison. It is not the main tracking result."
      />
      <ResourceState resource={ablation} label="GMC ablation comparison">
        {(data) => (
          <>
            <AblationTable data={data} />
            <div className="mb-5 mt-5">
              <label className="text-xs text-slate-500">
                Chart metric
                <select
                  aria-label="Chart metric"
                  value={chartMetric}
                  onChange={(event) =>
                    setChartMetric(event.target.value as 'HOTA' | 'IDF1' | 'MOTA')
                  }
                  className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
                >
                  {['HOTA', 'IDF1', 'MOTA'].map((metric) => (
                    <option key={metric} value={metric}>
                      {metric}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <AblationChart data={data} metric={chartMetric} />
            <div className="mt-5">
              <RegressionPanel data={data} />
            </div>
          </>
        )}
      </ResourceState>
    </>
  )
}

function DistributionRows({ data }: { data: Mot17ErrorProfileResponse }) {
  const rows: Array<[string, string, string]> = [
    ['Matched box IoU', fixed(data.overall.iou.median, 3), fixed(data.overall.iou.p95, 3)],
    ['Centre error (px)', fixed(data.overall.center_error_px.median, 2), fixed(data.overall.center_error_px.p95, 2)],
    ['Centre error / box height', fixed(data.overall.center_error_over_height.median, 4), fixed(data.overall.center_error_over_height.p95, 4)],
    ['Interior gap (s)', fixed(data.overall.interior_gap_seconds.median, 3), fixed(data.overall.interior_gap_seconds.p95, 3)],
    ['Track duration (s)', fixed(data.track_duration_seconds.median, 2), fixed(data.track_duration_seconds.p95, 2)],
  ]
  return (
    <DataTable headers={['Measure', 'Median', 'P95']}>
      {rows.map(([label, median, p95]) => (
        <tr key={label} className="border-b border-white/[.05]">
          <Cell strong>{label}</Cell>
          <Cell>{median}</Cell>
          <Cell>{p95}</Cell>
        </tr>
      ))}
    </DataTable>
  )
}

function ErrorProfilePanels({ data }: { data: Mot17ErrorProfileResponse }) {
  const windows = [...new Set(data.availability_by_sequence.map((row) => row.window_seconds))]
  const sequences = [...new Set(data.availability_by_sequence.map((row) => row.sequence))]
  const lookup = new Map(
    data.availability_by_sequence.map((row) => [
      `${row.sequence}|${row.window_seconds}`,
      row.availability,
    ]),
  )
  return (
    <>
      <Panel title="Error profile summary" eyebrow={`${data.experiment} · supplementary analysis`}>
        <MetricGrid
          items={[
            { label: 'Tracker recall', value: `${percent(data.overall.recall)}%` },
            { label: 'Tracker precision', value: `${percent(data.overall.precision)}%` },
            {
              label: 'YOLO recall, conf ≥ 0.25',
              value: `${percent(data.overall['detector_conf_0.25'].recall)}%`,
              detail: 'Saved detections only',
            },
            {
              label: 'YOLO recall, conf ≥ 0.10',
              value: `${percent(data.overall['detector_conf_0.10'].recall)}%`,
              detail: 'Saved detections only',
            },
            {
              label: 'Median coverage',
              value: `${percent(data.overall.per_pedestrian_coverage.median, 1)}%`,
              detail: 'Share of a pedestrian’s frames tracked',
            },
            {
              label: 'Pedestrians',
              value: count(data.coverage_counts.pedestrians),
              detail: 'Evaluated ground-truth identities',
            },
            {
              label: 'Coverage below 5%',
              value: count(data.coverage_counts.below_5_percent),
              detail: `${count(data.coverage_counts.zero)} with zero coverage`,
            },
            {
              label: 'Interior gaps',
              value: count(data.overall.interior_gap_seconds.count),
              detail: 'Fragmentation events',
            },
          ]}
        />
        <div className="mt-5">
          <DistributionRows data={data} />
        </div>
        <ScienceNote>{data.kind}</ScienceNote>
        <ScienceNote>{PIXEL_NOTE}</ScienceNote>
      </Panel>
      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <Panel title="Matched-box IoU" eyebrow={`Localization · ${data.experiment}`}>
          <HistogramChart bins={data.iou_histogram} xLabel="IoU (only values ≥ 0.5 can match)" />
        </Panel>
        <Panel title="Centre error / box height" eyebrow={`Localization · ${data.experiment}`}>
          <HistogramChart bins={data.center_error_histogram} xLabel="Share of ground-truth box height" />
        </Panel>
        <Panel title="Track duration" eyebrow={`Continuity · ${data.experiment}`}>
          <HistogramChart bins={data.track_duration_histogram} xLabel="Seconds" />
        </Panel>
        <Panel title="Missing-frame gaps" eyebrow={`Continuity · ${data.experiment}`}>
          <HistogramChart bins={data.gap_histogram} xLabel="Gap inside a ground-truth track (seconds)" />
        </Panel>
        <Panel title="Per-pedestrian coverage" eyebrow={`Continuity · ${data.experiment}`}>
          <HistogramChart bins={data.coverage_histogram} xLabel="Share of a pedestrian's frames tracked" />
        </Panel>
        <Panel title="Unbroken-identity availability" eyebrow={`Pooled · ${data.experiment}`}>
          <DataTable minWidthClass="min-w-[520px]" headers={['Window (s)', 'Available', 'Possible', 'Availability']}>
            {data.availability.map((row) => (
              <tr key={row.window_seconds} className="border-b border-white/[.05]">
                <Cell strong>{row.window_seconds}</Cell>
                <Cell>{count(row.available_windows)}</Cell>
                <Cell>{count(row.possible_windows)}</Cell>
                <Cell>{row.availability === null ? '—' : `${percent(row.availability)}%`}</Cell>
              </tr>
            ))}
          </DataTable>
          <ScienceNote>{data.definitions.window_availability}</ScienceNote>
          <ScienceNote>
            Pedestrians shorter than a window cannot contribute to it, windows
            overlap, and MOT17-04 dominates the long-window pool.
          </ScienceNote>
        </Panel>
      </div>
      <div className="mt-5">
        <Panel title="Availability by sequence" eyebrow={`Percent · ${data.experiment}`}>
          <DataTable
            minWidthClass="min-w-[900px]"
            headers={['Sequence', ...windows.map((seconds) => `${seconds} s`)]}
          >
            {sequences.map((sequence) => (
              <tr key={sequence} className="border-b border-white/[.05]">
                <Cell strong>{sequence}</Cell>
                {windows.map((seconds) => {
                  const value = lookup.get(`${sequence}|${seconds}`)
                  return <Cell key={seconds}>{value == null ? '—' : percent(value)}</Cell>
                })}
              </tr>
            ))}
          </DataTable>
        </Panel>
      </div>
      <div className="mt-5">
        <Panel title="Per-sequence localization" eyebrow={`${data.experiment} · matched boxes only`}>
          <DataTable
            minWidthClass="min-w-[900px]"
            headers={['Sequence', 'Recall', 'Precision', 'Median IoU', 'Median centre error (px)', 'P95 centre error (px)', 'Centre error / height', 'Median track span (s)']}
          >
            {data.sequences.map((row) => (
              <tr key={row.sequence} className="border-b border-white/[.05]">
                <Cell strong>{row.sequence}</Cell>
                <Cell>{`${percent(row.recall)}%`}</Cell>
                <Cell>{`${percent(row.precision)}%`}</Cell>
                <Cell>{fixed(row.median_iou, 3)}</Cell>
                <Cell>{fixed(row.median_center_error_px, 2)}</Cell>
                <Cell>{fixed(row.p95_center_error_px, 2)}</Cell>
                <Cell>{fixed(row.median_center_error_over_height, 4)}</Cell>
                <Cell>{fixed(row.median_track_span_seconds, 2)}</Cell>
              </tr>
            ))}
          </DataTable>
          <ul className="mt-4 space-y-1 text-xs leading-5 text-slate-500">
            {data.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <ProvenanceNote paths={data.provenance} />
        </Panel>
      </div>
    </>
  )
}

export function Mot17AnalysisSection() {
  const [experiment, setExperiment] = useState<Mot17ExperimentKey>('gmc_enabled')
  const profile = useApiResource<Mot17ErrorProfileResponse>(
    `${API}/error-profile?experiment=${experiment}`,
  )
  return (
    <>
      <DatasetSection
        dataset="MOT17"
        title="Tracking analysis"
        detail="Localization, continuity and availability from real tracker outputs. These are separate from the ETH prediction analysis above."
      />
      <div className="mb-5 border border-white/[.08] bg-[#0b1017] p-4">
        <ExperimentFilter value={experiment} onChange={setExperiment} />
      </div>
      <ResourceState resource={profile} label="MOT17 error profile">
        {(data) => <ErrorProfilePanels data={data} />}
      </ResourceState>
    </>
  )
}

export function Mot17ExperimentsSection() {
  const ablation = useApiResource<Mot17AblationResponse>(`${API}/ablation`)
  return (
    <>
      <DatasetSection
        dataset="MOT17"
        title="Tracking experiments"
        detail="Real tracker runs, not ETH error injections. Open the Tracking stage for full results."
      />
      <ResourceState resource={ablation} label="MOT17 experiments">
        {(data) => {
          const combined = data.rows.find((row) => row.sequence === 'COMBINED')
          if (!combined) {
            return <EmptyState title="No combined result" message="The API returned no combined row." />
          }
          const rows: Array<{ label: string; role: string; key: 'gmc_enabled' | 'no_gmc' }> = [
            { label: data.primary_experiment, role: 'Primary result', key: 'gmc_enabled' },
            { label: data.comparison_experiment, role: 'Ablation comparison', key: 'no_gmc' },
          ]
          return (
            <Panel title="MOT17 tracking experiments" eyebrow="Combined TrackEval results">
              <DataTable
                minWidthClass="min-w-[900px]"
                headers={['Experiment', 'Role', 'HOTA', 'MOTA', 'IDF1', 'IDSW', 'FP', 'FN', 'Frag']}
              >
                {rows.map((row) => (
                  <tr key={row.key} className="border-b border-white/[.05]">
                    <Cell strong>{row.label}</Cell>
                    <Cell>{row.role}</Cell>
                    <Cell>{percent(combined.metrics.HOTA[row.key])}</Cell>
                    <Cell>{percent(combined.metrics.MOTA[row.key])}</Cell>
                    <Cell>{percent(combined.metrics.IDF1[row.key])}</Cell>
                    <Cell>{count(combined.metrics.IDSW[row.key])}</Cell>
                    <Cell>{count(combined.metrics.CLR_FP[row.key])}</Cell>
                    <Cell>{count(combined.metrics.CLR_FN[row.key])}</Cell>
                    <Cell>{count(combined.metrics.Frag[row.key])}</Cell>
                  </tr>
                ))}
              </DataTable>
              <ScienceNote>
                BoT-SORT_no_GMC is an archived run in which motion compensation
                did not operate. It is kept only as a baseline.
              </ScienceNote>
            </Panel>
          )
        }}
      </ResourceState>
    </>
  )
}

export function Mot17VisualGallery() {
  const sequences = useApiResource<Mot17SequencesResponse>(`${API}/sequences`)
  const [experiment, setExperiment] = useState<Mot17ExperimentKey>('gmc_enabled')
  const [sequence, setSequence] = useState('MOT17-10')
  return (
    <>
      <DatasetSection
        dataset="MOT17"
        title="Real tracking examples"
        detail="Saved frames and error figures from real YOLOv8s and BoT-SORT outputs."
      />
      <ResourceState resource={sequences} label="MOT17 sequences">
        {(data) => (
          <>
            <div className="mb-5 grid gap-2 border border-white/[.08] bg-[#0b1017] p-4 sm:grid-cols-2">
              <ExperimentFilter value={experiment} onChange={setExperiment} />
              <label className="text-xs text-slate-500">
                MOT17 sequence
                <select
                  aria-label="MOT17 sequence"
                  value={sequence}
                  onChange={(event) => setSequence(event.target.value)}
                  className="mt-2 w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300"
                >
                  {data.sequences.map((row) => (
                    <option key={row.sequence} value={row.sequence}>
                      {row.sequence} ({row.camera_class})
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <Panel title={`${sequence} frame`} eyebrow={`Original · YOLO · BoT-SORT with IDs · ${experiment === 'gmc_enabled' ? 'BoT-SORT_GMC_enabled' : 'BoT-SORT_no_GMC'}`}>
              <SavedFigure
                src={getMot17FigureUrl(sequence, experiment)}
                alt={`${sequence} original frame, YOLO detections and BoT-SORT tracks`}
              />
            </Panel>
            <div className="mt-5 grid gap-5 xl:grid-cols-2">
              {FIGURES.map((figure) => (
                <Panel
                  key={figure.id}
                  title={figure.title}
                  eyebrow={experiment === 'gmc_enabled' ? 'BoT-SORT_GMC_enabled' : 'BoT-SORT_no_GMC'}
                >
                  <SavedFigure
                    src={getMot17FigureUrl(figure.id, experiment)}
                    alt={figure.title}
                  />
                </Panel>
              ))}
            </div>
            <ScienceNote>{PIXEL_NOTE}</ScienceNote>
          </>
        )}
      </ResourceState>
    </>
  )
}

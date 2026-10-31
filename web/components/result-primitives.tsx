'use client'

import { useState } from 'react'
import { EmptyState, Panel, Status } from '@/components/trackvision-app'

export function fixed(value: number | null | undefined, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(digits)
    : '—'
}

export function stat(value: { mean: number; population_std?: number | null } | null | undefined): string {
  if (!value) return '—'
  const mean = fixed(value.mean)
  return value.population_std == null
    ? mean
    : `${mean} ± ${fixed(value.population_std)}`
}

function ApiLoading({ label }: { label: string }) {
  return (
    <Panel title="Loading saved results" eyebrow="Read-only results API">
      <p className="text-sm text-slate-500" role="status">
        Loading {label} from the TrackVision API…
      </p>
    </Panel>
  )
}

function ApiError({ message }: { message: string }) {
  return (
    <Panel title="Results unavailable" eyebrow="TrackVision API error">
      <div
        className="border border-amber-400/20 bg-amber-400/[.04] p-4 text-sm leading-6 text-amber-200"
        role="alert"
      >
        {message}
      </div>
      <p className="mt-3 text-xs text-slate-500">
        No mock results are shown. Confirm the API is running at the configured
        URL and reload this page.
      </p>
    </Panel>
  )
}

export function ResourceState<T>({
  resource,
  label,
  children,
}: {
  resource: { data: T | null; isLoading: boolean; error: string | null }
  label: string
  children: (data: T) => React.ReactNode
}) {
  if (resource.isLoading) return <ApiLoading label={label} />
  if (resource.error) return <ApiError message={resource.error} />
  if (!resource.data) {
    return (
      <EmptyState
        title="No saved result"
        message={`The API returned no ${label} result.`}
      />
    )
  }
  return <>{children(resource.data)}</>
}

export function UnitNote({ units }: { units: string }) {
  return (
    <p className="mt-4 border-l border-amber-400/40 pl-3 text-xs leading-5 text-amber-200/80">
      Coordinate units: {units}. These values are not labelled as metres.
    </p>
  )
}

export function ScienceNote({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-4 border-l border-amber-400/40 pl-3 text-xs leading-5 text-amber-200/80">
      {children}
    </p>
  )
}

/** Marks where one dataset's results begin, so the two are never confused. */
export function DatasetSection({
  dataset,
  title,
  detail,
}: {
  dataset: 'ETH' | 'MOT17'
  title: string
  detail: string
}) {
  return (
    <div className="mb-5 mt-8 flex flex-col gap-3 border-b border-white/[.07] pb-3 sm:flex-row sm:items-center">
      <Status label={dataset} kind={dataset === 'MOT17' ? 'warning' : 'success'} />
      <div>
        <div className="text-sm font-medium text-slate-200">{title}</div>
        <div className="mt-1 text-xs text-slate-500">{detail}</div>
      </div>
    </div>
  )
}

export function SavedFigure({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState<string | null>(null)
  if (failed === src) {
    return (
      <EmptyState
        title="Visualization unavailable"
        message="The API could not serve this saved image."
      />
    )
  }
  return (
    // Sources are fixed, allowlisted API figures.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      key={src}
      src={src}
      alt={alt}
      onError={() => setFailed(src)}
      className="mx-auto max-h-[560px] max-w-full object-contain"
    />
  )
}

export function DataTable({
  headers,
  minWidthClass = 'min-w-[600px]',
  children,
}: {
  headers: string[]
  minWidthClass?: string
  children: React.ReactNode
}) {
  return (
    <div className="overflow-x-auto">
      <table className={`w-full ${minWidthClass} text-left text-xs`}>
        <thead className="border-b border-white/[.08] font-mono text-[10px] uppercase tracking-wider text-slate-600">
          <tr>
            {headers.map((label) => (
              <th key={label} className="pb-3 pr-4 font-normal">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

export function Cell({
  children,
  strong = false,
}: {
  children: React.ReactNode
  strong?: boolean
}) {
  return (
    <td className={`py-3 pr-4 font-mono ${strong ? 'text-slate-200' : 'text-slate-400'}`}>
      {children}
    </td>
  )
}

'use client'

import { useApiHealth } from '@/lib/trackvision-api'

export function ApiConnectionBadge({
  variant = 'header',
}: {
  variant?: 'header' | 'sidebar'
}) {
  const health = useApiHealth()
  const connected = health.data?.status === 'ok'
  const label = health.isLoading
    ? 'API checking'
    : connected
      ? 'Results API connected'
      : 'Results API unavailable'
  const color = connected
    ? 'text-emerald-300'
    : health.isLoading
      ? 'text-slate-500'
      : 'text-amber-300'

  if (variant === 'sidebar') {
    return (
      <div
        className={`mb-3 flex items-center gap-2 text-[10px] uppercase tracking-wider ${color}`}
        role="status"
      >
        <span className="size-1.5 bg-current" />
        {label}
      </div>
    )
  }

  return (
    <div
      className={`hidden items-center gap-2 border-r border-white/[.08] pr-4 text-xs md:flex ${color}`}
      role="status"
    >
      <span className="size-1.5 bg-current" />
      {label}
    </div>
  )
}

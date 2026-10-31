'use client'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useState } from 'react'
import { Activity, AlertTriangle, ArrowLeft, ArrowRight, BarChart3, ChevronRight, Database, GitCompareArrows, Menu, Settings2, ShieldCheck, Sparkles, Target } from 'lucide-react'
import { navItems } from '@/config/app'
import { ApiConnectionBadge } from '@/components/api-connection-badge'
import { getApiBaseUrl, useApiHealth } from '@/lib/trackvision-api'
import { type WorkflowStage } from '@/types'
import { useExperiment } from '@/stores/experiment-store'

const icons = {overview:Target,data:Database,tracking:Activity,'error-injection':AlertTriangle,'error-visualization':GitCompareArrows,prediction:Sparkles,analysis:BarChart3,'application-impact':ShieldCheck}
function classNames(...values:(string|false|undefined)[]) { return values.filter(Boolean).join(' ') }

export function AppShell({
  children,
  stage,
}: {
  children: React.ReactNode
  stage?: WorkflowStage
}) {
  const pathname = usePathname()
  const router = useRouter()
  const { setStage } = useExperiment()
  const [open, setOpen] = useState(false)
  const active =
    stage ?? (pathname.split('/')[1] as WorkflowStage) ?? 'overview'
  const stages = Object.entries(icons) as [WorkflowStage, typeof Target][]
  const go = (href: string, id: WorkflowStage) => {
    setStage(id)
    setOpen(false)
    router.push(href)
  }

  return (
    <div className="min-h-screen bg-[#070a0f] text-slate-100">
      <aside
        className={classNames(
          'fixed inset-y-0 left-0 z-40 flex w-[276px] flex-col border-r border-white/[.07] bg-[#0a0e14] transition-transform lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex h-[78px] items-center gap-3 border-b border-white/[.07] px-6">
          <div className="grid size-9 place-items-center border border-cyan-400/40 bg-cyan-400/10 text-cyan-300">
            <span className="font-mono text-sm font-bold">TV</span>
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight">TrackVision</div>
            <div className="font-mono text-[9px] uppercase tracking-[.22em] text-slate-500">
              Research workspace
            </div>
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto px-3 py-5">
          <div className="mb-5 space-y-1">
            {navItems.map((item, index) => (
              <Link
                key={item.label}
                href={item.href}
                onClick={() => setOpen(false)}
                className={classNames(
                  'flex items-center gap-3 px-3 py-2.5 text-sm transition-colors',
                  (index === 0 && active === 'overview') ||
                    (item.label === 'Results' && active === 'analysis') ||
                    (index !== 2 && pathname === item.href)
                    ? 'bg-white/[.07] text-white'
                    : 'text-slate-500 hover:bg-white/[.04] hover:text-slate-200',
                )}
              >
                <span className="w-4 font-mono text-[10px] text-slate-600">
                  {index === 0
                    ? '◈'
                    : index === 1
                      ? '◫'
                      : index === 2
                        ? '↳'
                        : index === 3
                          ? '◌'
                          : '⚙'}
                </span>
                {item.label}
              </Link>
            ))}
          </div>
          <div className="mb-3 flex items-center justify-between px-3">
            <span className="font-mono text-[9px] uppercase tracking-[.2em] text-slate-600">
              Guided workflow
            </span>
            <span className="font-mono text-[10px] text-slate-700">
              {stages.some(([id]) => id === active)
                ? String(stages.findIndex(([id]) => id === active) + 1).padStart(
                    2,
                    '0',
                  )
                : '—'}
              /08
            </span>
          </div>
          <div className="relative">
            {stages.map(([id, Icon], index) => {
              const current = id === active
              return (
                <div key={id} className="relative">
                  <Link
                    href={`/${id}`}
                    onClick={() => go(`/${id}`, id)}
                    className={classNames(
                      'group relative flex items-center gap-3 px-3 py-2.5 text-[13px]',
                      current
                        ? 'text-cyan-300'
                        : 'text-slate-500 hover:text-slate-300',
                    )}
                  >
                    <span
                      className={classNames(
                        'grid size-7 place-items-center border font-mono text-[10px]',
                        current
                          ? 'border-cyan-400/50 bg-cyan-400/10 text-cyan-300'
                          : 'border-white/[.08] text-slate-600',
                      )}
                    >
                      <Icon />
                    </span>
                    <span>
                      {String(index + 1).padStart(2, '0')}{' '}
                      {id === 'data'
                        ? 'Data & Ground Truth'
                        : id === 'error-injection'
                          ? 'Error Injection'
                          : id === 'error-visualization'
                            ? 'Error Visualization'
                            : id === 'application-impact'
                              ? 'Application Impact'
                              : id[0].toUpperCase() + id.slice(1)}
                    </span>
                    {current && <ChevronRight className="ml-auto" />}
                  </Link>
                  {index < stages.length - 1 && (
                    <div className="ml-[27px] h-2 border-l border-dashed border-white/[.1]" />
                  )}
                </div>
              )
            })}
          </div>
        </nav>
        <div className="border-t border-white/[.07] p-4">
          <ApiConnectionBadge variant="sidebar" />
          <Link
            href="/settings"
            className="flex items-center gap-3 px-2 py-2 text-xs text-slate-500 hover:text-slate-200"
          >
            <Settings2 /> Settings
          </Link>
        </div>
      </aside>
      <div className="lg:pl-[276px]">
        <header className="sticky top-0 z-30 flex h-[78px] items-center justify-between border-b border-white/[.07] bg-[#070a0f]/90 px-5 backdrop-blur-xl lg:px-8">
          <div className="flex items-center gap-3">
            <button
              aria-label="Open navigation"
              className="text-slate-400 lg:hidden"
              onClick={() => setOpen(!open)}
            >
              <Menu />
            </button>
            <div>
              <div className="font-mono text-[10px] uppercase tracking-[.2em] text-slate-600">
                Saved research results
              </div>
              <div className="text-sm font-medium text-slate-200">
                ETH prediction{' '}
                <span className="px-1.5 text-slate-700">/</span> MOT17 tracking
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ApiConnectionBadge />
            <div className="grid size-8 place-items-center border border-white/[.1] bg-white/[.04] font-mono text-xs">
              RV
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-[1500px] px-5 py-7 lg:px-10 lg:py-10">
          {children}
        </main>
      </div>
    </div>
  )
}

export function PageHeader({eyebrow,title,description,stage}:{eyebrow:string;title:string;description:string;stage?:string}){return <div className="mb-8 flex flex-col justify-between gap-5 border-b border-white/[.07] pb-7 xl:flex-row xl:items-end"><div><div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.25em] text-cyan-400"><span className="size-1.5 bg-cyan-400" /> {eyebrow}</div><h1 className="max-w-4xl text-2xl font-semibold tracking-tight text-white sm:text-3xl">{title}</h1><p className="mt-3 max-w-3xl text-sm leading-6 text-slate-500">{description}</p></div>{stage&&<div className="shrink-0 font-mono text-[10px] uppercase tracking-[.2em] text-slate-600">Stage {stage} / 08</div>}</div>}
export function Panel({title,eyebrow,children,className=''}:{title?:string;eyebrow?:string;children:React.ReactNode;className?:string}){return <section className={classNames('min-w-0 border border-white/[.08] bg-[#0b1017]',className)}><div className="flex items-center justify-between border-b border-white/[.07] px-5 py-4">{title&&<div><div className="text-sm font-medium text-slate-200">{title}</div>{eyebrow&&<div className="mt-1 font-mono text-[9px] uppercase tracking-[.18em] text-slate-600">{eyebrow}</div>}</div>}<span className="size-1.5 bg-cyan-400/70" /></div><div className="p-5">{children}</div></section>}
export function MetricGrid({items}:{items:{label:string;value:string;detail?:string}[]}){return <div className="grid grid-cols-2 gap-px border border-white/[.08] bg-white/[.08] md:grid-cols-4">{items.map(item=><div key={item.label} className="bg-[#0b1017] p-4"><div className="font-mono text-[10px] uppercase tracking-wider text-slate-600">{item.label}</div><div className="mt-2 font-mono text-xl text-slate-300">{item.value}</div>{item.detail&&<div className="mt-1 text-[11px] text-slate-600">{item.detail}</div>}</div>)}</div>}
export function EmptyState({title='Waiting for backend result',message='No result has been generated for this experiment yet.'}:{title?:string;message?:string}){return <div className="grid min-h-36 place-items-center border border-dashed border-white/[.12] text-center"><div><div className="font-mono text-xs uppercase tracking-wider text-slate-500">{title}</div><p className="mt-2 text-xs text-slate-600">{message}</p></div></div>}
export function Status({label,kind='placeholder'}:{label:string;kind?:'success'|'warning'|'placeholder'}){return <span className={classNames('inline-flex items-center gap-2 border px-2 py-1 font-mono text-[10px] uppercase tracking-wider',kind==='success'?'border-emerald-400/20 text-emerald-300':kind==='warning'?'border-amber-400/20 text-amber-300':'border-white/[.1] text-slate-500')}><span className="size-1.5 bg-current" />{label}</span>}
export function WorkflowFooter({stage}:{stage:WorkflowStage}){const {setStage}=useExperiment(); const index=Object.keys(icons).indexOf(stage); const prev=Object.keys(icons)[Math.max(0,index-1)] as WorkflowStage; const next=Object.keys(icons)[Math.min(Object.keys(icons).length-1,index+1)] as WorkflowStage; return <div className="mt-8 flex flex-col justify-between gap-3 border-t border-white/[.07] pt-5 sm:flex-row"><Link href={`/${prev}`} onClick={()=>setStage(prev)} className="flex items-center gap-2 text-xs text-slate-500 hover:text-white"><ArrowLeft /> Previous</Link><div className="flex gap-2"><Link href={`/${next}`} onClick={()=>setStage(next)} className="flex items-center gap-2 bg-cyan-300 px-4 py-2.5 text-xs font-semibold text-slate-950 hover:bg-cyan-200">Next stage <ArrowRight /></Link></div></div>}

function Field({label,children}:{label:string;children:React.ReactNode}){return <label className="block"><span className="mb-2 block font-mono text-[10px] uppercase tracking-wider text-slate-600">{label}</span>{children}</label>}








export function ImpactPage(){return <AppShell stage="application-impact"><PageHeader eyebrow="Stage 08 / Downstream measurement" title="Application impact" description="Downstream collision and risk measurements. No application-impact experiment has been run, so no values exist to display." stage="08"/><div className="mb-5 flex items-center gap-2 border border-amber-500/20 bg-amber-500/[.05] px-3 py-2 font-mono text-[10px] uppercase tracking-[.14em] text-amber-300" role="status"><AlertTriangle /> Not measured · no saved application-impact results exist</div><div className="mb-5 grid gap-3 md:grid-cols-4">{[['Prediction degradation','Observed error'],['Trajectory deviation','Measured distance'],['Conflict / risk','Measurement'],['Application metrics','Comparison']].map(([a,b],i)=><div key={a} className="flex items-center gap-3 border border-white/[.08] bg-[#0b1017] p-4"><span className="font-mono text-xs text-cyan-300">0{i+1}</span><div><div className="text-xs text-slate-300">{a}</div><div className="mt-1 text-[10px] text-slate-600">{b}</div></div></div>)}</div><Panel title="Planned measurement table" eyebrow="Not measured · no values available"><div className="overflow-x-auto"><table className="w-full min-w-[600px] text-left text-xs"><thead className="border-b border-white/[.08] font-mono text-[10px] uppercase tracking-wider text-slate-600"><tr>{['Metric','Clean condition','Error-controlled','Difference'].map(x=><th key={x} className="pb-3 pr-5 font-normal">{x}</th>)}</tr></thead><tbody>{['d_conflict','d_warn','tau_event','TP','FP','FN','Missed warning rate','False warning rate'].map(x=><tr key={x} className="border-b border-white/[.05]"><td className="py-3 pr-5 text-slate-300">{x}</td><td className="py-3 pr-5 font-mono text-slate-600">—</td><td className="py-3 pr-5 font-mono text-slate-600">—</td><td className="py-3 font-mono text-slate-600">—</td></tr>)}</tbody></table></div></Panel><div className="mt-5 grid gap-5 xl:grid-cols-2"><Panel title="Application impact visualization" eyebrow="Warning timeline / risk trajectory"><EmptyState title="No impact measurements" message="No saved artifact exists for this stage."/></Panel><Panel title="Interpretation boundary" eyebrow="Scientific integrity"><div className="border border-cyan-400/20 bg-cyan-400/[.04] p-4 text-sm leading-6 text-slate-400">This view reports measured experimental effects only. It does not claim that a tracking error causally produces a specific real-world outcome, identify a universally worst error, or recommend a tracker.</div></Panel></div><WorkflowFooter stage="application-impact"/></AppShell>}

export function SettingsPage({children}:{children?:React.ReactNode}){
  const health = useApiHealth()
  const connected = health.data?.status === 'ok'
  const statusLabel = health.isLoading
    ? 'Checking results API'
    : connected
      ? `Results API connected · ${health.data?.version ?? ''}`
      : health.error ?? 'Results API unavailable'
  return <AppShell><PageHeader eyebrow="Workspace settings" title="Settings" description="Read-only results API connection and the saved configuration behind the displayed results."/><div className="grid gap-5 xl:grid-cols-2"><Panel title="Backend configuration" eyebrow={connected ? 'Connected' : health.isLoading ? 'Checking' : 'Unavailable'}><div className="flex flex-col gap-5"><Field label="API endpoint"><input readOnly value={getApiBaseUrl()} className="w-full border border-white/[.1] bg-[#080c12] px-3 py-2.5 text-xs text-slate-300" /></Field><Status label={statusLabel} kind={connected ? 'success' : undefined}/></div></Panel>{children}</div></AppShell>
}

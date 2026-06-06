// src/pages/Experiments.tsx
import { useState } from 'react'
import { FlaskConical, Download, Eye, Filter } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { TopBar } from '@/components/layout/TopBar'
import {
  Card, CardHeader, Button, StatusBadge,
  SectionHeading, EmptyState, Spinner,
} from '@/components/ui'
import { useRuns, useExport, useMetrics } from '@/api/hooks/useRuns'
import { useProjects } from '@/api/hooks/useProjects'
import { MetricsChart } from '@/components/charts/MetricsChart'
import { useAppStore } from '@/store/appStore'
import { formatDistanceToNow } from 'date-fns'
import type { RunRead } from '@/types/api'

function RunRow({ run, onView, onExport }: { run: RunRead; onView: () => void; onExport: () => void }) {
  const [expanded, setExpanded] = useState(false)
  const { data: metrics } = useMetrics(run.id, expanded)

  const psnr = (run.final_metrics as any)?.psnr
  const ssim = (run.final_metrics as any)?.ssim
  const lpips = (run.final_metrics as any)?.lpips

  return (
    <div className="border-b border-panel-border last:border-b-0">
      <div
        className="px-5 py-4 flex items-center gap-4 hover:bg-surface/40 transition-colors cursor-pointer"
        onClick={() => setExpanded((v) => !v)}
      >
        {/* Name */}
        <div className="flex-1 min-w-0">
          <p className="text-sm text-white font-mono truncate">{run.run_name}</p>
          <p className="text-xs text-dim mt-0.5">{formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}</p>
        </div>

        {/* Metrics */}
        <div className="hidden md:flex items-center gap-6 text-xs font-mono">
          <div className="text-center">
            <p className="text-dim mb-0.5">PSNR</p>
            <p className="text-amber-400 font-bold">{psnr != null ? Number(psnr).toFixed(2) : '—'}</p>
          </div>
          <div className="text-center">
            <p className="text-dim mb-0.5">SSIM</p>
            <p className="text-jade font-bold">{ssim != null ? Number(ssim).toFixed(4) : '—'}</p>
          </div>
          <div className="text-center">
            <p className="text-dim mb-0.5">LPIPS</p>
            <p className="text-cobalt font-bold">{lpips != null ? Number(lpips).toFixed(4) : '—'}</p>
          </div>
        </div>

        <StatusBadge status={run.status} />

        {/* Actions */}
        <div className="flex items-center gap-2 shrink-0" onClick={(e) => e.stopPropagation()}>
          <Button size="sm" onClick={onView}>
            <Eye size={12} /> View
          </Button>
          <Button size="sm" onClick={onExport} disabled={run.status !== 'success'}>
            <Download size={12} /> Export
          </Button>
        </div>
      </div>

      {/* Expanded metrics chart */}
      {expanded && (
        <div className="px-5 pb-5 bg-void/40 border-t border-panel-border animate-slide-up">
          {!metrics || metrics.metrics.length === 0 ? (
            <div className="flex items-center justify-center gap-2 py-8 text-xs font-mono text-dim">
              <Spinner size={14} /> Loading metrics…
            </div>
          ) : (
            <div className="pt-4">
              <MetricsChart data={metrics.metrics} height={200} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function Experiments() {
  const navigate = useNavigate()
  const { data: projects } = useProjects()
  const [projectFilter, setProjectFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const { data: runsData, isLoading } = useRuns(projectFilter || undefined)
  const exportRun = useExport()
  const { setSelectedRun } = useAppStore()

  const runs = (runsData?.runs ?? []).filter(
    (r) => !statusFilter || r.status === statusFilter,
  )

  const handleExport = async (runId: string) => {
    await exportRun.mutateAsync({ run_id: runId, formats: ['ply', 'splat'] })
  }

  const handleView = (runId: string) => {
    setSelectedRun(runId)
    navigate('/viewer')
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <TopBar title="Experiments" subtitle="Compare and manage training runs" />

      <div className="flex-1 overflow-y-auto p-6 animate-fade-in">
        <div className="max-w-6xl mx-auto space-y-4">
          {/* Filters */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs font-mono text-dim">
              <Filter size={12} />
              <span>FILTER</span>
            </div>
            <select
              value={projectFilter}
              onChange={(e) => setProjectFilter(e.target.value)}
              className="bg-surface border border-panel-border rounded-md px-3 py-1.5 text-xs text-white font-mono
                         focus:border-amber-500/50 focus:outline-none transition-colors"
            >
              <option value="">All projects</option>
              {projects?.projects.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-surface border border-panel-border rounded-md px-3 py-1.5 text-xs text-white font-mono
                         focus:border-amber-500/50 focus:outline-none transition-colors"
            >
              <option value="">All statuses</option>
              {['pending', 'running', 'success', 'failed'].map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <span className="text-xs font-mono text-dim ml-auto">
              {runs.length} run{runs.length !== 1 ? 's' : ''}
            </span>
          </div>

          {/* Summary stats row */}
          {runs.length > 0 && (
            <div className="grid grid-cols-4 gap-3">
              {[
                { label: 'Runs', value: runs.length },
                { label: 'Successful', value: runs.filter((r) => r.status === 'success').length },
                {
                  label: 'Best PSNR',
                  value: Math.max(...runs.map((r) => (r.final_metrics as any)?.psnr ?? 0)).toFixed(2),
                },
                {
                  label: 'Avg PSNR',
                  value: runs.filter((r) => (r.final_metrics as any)?.psnr).length
                    ? (
                        runs.reduce((a, r) => a + ((r.final_metrics as any)?.psnr ?? 0), 0) /
                        runs.filter((r) => (r.final_metrics as any)?.psnr).length
                      ).toFixed(2)
                    : '—',
                },
              ].map(({ label, value }) => (
                <Card key={label} className="px-4 py-3">
                  <p className="text-[10px] font-mono text-dim uppercase tracking-widest">{label}</p>
                  <p className="text-lg font-mono font-bold text-white mt-0.5">{value}</p>
                </Card>
              ))}
            </div>
          )}

          {/* Runs table */}
          <Card>
            <CardHeader>
              <SectionHeading>Runs</SectionHeading>
            </CardHeader>
            {isLoading ? (
              <div className="flex justify-center py-16"><Spinner /></div>
            ) : !runs.length ? (
              <EmptyState
                icon={<FlaskConical size={20} />}
                title="No experiments found"
                desc="Adjust filters or start a new training run."
              />
            ) : (
              <div>
                {runs.map((run) => (
                  <RunRow
                    key={run.id}
                    run={run}
                    onView={() => handleView(run.id)}
                    onExport={() => handleExport(run.id)}
                  />
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
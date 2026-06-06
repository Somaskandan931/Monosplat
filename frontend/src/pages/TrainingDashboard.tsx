// src/pages/TrainingDashboard.tsx
import { useState } from 'react'
import { Play, RotateCcw, Cpu, Settings, ChevronDown, ChevronUp } from 'lucide-react'
import { TopBar } from '@/components/layout/TopBar'
import {
  Card, CardHeader, CardBody, Button, ProgressBar,
  StatusBadge, SectionHeading, EmptyState, StatCard, Spinner,
} from '@/components/ui'
import { useProjects } from '@/api/hooks/useProjects'
import { useRuns, useTrain, useMetrics } from '@/api/hooks/useRuns'
import { useJob } from '@/api/hooks/useJob'
import { useAppStore } from '@/store/appStore'
import { MetricsChart } from '@/components/charts/MetricsChart'
import { formatDistanceToNow } from 'date-fns'

function TrainForm({ onLaunched }: { onLaunched: (jobId: string, runId: string) => void }) {
  const { data: projects } = useProjects()
  const train = useTrain()
  const [form, setForm] = useState({
    project_id: '',
    sparse_path: 'data/sparse',
    image_dir: 'data/images',
    iterations: '30000',
    lr: '0.001',
  })
  const [showAdvanced, setShowAdvanced] = useState(false)

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }))

  const handleSubmit = async () => {
    const resp = await train.mutateAsync({
      project_id: form.project_id,
      sparse_path: form.sparse_path,
      image_dir: form.image_dir,
      config_overrides: {
        iterations: parseInt(form.iterations),
        learning_rate: parseFloat(form.lr),
      },
    })
    onLaunched(resp.job_id, resp.run_id)
  }

  const Field = ({ label, k, type = 'text' }: { label: string; k: string; type?: string }) => (
    <div>
      <label className="text-xs font-mono text-dim mb-1.5 block">{label}</label>
      <input
        type={type}
        value={(form as any)[k]}
        onChange={set(k)}
        className="w-full bg-surface border border-panel-border rounded-md px-3 py-2 text-sm text-white font-mono
                   placeholder:text-muted focus:border-amber-500/50 focus:outline-none transition-colors"
      />
    </div>
  )

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-amber-400" />
          <SectionHeading>Launch Training Run</SectionHeading>
        </div>
      </CardHeader>
      <CardBody className="space-y-4">
        <div>
          <label className="text-xs font-mono text-dim mb-1.5 block">Project</label>
          <select
            value={form.project_id}
            onChange={(e) => setForm((f) => ({ ...f, project_id: e.target.value }))}
            className="w-full bg-surface border border-panel-border rounded-md px-3 py-2 text-sm text-white font-mono
                       focus:border-amber-500/50 focus:outline-none transition-colors"
          >
            <option value="">Select a project…</option>
            {projects?.projects.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Field label="Sparse Path" k="sparse_path" />
          <Field label="Image Directory" k="image_dir" />
        </div>

        {/* Advanced toggle */}
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex items-center gap-1.5 text-xs font-mono text-dim hover:text-ghost transition-colors"
        >
          <Settings size={11} />
          Advanced options
          {showAdvanced ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-2 gap-3 border-t border-panel-border pt-4">
            <Field label="Iterations" k="iterations" type="number" />
            <Field label="Learning Rate" k="lr" type="number" />
          </div>
        )}

        {train.error && (
          <p className="text-xs text-crimson font-mono">{(train.error as Error).message}</p>
        )}

        <Button
          variant="primary"
          className="w-full justify-center"
          onClick={handleSubmit}
          loading={train.isPending}
          disabled={!form.project_id}
        >
          <Play size={13} /> Start Training
        </Button>
      </CardBody>
    </Card>
  )
}

function ActiveRunPanel({ jobId, runId }: { jobId: string; runId: string }) {
  const { data: job } = useJob(jobId)
  const { data: metrics } = useMetrics(runId, true)

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <SectionHeading>Active Run — {runId.slice(0, 8)}…</SectionHeading>
          {job && <StatusBadge status={job.status} />}
        </div>
      </CardHeader>
      <CardBody className="space-y-4">
        {job && (
          <>
            <ProgressBar value={job.progress} />
            <div className="flex justify-between text-xs font-mono text-dim">
              <span>{job.message ?? 'Processing…'}</span>
              <span>{job.progress.toFixed(0)}%</span>
            </div>
          </>
        )}
        {metrics && metrics.metrics.length > 0 && (
          <div className="pt-2">
            <p className="text-xs font-mono text-dim mb-3">{metrics.count} data points</p>
            <MetricsChart data={metrics.metrics} keys={['psnr', 'loss']} />
          </div>
        )}
        {(!metrics || metrics.metrics.length === 0) && (
          <div className="flex items-center justify-center gap-2 py-8 text-dim text-xs font-mono">
            <Spinner size={14} /> Waiting for first metrics…
          </div>
        )}
      </CardBody>
    </Card>
  )
}

export default function TrainingDashboard() {
  const [active, setActive] = useState<{ jobId: string; runId: string } | null>(null)
  const { setActiveJob, setSelectedRun } = useAppStore()
  const { data: runsData, isLoading } = useRuns()
  const runs = runsData?.runs ?? []
  const activeRuns = runs.filter((r) => r.status === 'running')

  const handleLaunched = (jobId: string, runId: string) => {
    setActive({ jobId, runId })
    setActiveJob(jobId)
    setSelectedRun(runId)
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <TopBar title="Training" subtitle="Configure and monitor 3DGS training jobs" />
      <div className="flex-1 overflow-y-auto p-6 animate-fade-in">
        <div className="max-w-5xl mx-auto space-y-6">
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4">
            <StatCard label="Active Runs" value={activeRuns.length} icon={<Cpu size={16} />} accent={activeRuns.length > 0 ? 'text-amber-400' : 'text-dim'} />
            <StatCard label="Total Runs" value={runsData?.meta.total ?? 0} icon={<RotateCcw size={16} />} accent="text-cobalt" />
            <StatCard
              label="Best PSNR"
              value={
                runs.length
                  ? Math.max(...runs.map((r) => (r.final_metrics as any)?.psnr ?? 0)).toFixed(2)
                  : '—'
              }
              accent="text-jade"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <TrainForm onLaunched={handleLaunched} />

            {active ? (
              <ActiveRunPanel jobId={active.jobId} runId={active.runId} />
            ) : (
              <Card>
                <CardHeader><SectionHeading>Run History</SectionHeading></CardHeader>
                <CardBody className="p-0">
                  {isLoading ? (
                    <div className="flex justify-center py-8"><Spinner /></div>
                  ) : !runs.length ? (
                    <EmptyState icon={<Cpu size={20} />} title="No runs yet" />
                  ) : (
                    <div className="divide-y divide-panel-border">
                      {runs.slice(0, 6).map((r) => (
                        <div key={r.id} className="px-5 py-3 flex items-center justify-between hover:bg-surface/50 transition-colors">
                          <div>
                            <p className="text-sm text-white font-mono">{r.run_name}</p>
                            <p className="text-xs text-dim">{formatDistanceToNow(new Date(r.created_at), { addSuffix: true })}</p>
                          </div>
                          <div className="flex items-center gap-3">
                            {(r.final_metrics as any)?.psnr && (
                              <span className="text-xs font-mono text-amber-400">
                                PSNR {Number((r.final_metrics as any).psnr).toFixed(2)}
                              </span>
                            )}
                            <StatusBadge status={r.status} />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardBody>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
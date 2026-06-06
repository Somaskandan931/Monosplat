// src/pages/Dashboard.tsx
import { Activity, Box, Database, Cpu, TrendingUp, Clock, Zap } from 'lucide-react'
import { TopBar } from '@/components/layout/TopBar'
import { StatCard, Card, CardHeader, CardBody, StatusBadge, SectionHeading, EmptyState, Spinner } from '@/components/ui'
import { useProjects } from '@/api/hooks/useProjects'
import { useRuns } from '@/api/hooks/useRuns'
import { formatDistanceToNow } from 'date-fns'
import { Link } from 'react-router-dom'

export default function Dashboard() {
  const { data: projects, isLoading: loadingProjects } = useProjects()
  const { data: runsData, isLoading: loadingRuns } = useRuns()

  const runs = runsData?.runs ?? []
  const activeRuns = runs.filter((r) => r.status === 'running').length
  const completedRuns = runs.filter((r) => r.status === 'success').length
  const latestRun = runs[0]

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <TopBar title="Dashboard" subtitle="MonoSplat — 3D Gaussian Splatting pipeline" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6 animate-fade-in">
        {/* Stat row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Projects"
            value={loadingProjects ? '—' : projects?.meta.total ?? 0}
            icon={<Database size={16} />}
          />
          <StatCard
            label="Total Runs"
            value={loadingRuns ? '—' : runsData?.meta.total ?? 0}
            icon={<Box size={16} />}
            accent="text-cobalt"
          />
          <StatCard
            label="Active Jobs"
            value={activeRuns}
            icon={<Cpu size={16} />}
            accent={activeRuns > 0 ? 'text-amber-400' : 'text-dim'}
          />
          <StatCard
            label="Completed"
            value={completedRuns}
            icon={<TrendingUp size={16} />}
            accent="text-jade"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Recent Runs */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between">
                <SectionHeading>Recent Runs</SectionHeading>
                <Link to="/experiments" className="text-xs text-amber-400 hover:text-amber-300 font-mono">
                  View all →
                </Link>
              </div>
            </CardHeader>
            <CardBody className="p-0">
              {loadingRuns ? (
                <div className="flex justify-center py-12"><Spinner /></div>
              ) : runs.length === 0 ? (
                <EmptyState icon={<Box size={20} />} title="No runs yet" desc="Start a training job to see results here." />
              ) : (
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-panel-border">
                      {['Run', 'Status', 'PSNR', 'Started'].map((h) => (
                        <th key={h} className="px-5 py-2.5 text-left text-dim font-normal uppercase tracking-widest text-[10px]">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {runs.slice(0, 8).map((run) => (
                      <tr key={run.id} className="border-b border-panel-border/50 hover:bg-surface/50 transition-colors">
                        <td className="px-5 py-3 text-white truncate max-w-[180px]">{run.run_name}</td>
                        <td className="px-5 py-3"><StatusBadge status={run.status} /></td>
                        <td className="px-5 py-3 text-amber-400">
                          {(run.final_metrics as any)?.psnr?.toFixed(2) ?? '—'}
                        </td>
                        <td className="px-5 py-3 text-dim flex items-center gap-1">
                          <Clock size={10} />
                          {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CardBody>
          </Card>

          {/* Quick actions */}
          <div className="space-y-4">
            <Card>
              <CardHeader><SectionHeading>Quick Actions</SectionHeading></CardHeader>
              <CardBody className="space-y-2">
                {[
                  { to: '/datasets', label: 'Upload Dataset', icon: Database, accent: 'text-cobalt' },
                  { to: '/training', label: 'Start Training', icon: Zap,      accent: 'text-amber-400' },
                  { to: '/viewer',   label: 'Open 3D Viewer', icon: Box,      accent: 'text-jade' },
                ].map(({ to, label, icon: Icon, accent }) => (
                  <Link
                    key={to}
                    to={to}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-md bg-surface hover:bg-surface-hover border border-panel-border/50 hover:border-panel-border transition-all group"
                  >
                    <Icon size={14} className={accent} />
                    <span className="text-sm text-ghost group-hover:text-white transition-colors">{label}</span>
                  </Link>
                ))}
              </CardBody>
            </Card>

            <Card>
              <CardHeader><SectionHeading>System</SectionHeading></CardHeader>
              <CardBody className="space-y-3 text-xs font-mono">
                {[
                  { label: 'Backend', value: 'FastAPI 0.111', ok: true },
                  { label: 'DB',      value: 'SQLite',         ok: true },
                  { label: 'Workers', value: 'ThreadPool',     ok: true },
                ].map(({ label, value, ok }) => (
                  <div key={label} className="flex items-center justify-between">
                    <span className="text-dim">{label}</span>
                    <div className="flex items-center gap-1.5">
                      <span className={ok ? 'w-1.5 h-1.5 rounded-full bg-jade' : 'w-1.5 h-1.5 rounded-full bg-crimson'} />
                      <span className="text-ghost">{value}</span>
                    </div>
                  </div>
                ))}
              </CardBody>
            </Card>
          </div>
        </div>

        {/* Latest metrics preview */}
        {latestRun?.final_metrics && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <SectionHeading>Latest Run Metrics — {latestRun.run_name}</SectionHeading>
                <Activity size={14} className="text-amber-400" />
              </div>
            </CardHeader>
            <CardBody>
              <div className="grid grid-cols-3 gap-6 mb-4 text-center">
                {(['psnr', 'ssim', 'lpips'] as const).map((k) => {
                  const v = (latestRun.final_metrics as any)?.[k]
                  return (
                    <div key={k}>
                      <p className="text-xs font-mono text-dim uppercase tracking-widest mb-1">{k.toUpperCase()}</p>
                      <p className="font-mono font-bold text-xl text-white">
                        {v != null ? Number(v).toFixed(3) : '—'}
                      </p>
                    </div>
                  )
                })}
              </div>
            </CardBody>
          </Card>
        )}
      </div>
    </div>
  )
}
// src/components/layout/TopBar.tsx
import { Activity, Server } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import { useAppStore } from '@/store/appStore'
import { useJob } from '@/api/hooks/useJob'
import clsx from 'clsx'

function ApiStatus() {
  const { data, isError } = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: 1,
  })
  const online = !isError && data?.status === 'ok'
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span className={clsx('w-1.5 h-1.5 rounded-full', online ? 'bg-jade animate-pulse-slow' : 'bg-crimson')} />
      <span className={online ? 'text-jade' : 'text-crimson'}>
        {online ? 'API ONLINE' : 'API OFFLINE'}
      </span>
    </div>
  )
}

function ActiveJob() {
  const { activeJobId } = useAppStore()
  const { data: job } = useJob(activeJobId)
  if (!job || job.status === 'success' || job.status === 'failed') return null
  return (
    <div className="flex items-center gap-2 text-xs font-mono text-amber-400">
      <Activity size={12} className="animate-pulse" />
      <span>{job.job_type.toUpperCase()}</span>
      <span className="text-dim">{job.progress.toFixed(0)}%</span>
    </div>
  )
}

interface TopBarProps { title: string; subtitle?: string }

export function TopBar({ title, subtitle }: TopBarProps) {
  return (
    <header className="h-14 flex items-center justify-between px-6 border-b border-panel-border bg-panel/60 backdrop-blur shrink-0">
      <div>
        <h1 className="font-display font-700 text-base text-white tracking-wide">{title}</h1>
        {subtitle && <p className="text-xs text-dim font-body">{subtitle}</p>}
      </div>
      <div className="flex items-center gap-5">
        <ActiveJob />
        <div className="w-px h-4 bg-panel-border" />
        <div className="flex items-center gap-1.5 text-dim">
          <Server size={12} />
          <ApiStatus />
        </div>
      </div>
    </header>
  )
}

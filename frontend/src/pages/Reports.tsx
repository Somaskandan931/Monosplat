// src/pages/Reports.tsx
import { useState } from 'react'
import { FileText, ChevronRight, BarChart2, Clock, Download } from 'lucide-react'
import { TopBar } from '@/components/layout/TopBar'
import {
  Card, CardHeader, CardBody, Button, StatusBadge,
  SectionHeading, EmptyState, Spinner, StatCard,
} from '@/components/ui'
import { useRuns, useReport, useExport } from '@/api/hooks/useRuns'
import { MetricsChart } from '@/components/charts/MetricsChart'
import { useMetrics } from '@/api/hooks/useRuns'
import { formatDistanceToNow, format } from 'date-fns'
import type { RunRead } from '@/types/api'

function RunReport({ runId }: { runId: string }) {
  const { data: report, isLoading } = useReport(runId, true)
  const { data: metrics } = useMetrics(runId, true)
  const exportRun = useExport()

  if (isLoading) return <div className="flex justify-center py-12"><Spinner /></div>
  if (!report) return <p className="text-sm text-dim p-6">Report not found.</p>

  const fm = report.final_metrics as Record<string, number> | undefined

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-display font-700 text-white">{report.run_name}</h3>
          <p className="text-xs font-mono text-dim mt-0.5">
            {format(new Date(report.created_at), 'PPpp')}
            {report.finished_at && ` → ${format(new Date(report.finished_at), 'p')}`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={report.status} />
          <Button
            size="sm"
            variant="primary"
            onClick={() => exportRun.mutate({ run_id: runId, formats: ['ply', 'splat'] })}
            disabled={report.status !== 'success'}
            loading={exportRun.isPending}
          >
            <Download size={12} /> Export PLY + SPLAT
          </Button>
        </div>
      </div>

      {/* Metrics grid */}
      {fm && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {(['psnr', 'ssim', 'lpips', 'loss'] as const).map((k) =>
            fm[k] != null ? (
              <StatCard
                key={k}
                label={k.toUpperCase()}
                value={Number(fm[k]).toFixed(4)}
                accent={k === 'psnr' ? 'text-amber-400' : k === 'ssim' ? 'text-jade' : k === 'lpips' ? 'text-crimson' : 'text-violet'}
              />
            ) : null,
          )}
        </div>
      )}

      {/* Training curve */}
      {metrics && metrics.metrics.length > 0 && (
        <Card>
          <CardHeader>
            <SectionHeading>Training Curves</SectionHeading>
          </CardHeader>
          <CardBody>
            <MetricsChart data={metrics.metrics} height={260} />
          </CardBody>
        </Card>
      )}

      {/* Config snapshot */}
      {report.config_snapshot && (
        <Card>
          <CardHeader><SectionHeading>Config Snapshot</SectionHeading></CardHeader>
          <CardBody>
            <pre className="text-xs font-mono text-ghost overflow-x-auto whitespace-pre-wrap">
              {JSON.stringify(report.config_snapshot, null, 2)}
            </pre>
          </CardBody>
        </Card>
      )}

      {/* Paths */}
      <Card>
        <CardHeader><SectionHeading>Artifacts</SectionHeading></CardHeader>
        <CardBody className="space-y-2 text-xs font-mono">
          {[
            { label: 'Dataset', val: report.dataset_path },
            { label: 'Model',   val: report.model_path   },
          ].map(({ label, val }) => (
            <div key={label} className="flex items-center justify-between">
              <span className="text-dim">{label}</span>
              <span className="text-ghost truncate max-w-[320px]">{val ?? '—'}</span>
            </div>
          ))}
        </CardBody>
      </Card>
    </div>
  )
}

export default function Reports() {
  const { data: runsData, isLoading } = useRuns()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const runs = runsData?.runs ?? []

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <TopBar title="Reports" subtitle="Detailed run reports and training analytics" />

      <div className="flex-1 flex overflow-hidden">
        {/* Run list */}
        <div className="w-72 border-r border-panel-border flex flex-col bg-panel shrink-0">
          <div className="px-4 py-3 border-b border-panel-border">
            <SectionHeading>Runs ({runs.length})</SectionHeading>
          </div>
          <div className="flex-1 overflow-y-auto">
            {isLoading ? (
              <div className="flex justify-center py-8"><Spinner /></div>
            ) : !runs.length ? (
              <EmptyState icon={<FileText size={16} />} title="No runs" />
            ) : (
              runs.map((run) => (
                <button
                  key={run.id}
                  onClick={() => setSelectedId(run.id)}
                  className={`w-full text-left px-4 py-3 border-b border-panel-border/50 hover:bg-surface transition-colors
                    ${selectedId === run.id ? 'bg-surface border-l-2 border-l-amber-500' : ''}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono text-white truncate max-w-[160px]">{run.run_name}</span>
                    <StatusBadge status={run.status} />
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-dim">
                    <Clock size={9} />
                    {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                  </div>
                  {(run.final_metrics as any)?.psnr && (
                    <p className="text-[10px] font-mono text-amber-400 mt-0.5">
                      PSNR {Number((run.final_metrics as any).psnr).toFixed(2)}
                    </p>
                  )}
                </button>
              ))
            )}
          </div>
        </div>

        {/* Report detail */}
        <div className="flex-1 overflow-y-auto p-6">
          {!selectedId ? (
            <div className="flex items-center justify-center h-full">
              <EmptyState
                icon={<BarChart2 size={20} />}
                title="Select a run"
                desc="Choose a run from the left panel to view its full report."
              />
            </div>
          ) : (
            <RunReport runId={selectedId} />
          )}
        </div>
      </div>
    </div>
  )
}
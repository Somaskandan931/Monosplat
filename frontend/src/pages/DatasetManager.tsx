/// <reference types="vite/client" />
// src/pages/DatasetManager.tsx
import { useCallback, useState, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Upload, Database, CheckCircle, AlertCircle, Trash2,
  Film, Download, Map, ScanLine,
  Package, Loader2, XCircle,
  FileArchive,
} from 'lucide-react'
import clsx from 'clsx'
import { TopBar } from '@/components/layout/TopBar'
import {
  Card, CardHeader, CardBody, Button, ProgressBar,
  SectionHeading, EmptyState, Spinner,
} from '@/components/ui'
import { useProjects } from '@/api/hooks/useProjects'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'
import { useJob } from '@/api/hooks/useJob'
import type { ImportResultsResponse } from '@/types/api'

// ── Types ─────────────────────────────────────────────────────────────────────

interface UploadState {
  phase: 'idle' | 'uploading' | 'success' | 'error'
  progress: number
  filename?: string
  fileSize?: number
  jobId?: string
  error?: string
}

interface ImportState {
  phase: 'idle' | 'uploading' | 'success' | 'error'
  progress: number
  filename?: string
  result?: ImportResultsResponse
  error?: string
}

type StageStatus = 'waiting' | 'running' | 'done' | 'error'

interface PipelineStage {
  id: string
  label: string
  sublabel: string
  icon: React.ReactNode
  stat?: string
  threshold: number
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtBytes(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 ** 2).toFixed(1)} MB`
}

function stageStatus(
  threshold: number,
  nextThreshold: number | null,
  progress: number,
  jobStatus: string,
): StageStatus {
  if (jobStatus === 'failed') {
    if (progress >= threshold && (nextThreshold === null || progress < nextThreshold)) return 'error'
    if (progress >= threshold) return 'done'
    return 'waiting'
  }
  if (progress >= threshold) return 'done'
  if (progress > (threshold - 30) && progress < threshold) return 'running'
  return 'waiting'
}

// ── Stage icon ────────────────────────────────────────────────────────────────

function StageIcon({ status, icon }: { status: StageStatus; icon: React.ReactNode }) {
  if (status === 'done') {
    return (
      <div className="w-9 h-9 rounded-full bg-jade/15 border border-jade/30 flex items-center justify-center shrink-0">
        <CheckCircle size={15} className="text-jade" />
      </div>
    )
  }
  if (status === 'running') {
    return (
      <div className="w-9 h-9 rounded-full bg-amber-500/10 border border-amber-500/30 flex items-center justify-center shrink-0 relative">
        <span className="absolute inset-0 rounded-full border border-amber-500/20 animate-ping" />
        <Loader2 size={15} className="text-amber-400 animate-spin" />
      </div>
    )
  }
  if (status === 'error') {
    return (
      <div className="w-9 h-9 rounded-full bg-crimson/10 border border-crimson/30 flex items-center justify-center shrink-0">
        <XCircle size={15} className="text-crimson" />
      </div>
    )
  }
  return (
    <div className="w-9 h-9 rounded-full bg-surface border border-panel-border flex items-center justify-center shrink-0 opacity-40">
      {icon}
    </div>
  )
}

// ── Pipeline stepper ──────────────────────────────────────────────────────────

const STAGES: PipelineStage[] = [
  {
    id: 'frames',
    label: 'Frame Extraction',
    sublabel: 'FFmpeg · blur & exposure filtering',
    icon: <Film size={14} className="text-dim" />,
    threshold: 25,
  },
  {
    id: 'colmap',
    label: 'COLMAP',
    sublabel: 'Feature extraction · sparse reconstruction',
    icon: <Map size={14} className="text-dim" />,
    threshold: 60,
  },
  {
    id: 'normalize',
    label: 'Scene Normalization',
    sublabel: 'Camera centres → unit ball',
    icon: <ScanLine size={14} className="text-dim" />,
    threshold: 80,
  },
  {
    id: 'package',
    label: 'Colab Package',
    sublabel: 'ZIP ready for GPU training',
    icon: <Package size={14} className="text-dim" />,
    threshold: 100,
  },
]

function PipelineStepper({ jobId }: { jobId: string }) {
  const { data: job } = useJob(jobId)

  if (!job) {
    return (
      <Card>
        <CardBody className="flex items-center gap-3 py-6">
          <Spinner size={16} />
          <span className="text-xs font-mono text-dim">Connecting to pipeline…</span>
        </CardBody>
      </Card>
    )
  }

  const progress = job.progress ?? 0
  const jobStatus = job.status

  const statMap: Record<string, string> = {}
  const msg = job.message ?? ''
  const frameMatch = msg.match(/(\d+)\s*frame/i)
  const pointMatch = msg.match(/(\d+)\s*(3d\s*)?point/i)
  const zipMatch = msg.match(/zip[:\s]+([^\s]+\.zip)/i)
  if (frameMatch) statMap['frames'] = `${frameMatch[1]} frames`
  if (pointMatch) statMap['colmap'] = `${Number(pointMatch[1]).toLocaleString()} pts`
  if (zipMatch) statMap['package'] = zipMatch[1]

  const StatusBadge = ({ status }: { status: string }) => {
    const colour =
      status === 'success' ? 'text-jade border-jade/30 bg-jade/10' :
      status === 'failed'  ? 'text-crimson border-crimson/30 bg-crimson/10' :
      status === 'running' ? 'text-amber-400 border-amber-500/30 bg-amber-500/10' :
      'text-dim border-panel-border bg-surface'
    return (
      <span className={clsx('text-[10px] font-mono px-2 py-0.5 rounded-full border capitalize', colour)}>
        {status}
      </span>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <SectionHeading>Pipeline</SectionHeading>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-dim">{progress.toFixed(0)}%</span>
            <StatusBadge status={jobStatus} />
          </div>
        </div>
      </CardHeader>

      <CardBody className="space-y-5">
        <ProgressBar value={progress} />

        {msg && (
          <p className="text-[11px] font-mono text-dim leading-relaxed truncate" title={msg}>
            {msg}
          </p>
        )}

        <div className="space-y-0">
          {STAGES.map((stage, i) => {
            const nextThreshold = STAGES[i + 1]?.threshold ?? null
            const status = stageStatus(stage.threshold, nextThreshold, progress, jobStatus)

            return (
              <div key={stage.id} className="flex gap-3">
                <div className="flex flex-col items-center">
                  <StageIcon status={status} icon={stage.icon} />
                  {i < STAGES.length - 1 && (
                    <div
                      className={clsx(
                        'w-px flex-1 min-h-[28px] mt-1 mb-1 transition-colors duration-700',
                        status === 'done' ? 'bg-jade/30' : 'bg-panel-border',
                      )}
                    />
                  )}
                </div>

                <div
                  className={clsx(
                    'pb-7 flex-1 pt-1.5 transition-opacity duration-300',
                    i === STAGES.length - 1 ? 'pb-0' : '',
                    status === 'waiting' ? 'opacity-35' : 'opacity-100',
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p
                        className={clsx(
                          'text-xs font-mono font-semibold',
                          status === 'done'    ? 'text-jade' :
                          status === 'running' ? 'text-amber-400' :
                          status === 'error'   ? 'text-crimson' :
                          'text-ghost',
                        )}
                      >
                        {stage.label}
                      </p>
                      <p className="text-[10px] font-mono text-dim mt-0.5">{stage.sublabel}</p>
                    </div>

                    {(statMap[stage.id] || status === 'running') && (
                      <span
                        className={clsx(
                          'text-[10px] font-mono px-2 py-0.5 rounded border shrink-0',
                          status === 'done'
                            ? 'bg-jade/10 text-jade border-jade/20'
                            : 'bg-amber-500/10 text-amber-400 border-amber-500/20 animate-pulse',
                        )}
                      >
                        {statMap[stage.id] ?? 'running…'}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {jobStatus === 'success' && (
          <div className="mt-1 rounded-lg border border-jade/20 bg-jade/5 p-4 space-y-3">
            <div className="flex items-center gap-2">
              <CheckCircle size={13} className="text-jade" />
              <span className="text-xs font-mono text-jade font-semibold">Preprocessing complete</span>
            </div>
            <p className="text-[11px] font-mono text-dim">
              Download the Colab training package, upload it to your Colab notebook, train, then import results below.
            </p>
            <a
              href={`${import.meta.env.VITE_API_URL ?? '/api'}/download/${jobId}/colab-package`}
              download="colab_training_package.zip"
              className="inline-flex items-center gap-2 rounded-md border border-jade/30 bg-jade/10 px-3 py-2 text-xs font-mono text-jade hover:bg-jade/20 transition-colors"
            >
              <Download size={12} />
              Download colab_training_package.zip
            </a>
          </div>
        )}

        {jobStatus === 'failed' && job.error && (
          <div className="rounded-lg border border-crimson/20 bg-crimson/5 p-3">
            <div className="flex items-center gap-2 mb-1">
              <AlertCircle size={12} className="text-crimson" />
              <span className="text-xs font-mono text-crimson font-semibold">Pipeline failed</span>
            </div>
            <p className="text-[11px] font-mono text-dim">{job.error}</p>
          </div>
        )}
      </CardBody>
    </Card>
  )
}

// ── Drop zone ─────────────────────────────────────────────────────────────────

function DropZone({
  onFile,
  accept,
  label,
  sublabel,
  disabled,
}: {
  onFile: (f: File) => void
  accept: Record<string, string[]>
  label: string
  sublabel?: string
  disabled?: boolean
}) {
  const onDrop = useCallback((files: File[]) => { if (files[0]) onFile(files[0]) }, [onFile])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept, multiple: false, disabled })

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200 group overflow-hidden',
        disabled
          ? 'opacity-40 cursor-not-allowed border-panel-border'
          : isDragActive
          ? 'border-amber-500/50 bg-amber-500/5'
          : 'border-panel-border hover:border-amber-500/20 hover:bg-surface/40',
      )}
    >
      <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-amber-500/20 rounded-tl-xl pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-amber-500/20 rounded-br-xl pointer-events-none" />
      <input {...getInputProps()} />
      <Upload
        size={24}
        className={clsx(
          'mx-auto mb-3 transition-colors duration-200',
          isDragActive ? 'text-amber-400' : 'text-dim group-hover:text-amber-400/50',
        )}
      />
      <p className="text-sm font-mono text-ghost">{isDragActive ? 'Release to upload' : label}</p>
      {sublabel && <p className="text-[11px] font-mono text-dim mt-1">{sublabel}</p>}
    </div>
  )
}

// ── Video file info card ───────────────────────────────────────────────────────

function VideoInfo({ filename, fileSize }: { filename: string; fileSize: number }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-panel-border bg-surface/50 px-4 py-3">
      <div className="w-8 h-8 rounded-md bg-amber-500/10 border border-amber-500/20 flex items-center justify-center shrink-0">
        <Film size={14} className="text-amber-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-mono text-white truncate">{filename}</p>
        <p className="text-[10px] font-mono text-dim">{fmtBytes(fileSize)}</p>
      </div>
      <div className="flex items-center gap-1.5 text-[10px] font-mono text-jade">
        <CheckCircle size={10} />
        Uploaded
      </div>
    </div>
  )
}

// ── Import Results section ────────────────────────────────────────────────────

function ImportResultsSection({ defaultJobId }: { defaultJobId?: string }) {
  const [importState, setImportState] = useState<ImportState>({ phase: 'idle', progress: 0 })
  const [jobIdInput, setJobIdInput] = useState(defaultJobId ?? '')
  const { setActiveJob } = useAppStore()

  // Keep job ID input in sync if parent updates it (e.g. after upload)
  useEffect(() => {
    if (defaultJobId) setJobIdInput(defaultJobId)
  }, [defaultJobId])

  const handleResultsFile = async (file: File) => {
    const jobId = jobIdInput.trim()
    if (!jobId) return
    setImportState({ phase: 'uploading', progress: 0, filename: file.name })
    try {
      const resp = await api.importResults(jobId, file, (pct) =>
        setImportState(s => ({ ...s, progress: pct })),
      )
      setImportState({ phase: 'success', progress: 100, filename: file.name, result: resp })
      // Update the viewer store so the Viewer tab auto-loads
      setActiveJob(jobId)
    } catch (err: any) {
      setImportState({ phase: 'error', progress: 0, error: err.message })
    }
  }

  const reset = () => setImportState({ phase: 'idle', progress: 0 })

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded bg-cobalt/10 border border-cobalt/20 flex items-center justify-center">
            <span className="text-[9px] font-mono text-cobalt font-bold">03</span>
          </div>
          <SectionHeading>Import Colab Results</SectionHeading>
        </div>
      </CardHeader>
      <CardBody className="space-y-4">
        <p className="text-[11px] font-mono text-dim leading-relaxed">
          After training on Google Colab, run the export cell to produce{' '}
          <span className="text-ghost">results.zip</span> (containing{' '}
          <span className="text-ghost">exports/final.ply</span> and optionally{' '}
          <span className="text-ghost">exports/final.splat</span>), then upload it here.
        </p>

        {/* Job ID field */}
        <div>
          <label className="text-[10px] font-mono text-dim uppercase tracking-widest mb-1.5 block">
            Pipeline Job ID
          </label>
          <input
            value={jobIdInput}
            onChange={e => setJobIdInput(e.target.value)}
            placeholder="auto-filled after upload, or paste manually"
            className="w-full bg-surface border border-panel-border rounded-md px-3 py-2 text-xs text-white font-mono placeholder:text-muted focus:border-amber-500/50 focus:outline-none transition-colors"
          />
        </div>

        {importState.phase === 'idle' && (
          <DropZone
            onFile={handleResultsFile}
            accept={{ 'application/zip': ['.zip'], 'application/octet-stream': ['.zip'] }}
            label="Drag results.zip here"
            sublabel="The ZIP exported from Colab Cell 11"
            disabled={!jobIdInput.trim()}
          />
        )}

        {!jobIdInput.trim() && importState.phase === 'idle' && (
          <p className="text-[10px] font-mono text-dim text-center -mt-2">
            Enter a Job ID above to enable the drop zone
          </p>
        )}

        {importState.phase === 'uploading' && (
          <div className="space-y-3 py-3">
            <div className="flex items-center gap-3">
              <Spinner size={15} />
              <div>
                <p className="text-xs font-mono text-ghost">{importState.filename}</p>
                <p className="text-[10px] font-mono text-dim">Uploading results…</p>
              </div>
            </div>
            <ProgressBar value={importState.progress} />
            <div className="flex justify-between text-[10px] font-mono text-dim">
              <span>Importing to backend…</span>
              <span>{importState.progress}%</span>
            </div>
          </div>
        )}

        {importState.phase === 'success' && importState.result && (
          <div className="space-y-3">
            <div className="rounded-lg border border-jade/20 bg-jade/5 p-4 space-y-3">
              <div className="flex items-center gap-2">
                <CheckCircle size={13} className="text-jade" />
                <span className="text-xs font-mono text-jade font-semibold">Results imported</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div className="flex justify-between">
                  <span className="text-dim">PLY</span>
                  <span className={importState.result.ply_url ? 'text-jade' : 'text-dim'}>
                    {importState.result.ply_url ? '✓ ready' : 'not found'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dim">SPLAT</span>
                  <span className={importState.result.splat_url ? 'text-jade' : 'text-dim'}>
                    {importState.result.splat_url ? '✓ ready' : 'not found'}
                  </span>
                </div>
              </div>
              <p className="text-[11px] font-mono text-dim">
                Switch to the Viewer tab — the scene will load automatically.
              </p>
            </div>
            <Button size="sm" variant="ghost" onClick={reset}>
              <FileArchive size={11} /> Import different results
            </Button>
          </div>
        )}

        {importState.phase === 'error' && (
          <div className="space-y-3">
            <div className="rounded-lg border border-crimson/20 bg-crimson/5 p-3 flex items-start gap-2">
              <AlertCircle size={13} className="text-crimson shrink-0 mt-0.5" />
              <p className="text-xs font-mono text-crimson">{importState.error}</p>
            </div>
            <Button size="sm" onClick={reset}>Retry</Button>
          </div>
        )}
      </CardBody>
    </Card>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DatasetManager() {
  const [uploadState, setUploadState] = useState<UploadState>({ phase: 'idle', progress: 0 })
  const [projectName, setProjectName] = useState('')
  const { data: projects, isLoading } = useProjects()
  const { setActiveJob } = useAppStore()

  const handleFile = async (file: File) => {
    const name = projectName.trim() || file.name.replace(/\.[^.]+$/, '')
    setUploadState({ phase: 'uploading', progress: 0, filename: file.name, fileSize: file.size })
    try {
      const resp = await api.uploadFile(file, name, (pct) =>
        setUploadState(s => ({ ...s, progress: pct })),
      )
      setUploadState({ phase: 'success', progress: 100, filename: file.name, fileSize: file.size, jobId: resp.job_id })
      setActiveJob(resp.job_id)
    } catch (err: any) {
      setUploadState({ phase: 'error', progress: 0, error: err.message })
    }
  }

  const reset = () => {
    setUploadState({ phase: 'idle', progress: 0 })
    setProjectName('')
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <TopBar
        title="Dataset Manager"
        subtitle="Video → Frame Extraction → COLMAP → Scene Normalization → Colab Package → Import Results"
      />
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto space-y-5">

          {/* ── Step 1: Video upload ──────────────────────────────────────── */}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 rounded bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
                  <span className="text-[9px] font-mono text-amber-400 font-bold">01</span>
                </div>
                <SectionHeading>Upload Video</SectionHeading>
              </div>
            </CardHeader>
            <CardBody className="space-y-4">

              {uploadState.phase === 'idle' && (
                <>
                  <div>
                    <label className="text-[10px] font-mono text-dim uppercase tracking-widest mb-1.5 block">
                      Project Name (optional)
                    </label>
                    <input
                      value={projectName}
                      onChange={e => setProjectName(e.target.value)}
                      placeholder="my-scene"
                      className="w-full bg-surface border border-panel-border rounded-md px-3 py-2 text-sm text-white font-mono placeholder:text-muted focus:border-amber-500/50 focus:outline-none transition-colors"
                    />
                  </div>
                  <DropZone
                    onFile={handleFile}
                    accept={{ 'video/*': ['.mp4', '.mov', '.avi', '.mkv'] }}
                    label="Drag a video here"
                    sublabel="MP4 · MOV · AVI · MKV — 1080p minimum recommended"
                  />
                </>
              )}

              {uploadState.phase === 'uploading' && (
                <div className="space-y-3 py-3">
                  <div className="flex items-center gap-3">
                    <Spinner size={15} />
                    <div>
                      <p className="text-xs font-mono text-ghost">{uploadState.filename}</p>
                      <p className="text-[10px] font-mono text-dim">{fmtBytes(uploadState.fileSize ?? 0)}</p>
                    </div>
                  </div>
                  <ProgressBar value={uploadState.progress} />
                  <div className="flex justify-between text-[10px] font-mono text-dim">
                    <span>Uploading to FastAPI…</span>
                    <span>{uploadState.progress}%</span>
                  </div>
                </div>
              )}

              {uploadState.phase === 'success' && uploadState.filename && (
                <div className="space-y-3">
                  <VideoInfo filename={uploadState.filename} fileSize={uploadState.fileSize ?? 0} />
                  <Button size="sm" variant="ghost" onClick={reset}>
                    <Trash2 size={11} /> Upload different video
                  </Button>
                </div>
              )}

              {uploadState.phase === 'error' && (
                <div className="space-y-3">
                  <div className="rounded-lg border border-crimson/20 bg-crimson/5 p-3 flex items-center gap-2">
                    <AlertCircle size={13} className="text-crimson shrink-0" />
                    <p className="text-xs font-mono text-crimson">{uploadState.error}</p>
                  </div>
                  <Button size="sm" onClick={reset}>Retry</Button>
                </div>
              )}

            </CardBody>
          </Card>

          {/* ── Step 2: Pipeline progress ─────────────────────────────────── */}
          {uploadState.phase === 'success' && uploadState.jobId && (
            <div className="space-y-1">
              <div className="flex items-center gap-2 px-1 mb-2">
                <div className="w-5 h-5 rounded bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
                  <span className="text-[9px] font-mono text-amber-400 font-bold">02</span>
                </div>
                <span className="text-[10px] font-mono text-dim uppercase tracking-widest">Pipeline</span>
              </div>
              <PipelineStepper jobId={uploadState.jobId} />
            </div>
          )}

          {/* ── Step 3: Import Colab results ──────────────────────────────── */}
          <ImportResultsSection defaultJobId={uploadState.jobId} />

          {/* ── Projects list ─────────────────────────────────────────────── */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded bg-cobalt/10 border border-cobalt/20 flex items-center justify-center">
                    <Database size={10} className="text-cobalt" />
                  </div>
                  <SectionHeading>
                    Projects
                    {projects?.meta.total != null && (
                      <span className="ml-2 text-[10px] font-mono text-amber-400 font-normal normal-case tracking-normal">
                        {projects.meta.total}
                      </span>
                    )}
                  </SectionHeading>
                </div>
              </div>
            </CardHeader>
            <CardBody className="p-0">
              {isLoading ? (
                <div className="flex justify-center py-10"><Spinner /></div>
              ) : !projects?.projects.length ? (
                <EmptyState
                  icon={<Database size={18} />}
                  title="No projects yet"
                  desc="Upload a video above to create your first project."
                />
              ) : (
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-panel-border">
                      {['Name', 'ID', 'Created'].map(h => (
                        <th
                          key={h}
                          className="px-5 py-2.5 text-left text-dim font-normal uppercase tracking-widest text-[9px]"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {projects.projects.map(p => (
                      <tr
                        key={p.id}
                        className="border-b border-panel-border/40 hover:bg-surface/40 transition-colors group"
                      >
                        <td className="px-5 py-3 text-white font-medium">{p.name}</td>
                        <td className="px-5 py-3 text-dim truncate max-w-[160px]">{p.id}</td>
                        <td className="px-5 py-3 text-dim">
                          {new Date(p.created_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CardBody>
          </Card>

        </div>
      </div>
    </div>
  )
}
// src/types/api.ts — mirrors backend schemas exactly

export interface OKResponse { ok: boolean; message: string }

export interface PaginationMeta { total: number; skip: number; limit: number }

export interface ProjectCreate { name: string; description?: string }
export interface ProjectRead {
  id: string; name: string; description?: string
  created_at: string; updated_at: string
}
export interface ProjectsListResponse {
  projects: ProjectRead[]
  meta: PaginationMeta
}

export type JobStatus = 'pending' | 'running' | 'success' | 'failed' | 'cancelled'
export interface JobRead {
  id: string; job_type: string; status: JobStatus
  progress: number; message?: string
  result?: Record<string, unknown>; error?: string
  created_at: string; started_at?: string; finished_at?: string
}

// POST /upload response — matches routes.py return dict
export interface UploadResponse {
  job_id: string
  project_id: string        // project name used as logical ID
  dataset_id: string | null
  upload_path: string
  frames_path: string | null
  filename: string
  size_bytes: number
  message: string
}

// POST /upload-results/{job_id} response
export interface ImportResultsResponse {
  import_job_id: string
  status: string
  message: string
  ply_url: string | null
  splat_url: string | null
}

// GET /results/{job_id} response
export interface ResultsResponse {
  job_id: string; ply_url: string | null; splat_url: string | null; result_dir: string
}

// POST /analyze
export interface AnalyzeRequest {
  project_id: string; dataset_id?: string; image_dir: string; blur_threshold?: number
}
export interface AnalyzeResponse {
  job_id: string; project_id: string; dataset_id?: string; image_dir: string; message: string
}

// POST /predict
export interface PredictRequest { project_id: string; analysis_report: Record<string, unknown> }
export interface PredictResponse { job_id: string; project_id: string; message: string }

// POST /train
export interface TrainRequest {
  project_id: string; experiment_id?: string; dataset_id?: string
  sparse_path: string; image_dir: string
  config_overrides?: Record<string, unknown>; resume_checkpoint?: string
}
export interface TrainResponse { job_id: string; run_id: string; project_id: string; message: string }

// POST /resume
export interface ResumeRequest { run_id: string; checkpoint_path?: string; config_overrides?: Record<string, unknown> }
export interface ResumeResponse { job_id: string; run_id: string; checkpoint_path: string; message: string }

// GET /metrics/{run_id}
export interface MetricPoint {
  iteration: number; psnr?: number; ssim?: number; lpips?: number
  loss?: number; n_gaussians?: number; logged_at: string
}
export interface MetricsResponse { run_id: string; metrics: MetricPoint[]; count: number }

// GET /report/{run_id}
// The backend /report/{run_id} endpoint returns the full RunRead shape
// (merged report + run data). All fields from RunRead are included here
// so Reports.tsx can access run_name, status, finished_at, etc. directly.
export interface ReportResponse {
  id: string; project_id: string; report_type: string; title?: string
  summary?: string; payload?: Record<string, unknown>; created_at: string; updated_at: string
  // RunRead fields surfaced by the report endpoint
  run_name: string
  status: string
  finished_at?: string
  final_metrics?: Record<string, unknown>
  config_snapshot?: Record<string, unknown>
  dataset_path?: string
  model_path?: string
}

// Runs
export interface RunRead {
  id: string; project_id?: string; run_name: string; status: string
  sparse_path?: string; model_path?: string; checkpoint_path?: string
  final_metrics?: Record<string, unknown>; best_metrics?: Record<string, unknown>
  total_iterations?: number; duration_seconds?: number
  created_at: string; started_at?: string; finished_at?: string; updated_at: string
}
export interface RunsListResponse { runs: RunRead[]; meta: PaginationMeta }

// POST /export
export interface ExportRequest { run_id: string; formats?: string[]; output_dir?: string }
export interface ExportResponse { job_id: string; run_id: string; formats: string[]; message: string }
/// <reference types="vite/client" />
// src/api/client.ts
import axios from 'axios'
import type {
  AnalyzeRequest, AnalyzeResponse,
  ExportRequest, ExportResponse,
  ImportResultsResponse, ResultsResponse,
  MetricsResponse, PredictRequest, PredictResponse,
  ProjectCreate, ProjectRead, ProjectsListResponse,
  ReportResponse, ResumeRequest, ResumeResponse,
  RunsListResponse, TrainRequest, TrainResponse,
  UploadResponse, JobRead,
} from '@/types/api'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '/api',
  timeout: 60_000,
  headers: { 'Content-Type': 'application/json' },
})

apiClient.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err.response?.data?.detail ??
      err.response?.data?.message ??
      err.message ?? 'Unknown error'
    return Promise.reject(new Error(msg))
  },
)

export const api = {
  // Health
  health: () =>
    apiClient.get<{ status: string }>('/health').then(r => r.data),

  // Projects
  listProjects: () =>
    apiClient.get<ProjectsListResponse>('/projects').then(r => r.data),
  createProject: (body: ProjectCreate) =>
    apiClient.post<ProjectRead>('/projects', body).then(r => r.data),

  // Upload video → start pipeline
  uploadFile: (file: File, projectName: string, onProgress?: (pct: number) => void) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('project_name', projectName)
    return apiClient
      .post<UploadResponse>('/upload', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => { if (e.total) onProgress?.(Math.round((e.loaded / e.total) * 100)) },
      })
      .then(r => r.data)
  },

  // Download Colab training package ZIP (opens browser download)
  colabPackageUrl: (jobId: string): string =>
    `${import.meta.env.VITE_API_URL ?? '/api'}/download/${jobId}/colab-package`,

  // Import Colab results ZIP → POST /upload-results/{job_id}
  importResults: (
    jobId: string,
    file: File,
    onProgress?: (pct: number) => void,
  ) => {
    const fd = new FormData()
    fd.append('file', file)
    return apiClient
      .post<ImportResultsResponse>(`/upload-results/${jobId}`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => { if (e.total) onProgress?.(Math.round((e.loaded / e.total) * 100)) },
      })
      .then(r => r.data)
  },

  // Get viewer file URLs for a completed job
  getResults: (jobId: string) =>
    apiClient.get<ResultsResponse>(`/results/${jobId}`).then(r => r.data),

  // Job status polling
  getStatus: (jobId: string) =>
    apiClient.get<JobRead>(`/status/${jobId}`).then(r => r.data),

  // Dataset analysis
  analyze: (req: AnalyzeRequest) =>
    apiClient.post<AnalyzeResponse>('/analyze', req).then(r => r.data),
  predict: (req: PredictRequest) =>
    apiClient.post<PredictResponse>('/predict', req).then(r => r.data),

  // Training (local GPU — only if available)
  train: (req: TrainRequest) =>
    apiClient.post<TrainResponse>('/train', req).then(r => r.data),
  resume: (req: ResumeRequest) =>
    apiClient.post<ResumeResponse>('/resume', req).then(r => r.data),

  // Runs
  listRuns: (projectId?: string) =>
    apiClient
      .get<RunsListResponse>('/runs', { params: projectId ? { project_id: projectId } : {} })
      .then(r => r.data),
  getMetrics: (runId: string) =>
    apiClient.get<MetricsResponse>(`/metrics/${runId}`).then(r => r.data),
  getReport: (runId: string) =>
    apiClient.get<ReportResponse>(`/report/${runId}`).then(r => r.data),

  // Export
  exportRun: (req: ExportRequest) =>
    apiClient.post<ExportResponse>('/export', req).then(r => r.data),
}
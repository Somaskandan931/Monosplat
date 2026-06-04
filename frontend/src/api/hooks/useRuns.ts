// src/api/hooks/useRuns.ts
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { TrainRequest, ResumeRequest, ExportRequest } from '@/types/api'

export const runKeys = {
  all: ['runs'] as const,
  list: (projectId?: string) => [...runKeys.all, 'list', projectId ?? 'all'] as const,
  metrics: (runId: string) => [...runKeys.all, 'metrics', runId] as const,
  report:  (runId: string) => [...runKeys.all, 'report',  runId] as const,
}

export function useRuns(projectId?: string) {
  return useQuery({
    queryKey: runKeys.list(projectId),
    queryFn: () => api.listRuns(projectId),
    staleTime: 15_000,
  })
}

export function useMetrics(runId: string, enabled = true) {
  return useQuery({
    queryKey: runKeys.metrics(runId),
    queryFn: () => api.getMetrics(runId),
    enabled,
    refetchInterval: (query) => {
      // Poll while run is active
      return query.state.data ? false : 5_000
    },
  })
}

export function useReport(runId: string, enabled = true) {
  return useQuery({
    queryKey: runKeys.report(runId),
    queryFn: () => api.getReport(runId),
    enabled,
    staleTime: 60_000,
  })
}

export function useTrain() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (req: TrainRequest) => api.train(req),
    onSuccess: () => qc.invalidateQueries({ queryKey: runKeys.all }),
  })
}

export function useResume() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (req: ResumeRequest) => api.resume(req),
    onSuccess: () => qc.invalidateQueries({ queryKey: runKeys.all }),
  })
}

export function useExport() {
  return useMutation({
    mutationFn: (req: ExportRequest) => api.exportRun(req),
  })
}

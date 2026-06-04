// src/api/hooks/useJob.ts
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { JobStatus } from '@/types/api'

const TERMINAL: JobStatus[] = ['success', 'failed']

export function useJob(jobId: string | null, enabled = true) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.getStatus(jobId!),
    enabled: !!jobId && enabled,
    refetchInterval: (query) => {
      const status = query.state.data?.status as JobStatus | undefined
      return status && TERMINAL.includes(status) ? false : 2_000
    },
  })
}

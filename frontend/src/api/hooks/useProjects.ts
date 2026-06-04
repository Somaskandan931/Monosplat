// src/api/hooks/useProjects.ts
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { ProjectCreate } from '@/types/api'

export const projectKeys = {
  all: ['projects'] as const,
  list: () => [...projectKeys.all, 'list'] as const,
  detail: (id: string) => [...projectKeys.all, 'detail', id] as const,
}

export function useProjects() {
  return useQuery({
    queryKey: projectKeys.list(),
    queryFn: api.listProjects,
    staleTime: 30_000,
  })
}

export function useCreateProject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: ProjectCreate) => api.createProject(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: projectKeys.list() }),
  })
}

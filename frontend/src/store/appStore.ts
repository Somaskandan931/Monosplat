// src/store/appStore.ts
import { create } from 'zustand'

interface AppStore {
  selectedProjectId: string | null
  selectedRunId: string | null
  activeJobId: string | null

  setSelectedProject: (id: string | null) => void
  setSelectedRun: (id: string | null) => void
  setActiveJob: (id: string | null) => void

  // Sidebar state
  sidebarCollapsed: boolean
  toggleSidebar: () => void
}

export const useAppStore = create<AppStore>((set) => ({
  selectedProjectId: null,
  selectedRunId: null,
  activeJobId: null,
  sidebarCollapsed: false,

  setSelectedProject: (id) => set({ selectedProjectId: id }),
  setSelectedRun:     (id) => set({ selectedRunId: id }),
  setActiveJob:       (id) => set({ activeJobId: id }),

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}))

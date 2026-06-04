// src/components/layout/AppShell.tsx
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'

export function AppShell() {
  return (
    <div className="flex h-screen overflow-hidden bg-void font-body text-ghost antialiased">
      {/* Subtle grid background */}
      <div className="fixed inset-0 bg-grid-faint bg-grid opacity-40 pointer-events-none" />

      <Sidebar />

      <div className="flex-1 flex flex-col overflow-hidden relative">
        <Outlet />
      </div>
    </div>
  )
}

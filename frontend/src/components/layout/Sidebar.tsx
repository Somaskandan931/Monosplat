// src/components/layout/Sidebar.tsx
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Database, Box,
  ChevronLeft, ChevronRight, Atom,
} from 'lucide-react'
import clsx from 'clsx'
import { useAppStore } from '@/store/appStore'

const NAV = [
  { to: '/',         icon: LayoutDashboard, label: 'Dashboard'      },
  { to: '/datasets', icon: Database,        label: 'Dataset Manager' },
  { to: '/viewer',   icon: Box,             label: '3D Viewer'       },
]

export function Sidebar() {
  const { sidebarCollapsed, toggleSidebar } = useAppStore()

  return (
    <aside
      className={clsx(
        'relative flex flex-col bg-panel border-r border-panel-border transition-all duration-300 z-20',
        sidebarCollapsed ? 'w-16' : 'w-56',
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-14 border-b border-panel-border shrink-0">
        <div className="shrink-0 w-7 h-7 rounded bg-amber-500/10 border border-amber-500/30 flex items-center justify-center">
          <Atom size={14} className="text-amber-400" />
        </div>
        {!sidebarCollapsed && (
          <span className="font-display font-700 text-sm tracking-wider text-white whitespace-nowrap">
            MONO<span className="text-amber-400">SPLAT</span>
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 space-y-0.5 px-2 overflow-hidden">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-2.5 py-2 rounded-md text-sm transition-all duration-150 group',
                isActive
                  ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                  : 'text-ghost hover:text-white hover:bg-surface',
              )
            }
          >
            {({ isActive }) => (
              <>
                <Icon
                  size={16}
                  className={clsx(
                    'shrink-0 transition-colors',
                    isActive ? 'text-amber-400' : 'text-dim group-hover:text-white',
                  )}
                />
                {!sidebarCollapsed && (
                  <span className="font-body font-medium whitespace-nowrap">{label}</span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={toggleSidebar}
        className="absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full
                   bg-panel border border-panel-border text-dim hover:text-white
                   flex items-center justify-center transition-colors z-30"
      >
        {sidebarCollapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
      </button>
    </aside>
  )
}
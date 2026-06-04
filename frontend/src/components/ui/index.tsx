// src/components/ui/index.tsx
import clsx from 'clsx'
import type { ReactNode, ButtonHTMLAttributes } from 'react'

// ── Badge ─────────────────────────────────────────────────────────────────────
type BadgeVariant = 'amber' | 'jade' | 'crimson' | 'cobalt' | 'violet' | 'dim'
const BADGE_STYLES: Record<BadgeVariant, string> = {
  amber:   'bg-amber-500/10 text-amber-400 border-amber-500/20',
  jade:    'bg-jade/10 text-jade border-jade/20',
  crimson: 'bg-crimson/10 text-crimson border-crimson/20',
  cobalt:  'bg-cobalt/10 text-cobalt border-cobalt/20',
  violet:  'bg-violet/10 text-violet border-violet/20',
  dim:     'bg-surface text-dim border-panel-border',
}
export function Badge({ children, variant = 'dim' }: { children: ReactNode; variant?: BadgeVariant }) {
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-mono border', BADGE_STYLES[variant])}>
      {children}
    </span>
  )
}

// ── Status badge ──────────────────────────────────────────────────────────────
const STATUS_MAP: Record<string, BadgeVariant> = {
  success: 'jade', failed: 'crimson', running: 'amber', pending: 'dim',
  completed: 'jade', active: 'cobalt',
}
export function StatusBadge({ status }: { status: string }) {
  return <Badge variant={STATUS_MAP[status] ?? 'dim'}>{status.toUpperCase()}</Badge>
}

// ── Button ────────────────────────────────────────────────────────────────────
interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
  loading?: boolean
}
const BTN: Record<string, string> = {
  primary: 'bg-amber-500 hover:bg-amber-400 text-void font-600 shadow-glow-amber',
  ghost:   'bg-surface hover:bg-surface-hover text-ghost hover:text-white border border-panel-border',
  danger:  'bg-crimson/10 hover:bg-crimson/20 text-crimson border border-crimson/20',
}
export function Button({ children, variant = 'ghost', size = 'md', loading, className, ...props }: ButtonProps) {
  return (
    <button
      {...props}
      disabled={loading || props.disabled}
      className={clsx(
        'inline-flex items-center gap-2 rounded-md font-body transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed',
        BTN[variant],
        size === 'sm' ? 'text-xs px-3 py-1.5' : 'text-sm px-4 py-2',
        className,
      )}
    >
      {loading && <Spinner size={12} />}
      {children}
    </button>
  )
}

// ── Card ──────────────────────────────────────────────────────────────────────
export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx('bg-panel border border-panel-border rounded-lg shadow-inner-dark', className)}>
      {children}
    </div>
  )
}

export function CardHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx('px-5 py-4 border-b border-panel-border', className)}>
      {children}
    </div>
  )
}

export function CardBody({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={clsx('px-5 py-4', className)}>{children}</div>
}

// ── Spinner ───────────────────────────────────────────────────────────────────
export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="animate-spin">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.2" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

// ── ProgressBar ───────────────────────────────────────────────────────────────
export function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={clsx('h-1.5 bg-surface rounded-full overflow-hidden', className)}>
      <div
        className="h-full bg-gradient-to-r from-amber-500 to-amber-400 rounded-full transition-all duration-500"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}

// ── StatCard ──────────────────────────────────────────────────────────────────
interface StatCardProps {
  label: string
  value: ReactNode
  delta?: string
  deltaPositive?: boolean
  icon?: ReactNode
  accent?: string
}
export function StatCard({ label, value, delta, deltaPositive, icon, accent = 'text-amber-400' }: StatCardProps) {
  return (
    <Card className="p-5 flex flex-col gap-2 hover:border-amber-500/20 transition-colors">
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-dim uppercase tracking-widest">{label}</span>
        {icon && <span className={clsx('opacity-60', accent)}>{icon}</span>}
      </div>
      <span className={clsx('font-mono font-bold text-2xl text-white', accent)}>{value}</span>
      {delta && (
        <span className={clsx('text-xs font-mono', deltaPositive ? 'text-jade' : 'text-crimson')}>
          {deltaPositive ? '↑' : '↓'} {delta}
        </span>
      )}
    </Card>
  )
}

// ── Empty State ───────────────────────────────────────────────────────────────
export function EmptyState({ icon, title, desc }: { icon: ReactNode; title: string; desc?: string }) {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center">
      <div className="w-12 h-12 rounded-xl bg-surface border border-panel-border flex items-center justify-center text-dim">
        {icon}
      </div>
      <p className="font-display font-600 text-white text-sm">{title}</p>
      {desc && <p className="text-xs text-dim max-w-xs">{desc}</p>}
    </div>
  )
}

// ── Section heading ───────────────────────────────────────────────────────────
export function SectionHeading({ children }: { children: ReactNode }) {
  return (
    <h2 className="font-display font-700 text-xs text-dim uppercase tracking-[0.15em] mb-3">
      {children}
    </h2>
  )
}

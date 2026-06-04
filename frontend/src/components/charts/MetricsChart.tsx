// src/components/charts/MetricsChart.tsx
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import type { MetricPoint } from '@/types/api'

const METRICS = [
  { key: 'psnr',  color: '#f59e0b', label: 'PSNR'  },
  { key: 'ssim',  color: '#10b981', label: 'SSIM'  },
  { key: 'lpips', color: '#ef4444', label: 'LPIPS' },
  { key: 'loss',  color: '#8b5cf6', label: 'Loss'  },
]

interface Props {
  data: MetricPoint[]
  height?: number
  keys?: string[]
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-panel border border-panel-border rounded-lg px-3 py-2 text-xs font-mono shadow-xl">
      <p className="text-dim mb-1">iter {label}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  )
}

export function MetricsChart({ data, height = 260, keys }: Props) {
  const visibleMetrics = keys
    ? METRICS.filter((m) => keys.includes(m.key))
    : METRICS

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: -16, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis
          dataKey="iteration"
          tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'Space Mono' }}
          axisLine={{ stroke: '#1a2235' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'Space Mono' }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 10, fontFamily: 'Space Mono', color: '#6b7280' }}
        />
        {visibleMetrics.map(({ key, color, label }) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            name={label}
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, fill: color }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

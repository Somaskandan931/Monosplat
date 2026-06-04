/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // MonoSplat design tokens — keep in sync with index.css CSS variables
        void:          '#080b10',
        panel:         '#0d1117',
        surface:       '#111827',
        'panel-border':'#1e2a3a',
        ghost:         '#94a3b8',
        dim:           '#475569',
        muted:         '#334155',
        jade:          '#34d399',
        cobalt:        '#60a5fa',
        crimson:       '#f87171',
        amber:         { 400: '#fbbf24', 500: '#f59e0b' },
      },
      fontFamily: {
        sans:  ['DM Sans', 'system-ui', 'sans-serif'],
        mono:  ['Space Mono', 'ui-monospace', 'monospace'],
        display: ['Syne', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
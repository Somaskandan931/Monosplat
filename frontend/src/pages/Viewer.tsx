// src/pages/Viewer.tsx
import { Suspense, useRef, useState, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, Stats } from '@react-three/drei'
import { Box as BoxIcon, RotateCcw, Settings, Maximize2, Link } from 'lucide-react'
import * as THREE from 'three'
import { TopBar } from '@/components/layout/TopBar'
import { Card, CardBody, Button, SectionHeading, Spinner, StatusBadge } from '@/components/ui'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'
import type { ResultsResponse } from '@/types/api'

// ── Demo point cloud (shown when no real data loaded) ─────────────────────────
function DemoCloud({ count = 50000 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null)
  const [geo] = useState(() => {
    const g = new THREE.BufferGeometry()
    const pos = new Float32Array(count * 3)
    const col = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const r = Math.random() * 2
      const θ = Math.random() * Math.PI * 2
      const φ = Math.acos(2 * Math.random() - 1)
      pos[i*3]   = r * Math.sin(φ) * Math.cos(θ) + (Math.random() - 0.5) * 4
      pos[i*3+1] = r * Math.sin(φ) * Math.sin(θ) + (Math.random() - 0.5) * 2
      pos[i*3+2] = r * Math.cos(φ) + (Math.random() - 0.5) * 4
      const t = Math.random()
      col[i*3] = 0.9*t + 0.1*(1-t); col[i*3+1] = 0.6*t + 0.7*(1-t); col[i*3+2] = 0.2*t + 0.9*(1-t)
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('color',    new THREE.BufferAttribute(col, 3))
    return g
  })
  useFrame((_, dt) => { if (ref.current) ref.current.rotation.y += dt * 0.04 })
  return (
    <points ref={ref} geometry={geo}>
      <pointsMaterial size={0.025} vertexColors sizeAttenuation transparent opacity={0.8} depthWrite={false} />
    </points>
  )
}

// ── PLY point cloud loader ────────────────────────────────────────────────────
function PlyCloud({ url }: { url: string }) {
  const ref = useRef<THREE.Points>(null)
  const [geo, setGeo] = useState<THREE.BufferGeometry | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setGeo(null)
    setError(null)
    setLoading(true)
    import('three/examples/jsm/loaders/PLYLoader').then(({ PLYLoader }) => {
      new PLYLoader().load(
        url,
        (g) => {
          g.computeVertexNormals()
          // If no vertex colours, generate a neutral grey palette so the
          // pointsMaterial vertexColors flag doesn't produce a black cloud.
          if (!g.attributes.color) {
            const n = g.attributes.position.count
            const col = new Float32Array(n * 3)
            for (let i = 0; i < n * 3; i++) col[i] = 0.6 + Math.random() * 0.3
            g.setAttribute('color', new THREE.BufferAttribute(col, 3))
          }
          setGeo(g)
          setLoading(false)
        },
        undefined,
        (e) => { setError(String(e)); setLoading(false) },
      )
    })
  }, [url])

  if (error) return null
  if (!geo)  return null
  return (
    <points ref={ref} geometry={geo}>
      <pointsMaterial size={0.015} vertexColors sizeAttenuation transparent opacity={0.9} depthWrite={false} />
    </points>
  )
}

// ── Scene ─────────────────────────────────────────────────────────────────────
function Scene({ plyUrl, showStats }: { plyUrl: string | null; showStats: boolean }) {
  return (
    <>
      {showStats && <Stats />}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} />
      {plyUrl ? <PlyCloud url={plyUrl} /> : <DemoCloud count={60000} />}
      <Grid args={[20, 20]} position={[0, -2.5, 0]} cellColor="#1a2235" sectionColor="#1a2235" fadeDistance={18} fadeStrength={2} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.05} minDistance={1} maxDistance={30} />
    </>
  )
}

// ── Main Viewer ───────────────────────────────────────────────────────────────
export default function Viewer() {
  // activeJobId is set to the PIPELINE job_id by DatasetManager after upload.
  // GET /results/{job_id} looks in data/results/{job_id}/ — which is where
  // the Colab results ZIP is extracted to — so the pipeline job_id is correct.
  const { activeJobId } = useAppStore()
  const [jobIdInput, setJobIdInput] = useState(activeJobId ?? '')
  const [results, setResults]       = useState<ResultsResponse | null>(null)
  const [loading, setLoading]       = useState(false)
  const [plyLoading, setPlyLoading] = useState(false)
  const [error, setError]           = useState<string | null>(null)
  const [showStats, setShowStats]   = useState(false)
  const [resetKey, setResetKey]     = useState(0)

  // Auto-load when activeJobId arrives from the store (set after video upload).
  // We pass the pipeline job_id — NOT an import job_id — which is correct for
  // GET /results/{job_id} since results are stored under the pipeline job folder.
  useEffect(() => {
    if (activeJobId) {
      setJobIdInput(activeJobId)
      loadResults(activeJobId)
    }
  }, [activeJobId])

  const loadResults = async (id: string) => {
    if (!id.trim()) return
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const r = await api.getResults(id.trim())
      if (!r.ply_url && !r.splat_url) {
        setError('Results found but no .ply or .splat files were exported. Check Colab Cell 11.')
      } else {
        setResults(r)
      }
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Build the full URL for the PLY file.
  // results.ply_url is a server-relative path like "/results/{job_id}/exports/final.ply".
  // results.ply_url is "/static/results/{job_id}/exports/final.ply" — root-relative,
  // served by FastAPI's StaticFiles mount (not under /api).
  // Strip the "/api" suffix from VITE_API_URL to get the bare origin.
  const apiBase = import.meta.env.VITE_API_URL ?? '/api'
  const origin  = apiBase.endsWith('/api') ? apiBase.slice(0, -4) : apiBase
  const plyUrl  = results?.ply_url ? `${origin}${results.ply_url}` : null

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <TopBar title="3D Viewer" subtitle="Gaussian Splat visualisation" />
      <div className="flex-1 flex overflow-hidden">

        {/* Sidebar */}
        <div className="w-64 border-r border-panel-border bg-panel flex flex-col gap-4 p-4 shrink-0 overflow-y-auto">
          <SectionHeading>Load Results</SectionHeading>
          <div className="space-y-2">
            <label className="text-xs font-mono text-dim block">Job ID</label>
            <input
              value={jobIdInput}
              onChange={e => setJobIdInput(e.target.value)}
              placeholder="paste job_id here"
              className="w-full bg-surface border border-panel-border rounded-md px-3 py-2 text-xs text-white font-mono placeholder:text-muted focus:border-amber-500/50 focus:outline-none"
            />
            <Button
              variant="primary"
              size="sm"
              className="w-full justify-center"
              onClick={() => loadResults(jobIdInput)}
              loading={loading}
              disabled={!jobIdInput.trim()}
            >
              <Link size={11} /> Load
            </Button>
          </div>

          {error && (
            <p className="text-xs text-crimson font-mono leading-relaxed">{error}</p>
          )}

          {results && (
            <div className="space-y-2 text-xs font-mono border-t border-panel-border pt-3">
              <p className="text-dim uppercase tracking-widest text-[10px]">Files</p>
              <div className="flex justify-between">
                <span className="text-dim">PLY</span>
                <span className={results.ply_url ? 'text-jade' : 'text-dim'}>
                  {results.ply_url ? '✓ ready' : 'not found'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-dim">SPLAT</span>
                <span className={results.splat_url ? 'text-jade' : 'text-dim'}>
                  {results.splat_url ? '✓ ready' : 'not found'}
                </span>
              </div>
            </div>
          )}

          <div className="border-t border-panel-border pt-4 space-y-2">
            <SectionHeading>Controls</SectionHeading>
            <Button size="sm" onClick={() => setShowStats(v => !v)} variant={showStats ? 'primary' : 'ghost'}>
              <Settings size={12} /> Stats
            </Button>
            <Button size="sm" onClick={() => { setResetKey(k => k + 1) }}>
              <RotateCcw size={12} /> Reset
            </Button>
          </div>

          <div className="border-t border-panel-border pt-4 text-xs font-mono text-dim space-y-1">
            <p>🖱 Left drag — orbit</p>
            <p>🖱 Right drag — pan</p>
            <p>🖱 Scroll — zoom</p>
          </div>

          {!results && (
            <div className="mt-auto border-t border-panel-border pt-4">
              <p className="text-[11px] font-mono text-dim leading-relaxed">
                After downloading results from Colab, paste your pipeline job ID above to view your scene.
              </p>
            </div>
          )}
        </div>

        {/* Canvas */}
        <div className="flex-1 relative bg-void">
          <Canvas
            key={resetKey}
            camera={{ position: [0, 2, 8], fov: 50 }}
            gl={{ antialias: true, alpha: false }}
            style={{ background: '#080b10' }}
          >
            <Suspense fallback={null}>
              <Scene plyUrl={plyUrl} showStats={showStats} />
            </Suspense>
          </Canvas>

          {/* Status overlay */}
          <div className="absolute top-4 left-4 flex items-center gap-2 bg-panel/80 backdrop-blur border border-panel-border rounded-md px-3 py-1.5 text-xs font-mono">
            <span className={`w-1.5 h-1.5 rounded-full ${results ? 'bg-jade animate-pulse-slow' : 'bg-dim'}`} />
            <span className="text-ghost">{results ? 'Scene loaded' : 'Demo point cloud'}</span>
          </div>

          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-void/60">
              <div className="flex items-center gap-3 bg-panel border border-panel-border rounded-lg px-5 py-3 text-xs font-mono text-ghost">
                <Spinner size={14} /> Fetching results…
              </div>
            </div>
          )}

          <div className="absolute bottom-4 right-4 text-xs font-mono text-dim/60 flex items-center gap-1.5">
            <Maximize2 size={10} /> F11 for fullscreen
          </div>
        </div>

      </div>
    </div>
  )
}
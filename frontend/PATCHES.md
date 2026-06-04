# Phase 11 — React Frontend Architecture Patches
## MonoSplat Desktop — Implementation Guide

---

## Overview

This patch adds a full React/TypeScript/Vite/TailwindCSS/Three.js frontend
that communicates with the existing FastAPI backend (`backend/app/main.py`).

```
monosplat_desktop/
└── frontend/               ← NEW — all files in this patch
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── tsconfig.node.json
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── index.css
        ├── types/
        │   └── api.ts            # Mirrors backend/app/schemas/api.py
        ├── api/
        │   ├── client.ts         # Axios instance + typed helpers
        │   └── hooks/
        │       ├── useProjects.ts
        │       ├── useRuns.ts    # Train, Resume, Export, Metrics, Report
        │       └── useJob.ts     # Auto-polling job status
        ├── store/
        │   └── appStore.ts       # Zustand global state
        ├── components/
        │   ├── layout/
        │   │   ├── AppShell.tsx  # Root layout with sidebar
        │   │   ├── Sidebar.tsx   # Collapsible nav
        │   │   └── TopBar.tsx    # Page header + API status
        │   ├── ui/
        │   │   └── index.tsx     # Badge, Button, Card, Spinner, etc.
        │   └── charts/
        │       └── MetricsChart.tsx  # PSNR/SSIM/LPIPS/Loss via Recharts
        └── pages/
            ├── Dashboard.tsx         # Overview stats + recent runs
            ├── DatasetManager.tsx    # Dropzone upload + analysis
            ├── TrainingDashboard.tsx # Launch + monitor training
            ├── Experiments.tsx       # Compare runs, export
            ├── Reports.tsx           # Full report + training curves
            └── Viewer.tsx            # React Three Fiber 3D viewer
```

---

## PATCH 1 — Install & Bootstrap

```bash
cd monosplat_desktop/frontend
npm install
npm run dev          # → http://localhost:5173
```

The Vite dev server proxies `/api/*` → `http://localhost:8000/*`, so the
FastAPI backend must be running:

```bash
# From monosplat_desktop/
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## PATCH 2 — Environment Variables

Create `frontend/.env.local` for custom API URL:

```env
VITE_API_URL=http://localhost:8000
```

For production builds using a different origin:

```env
VITE_API_URL=https://api.your-domain.com
```

---

## PATCH 3 — Backend CORS (already configured)

`backend/app/main.py` already has `allow_origins=["*"]`. Tighten for prod:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://app.your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## PATCH 4 — Real .splat Loader (Viewer page)

The Viewer currently renders a demo point cloud. To load real `.splat` files,
install the community loader:

```bash
npm install @antimatter15/splat
```

Then replace the `GaussianCloud` component in `src/pages/Viewer.tsx`:

```tsx
// Replace GaussianCloud with:
import { Splat } from '@react-three/drei'   // or use @antimatter15/splat

function SplatMesh({ url }: { url: string }) {
  return <Splat src={url} />
}
```

For binary `.ply` Gaussian files, use the THREE.js PLYLoader:

```tsx
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'
import { useLoader } from '@react-three/fiber'

function PLYMesh({ url }: { url: string }) {
  const geo = useLoader(PLYLoader, url)
  return (
    <points geometry={geo}>
      <pointsMaterial size={0.01} vertexColors />
    </points>
  )
}
```

---

## PATCH 5 — Build & Serve with FastAPI (Static Files)

Build the frontend and serve it from FastAPI:

```bash
cd frontend && npm run build
```

Add to `backend/app/main.py`:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib

FRONTEND_DIST = pathlib.Path(__file__).parents[2] / "frontend" / "dist"

# After app.include_router(router, ...)
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        return FileResponse(FRONTEND_DIST / "index.html")
```

Now `uvicorn backend.app.main:app` serves both API and frontend on port 8000.

---

## PATCH 6 — app.py Integration

If using `app.py` as the Electron/desktop launcher, add frontend startup:

```python
import subprocess, threading, pathlib

def start_frontend_dev():
    fe = pathlib.Path(__file__).parent / "frontend"
    subprocess.Popen(["npm", "run", "dev"], cwd=fe)

threading.Thread(target=start_frontend_dev, daemon=True).start()
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| React Query (TanStack) | Auto-caching, polling, deduplication for job status |
| Zustand | Lightweight cross-page state (selected run/project/job) |
| Recharts | Server-side metric plotting, no canvas complexity |
| React Three Fiber | Declarative Three.js for 3DGS point cloud viewer |
| Vite proxy | Zero CORS issues during development |
| TailwindCSS | Industrial dark theme with custom design tokens |
| Space Mono + Syne | Distinctive typographic identity for scientific UI |

---

## Page → API Endpoint Mapping

| Page | Endpoints used |
|---|---|
| Dashboard | `GET /projects`, `GET /runs`, `GET /health` |
| Dataset Manager | `POST /upload`, `POST /analyze`, `GET /status/{job_id}` |
| Training Dashboard | `POST /train`, `POST /resume`, `GET /status/{job_id}`, `GET /metrics/{run_id}` |
| Experiments | `GET /runs`, `GET /metrics/{run_id}`, `POST /export` |
| Reports | `GET /runs`, `GET /report/{run_id}`, `GET /metrics/{run_id}`, `POST /export` |
| Viewer | `GET /runs` (to list models), future: serve static `.splat` files |

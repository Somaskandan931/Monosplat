# MonoSplat — Architecture & Dependency Map

## Project Layout

```
monosplat/
├── backend/                   # FastAPI server (local desktop UI)
│   ├── app/
│   │   ├── main.py            # FastAPI app factory, mounts routes + static files
│   │   ├── api/routes.py      # All REST endpoints (upload, status, download, results)
│   │   ├── database/session.py# SQLAlchemy engine + Base + get_db()
│   │   ├── models/orm.py      # ORM: Job, Project, TrainingRun, RunMetric
│   │   ├── services/
│   │   │   ├── pipeline_service.py    # Wraps scripts/pipeline.py for background tasks
│   │   │   ├── experiment_service.py  # CRUD for projects/runs (list_projects, etc.)
│   │   │   └── result_service.py      # Unpacks Colab results ZIP → data/results/
│   │   └── workers/job_runner.py      # ProcessPoolExecutor job lifecycle
│   └── requirements-backend.txt
│
├── colab/
│   ├── train.py               # ★ PRIMARY TRAINING ENTRY POINT (Colab + desktop)
│   └── export_splat.py        # Export checkpoint → .ply + .splat
│
├── configs/
│   └── config.yaml            # ★ SINGLE CONFIG SOURCE OF TRUTH
│
├── docs/
│   ├── README.md
│   └── foggy_preview_fix.md   # Root-cause analysis of foggy preview bug
│
├── frontend/                  # React/TypeScript desktop UI
│   └── src/
│       ├── App.tsx            # Router + AppShell
│       ├── main.tsx           # Vite entry point
│       ├── api/
│       │   ├── client.ts      # Axios base client (VITE_API_URL)
│       │   └── hooks/
│       │       ├── useJob.ts      # usePollJob() — polls /status/{id}
│       │       ├── useProjects.ts # useProjects() — GET /projects
│       │       └── useRuns.ts     # useRuns() — GET runs for a project
│       ├── components/
│       │   ├── charts/MetricsChart.tsx  # Loss/PSNR curves (recharts)
│       │   ├── layout/AppShell.tsx      # Main layout wrapper
│       │   ├── layout/Sidebar.tsx       # Nav sidebar
│       │   ├── layout/TopBar.tsx        # Top header bar
│       │   └── ui/index.tsx             # Shared UI primitives
│       ├── pages/
│       │   ├── Dashboard.tsx        # Job status overview
│       │   ├── DatasetManager.tsx   # Upload video, monitor pipeline
│       │   ├── Experiments.tsx      # Project/run list
│       │   ├── Reports.tsx          # Quality reports
│       │   ├── TrainingDashboard.tsx# Live training metrics
│       │   └── Viewer.tsx           # 3DGS splat viewer
│       ├── store/appStore.ts        # Zustand global state
│       └── types/api.ts             # TypeScript API types
│
├── notebooks/
│   └── monosplat_colab_gpu.ipynb    # ★ Colab GPU notebook (cells 1-12)
│
├── scripts/
│   ├── pipeline.py            # Orchestrates: extract → COLMAP → ZIP
│   └── prepare_dataset.py     # CLI entrypoint for full preprocessing
│
├── src/                       # Core Python library (Colab + backend both import this)
│   ├── dataset/
│   │   └── loader.py          # ColmapDataset: loads cameras + images → Camera objects
│   ├── preprocessing/
│   │   ├── colmap_runner.py   # Runs COLMAP subprocess (feature_extractor → mapper)
│   │   ├── extract_frames.py  # FFmpeg frame extraction + smart selection
│   │   ├── normalize_scene.py # ★ FIXED: camera-radius normalization + P99 filter
│   │   └── utils.py           # read_cameras/images/points3d from COLMAP text format
│   ├── reconstruction/
│   │   ├── gaussian_model.py  # ★ FIXED: GaussianModel with correct scale ceiling
│   │   ├── loss.py            # L1 + SSIM + LPIPS combined loss
│   │   └── trainer.py         # ★ PATCHED: densify diagnostics, 250-iter preview
│   ├── renderer/
│   │   ├── camera.py          # Camera.from_colmap() → intrinsics + extrinsics
│   │   └── renderer.py        # gsplat rasterization wrapper
│   └── utils/
│       ├── colmap_utils.py    # load_colmap_model() + get_sparse_point_cloud()
│       ├── config_loader.py   # load_config() → _ConfigProxy (dict + attr access)
│       ├── env_detect.py      # has_cuda_colmap(), should_use_gpu()
│       ├── image_utils.py     # load_image_rgb(), image_to_tensor(), compute_psnr()
│       ├── io_utils.py        # save_ply(), save_splat(), save_checkpoint(), etc.
│       ├── math_utils.py      # look_at(), perspective_matrix(), build_covariance_3d()
│       └── metrics.py         # PipelineMetrics + TrainingMetricsLog
│
└── tests/                     # (placeholder — add pytest tests here)
```

---

## Dependency Map — Who Calls What

### Training path (primary, Colab-first)

```
notebooks/monosplat_colab_gpu.ipynb
  └─► colab/train.py
        ├─► src/utils/config_loader.py      load_config()
        ├─► src/utils/colmap_utils.py       load_colmap_model(), get_sparse_point_cloud()
        ├─► src/utils/io_utils.py           save_ply(), save_splat()
        ├─► src/dataset/loader.py           ColmapDataset
        ├─► src/preprocessing/normalize_scene.py   normalize_scene(), scene_stats()  ★FIXED
        ├─► src/reconstruction/gaussian_model.py   GaussianModel  ★FIXED
        └─► src/reconstruction/trainer.py          Trainer  ★PATCHED
              ├─► src/reconstruction/loss.py       combined_loss()
              ├─► src/renderer/renderer.py          render()
              └─► src/renderer/camera.py            Camera.from_colmap()
```

### Export path

```
colab/export_splat.py
  ├─► src/reconstruction/gaussian_model.py   GaussianModel (load from checkpoint)
  └─► src/utils/io_utils.py                  save_ply(), save_splat()
```

### Preprocessing path (local desktop)

```
scripts/prepare_dataset.py  (CLI)
scripts/pipeline.py         (called by backend routes.py)
  ├─► src/preprocessing/extract_frames.py   extract_from_video(), run_smart_frame_selection()
  ├─► src/preprocessing/colmap_runner.py    run_colmap()
  └─► src/utils/config_loader.py            load_config()
      src/utils/metrics.py                  PipelineMetrics
```

### Backend path (desktop UI)

```
backend/app/main.py
  └─► backend/app/api/routes.py
        ├─► backend/app/workers/job_runner.py    create_job(), submit_background_job()
        │     └─► backend/app/models/orm.py       Job
        ├─► backend/app/services/pipeline_service.py  → scripts/pipeline.py
        ├─► backend/app/services/experiment_service.py → orm.Project, TrainingRun
        └─► backend/app/services/result_service.py    → unpacks Colab ZIP
```

### Frontend path

```
frontend/src/App.tsx
  ├─► frontend/src/components/layout/AppShell.tsx
  │     ├─► Sidebar.tsx
  │     └─► TopBar.tsx
  ├─► frontend/src/pages/Dashboard.tsx         → api/hooks/useJob.ts → api/client.ts
  ├─► frontend/src/pages/DatasetManager.tsx    → POST /upload
  ├─► frontend/src/pages/TrainingDashboard.tsx → GET /status/{id}
  ├─► frontend/src/pages/Viewer.tsx            → GET /results/{id}
  ├─► frontend/src/pages/Experiments.tsx       → api/hooks/useProjects.ts
  └─► frontend/src/pages/Reports.tsx           → api/hooks/useRuns.ts
```

---

## Files Removed vs Original

| Removed File | Reason |
|---|---|
| `backend/app/services/dataset_analysis_service.py` | No route ever called it; `result_service.py` handles result imports |
| `configs/t4.yaml`, `configs/l4.yaml`, `configs/a100.yaml` | Replaced by `MONOSPLAT_EXTRA_TRAIN_ARGS` env var in Colab notebook |
| `src/preprocessing/normalize_scene_old.py` | Old implementation superseded by fixed version |

---

## Config → Code Flow

`configs/config.yaml` is the single source of truth. Values flow:

```
config.yaml
  → load_config()             (src/utils/config_loader.py)
    → cfg["training"][...]    (colab/train.py)
      → Trainer.__init__()   (src/reconstruction/trainer.py)
        iterations, densify_grad_threshold, max_gaussians, ...
      → GaussianModel.initialise_from_pcd(spatial_lr_scale=cameras_extent)
        _max_log_scale = log(cameras_extent * 0.1)
```

GPU-tier overrides (T4/L4/A100) are applied at runtime via:
```
MONOSPLAT_EXTRA_TRAIN_ARGS=--training.iterations 18000 --training.max_gaussians 120000
```
Set in Colab notebook Cell 2; read by `_apply_env_overrides()` in `colab/train.py`.

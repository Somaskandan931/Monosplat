# MonoSplat

> Single-camera 3D Gaussian Splat reconstruction pipeline — record a video,
> run COLMAP for camera alignment, train with PyTorch + gsplat on GPU,
> view the photorealistic 3D scene in your browser. Zero desktop apps required.
> **Now with AI-powered scene understanding, XR support, and cloud storage integration.**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)
![gsplat](https://img.shields.io/badge/gsplat-1.0+-green)
![FFmpeg](https://img.shields.io/badge/FFmpeg-required-red)
![COLMAP](https://img.shields.io/badge/COLMAP-3.8+-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-v3.0.0-success)

---

## Table of Contents

- [What Is MonoSplat?](#what-is-monosplat)
- [How Gaussian Splatting Works](#how-gaussian-splatting-works)
- [Project Architecture](#project-architecture)
- [Codebase Structure](#codebase-structure)
- [Design Deep-Dives](#design-deep-dives)
  - [GaussianModel (`src/reconstruction/gaussian_model.py`)](#gaussianmodel)
  - [GaussianTrainer (`src/reconstruction/trainer.py`)](#gaussiantrainer)
  - [Loss Functions (`src/reconstruction/loss.py`)](#loss-functions)
  - [Renderer (`src/renderer/renderer.py`)](#renderer)
  - [Camera (`src/renderer/camera.py`)](#camera)
  - [Frame Extraction (`src/preprocessing/extract_frames.py`)](#frame-extraction)
  - [COLMAP Runner (`src/preprocessing/colmap_runner.py`)](#colmap-runner)
  - [Pipeline Manager (`src/pipeline/pipeline_manager.py`)](#pipeline-manager)
  - [Server (`src/pipeline/server.py`)](#server)
  - [AI Layer (`src/ai/ai_layer.py`)](#ai-layer)
- [Pipeline Stages in Detail](#pipeline-stages-in-detail)
- [Quick Start](#quick-start)
- [GPU Training](#gpu-training)
- [Configuration Reference](#configuration-reference)
- [God Mode Features (v3.0)](#god-mode-features-v30)
- [Input Data Requirements](#input-data-requirements)
- [Capture Guide](#capture-guide)
- [Output Formats](#output-formats)
- [Troubleshooting](#troubleshooting)
- [Performance Targets](#performance-targets)
- [Tech Stack](#tech-stack)
- [Method Comparison](#method-comparison)
  - [Gaussian Splatting vs NeRF](#gaussian-splatting-vs-nerf)
  - [Why MonoSplat vs Other Pipelines](#why-monosplat-vs-other-pipelines)
  - [Software Renderer vs gsplat](#software-renderer-vs-gsplat)
- [Future Scope](#future-scope)
- [References](#references)

---

## What Is MonoSplat?

MonoSplat is a **complete end-to-end pipeline** that transforms a single video — shot on any phone — into a photorealistic, navigable 3D scene viewable in any browser. No desktop rendering app, no OpenGL environment, no calibration rig.

```
Video Input  →  Frame Extraction  →  Camera Poses  →  Gaussian Training  →  Browser Viewer
   MP4/MOV         FFmpeg              COLMAP (SfM)   PyTorch + gsplat       Three.js
                    ↓                   ↓                  ↓                    ↓
              Quality Warnings    Cloud Upload        AI Analysis         XR Features
```

### Pipeline Stack

| Stage | Tool | Why |
|-------|------|-----|
| Frame extraction | **FFmpeg** | Broad codec support (H.264/H.265/HEVC/ProRes), hardware-accelerated, faster than OpenCV |
| Camera alignment (SfM) | **COLMAP** | Geometry-based real camera poses — not neural approximations |
| Gaussian Splat training | **Custom PyTorch + gsplat** | GPU-accelerated with nerfstudio's actively-maintained CUDA kernels |
| Browser viewer | **Three.js** | Zero-install, cross-platform, real-time |
| Quality validation | **OpenCV + Custom** | Blur, motion, and exposure detection before COLMAP |
| Cloud storage | **S3 / GCS / Local** | Persistent scene storage and sharing |
| AI Layer | **YOLO + SAM + Transformers** | Object detection, segmentation, scene QA |
| XR Features | **WebXR + Three.js** | VR/AR mode, measurements, collaborative viewing |

This pipeline is:
- **Geometry-based** — real camera poses from real footage, not neural approximations
- **Single-camera** — works with standard phone video, no hardware depth sensors
- **Browser-native** — no desktop app installation required
- **Fully open source** — auditable, extensible, and reproducible

---

## How Gaussian Splatting Works

**Gaussian Splatting is fundamentally different from a normal trained ML model.** A conventional model is trained once and generalises to new inputs at inference time. Gaussian Splatting works in the opposite direction — the "model" is not a generalised neural network. It is a scene-specific set of 3D primitives that intentionally overfit to one particular captured environment.

```
Video A  →  COLMAP  →  PyTorch train  →  scene_A.splat   (only encodes scene A)
Video B  →  COLMAP  →  PyTorch train  →  scene_B.splat   (only encodes scene B)
```

There is no shared weights file. Every new video requires a new full training run. This is a property of the technology, not a limitation of this codebase.

### What Is a Gaussian?

Each 3D Gaussian is a small, oriented, semi-transparent ellipsoid in world space. The scene is the union of hundreds of thousands to millions of these ellipsoids. Every Gaussian has six learnable properties:

| Property | Shape | Description |
|----------|-------|-------------|
| Position | (N, 3) | World-space XYZ coordinates |
| SH coefficients (DC) | (N, 1, 3) | Base colour (zero-frequency spherical harmonic) |
| SH coefficients (rest) | (N, K−1, 3) | View-dependent colour variation (up to degree 3, 16 SH bands total) |
| Opacity | (N, 1) | How transparent the Gaussian is (stored as logit, activated with sigmoid) |
| Scale | (N, 3) | Size along each principal axis (stored as log, activated with exp) |
| Rotation | (N, 4) | Orientation as a unit quaternion |

The renderer sorts these by depth, projects each 3D covariance matrix to a 2D screen ellipse using the Jacobian of the perspective projection, then alpha-composites them back-to-front. Because every operation is differentiable, gradient-based optimisation can nudge position, scale, rotation, opacity, and colour to minimise the difference between the rendered image and the corresponding ground-truth captured frame.

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER (Browser)                          │
│   Upload video  →  "Processing…"  →  View 3D scene          │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP / SSE
┌───────────────────────────▼─────────────────────────────────┐
│         FastAPI Backend  (src/pipeline/server.py)           │
│  - Receives video upload                                    │
│  - Creates job in ModelRegistry (models/registry.json)      │
│  - Returns job_id, streams live status via SSE              │
│  - Serves .splat files to the browser viewer                │
│  - Manages AI results and cloud URLs per job                │
└───────────────────────────┬─────────────────────────────────┘
                            │ Job handoff
┌───────────────────────────▼─────────────────────────────────┐
│       DynamicPipelineManager (src/pipeline/pipeline_manager)│
│  - Watches uploads/ directory                               │
│  - Dispatches jobs to thread/RQ queue                       │
│  - Invokes extract_frames → colmap_runner → trainer         │
└───────────────────────────┬─────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
    FFmpeg (CPU)      COLMAP (SfM)    PyTorch GPU Worker
  ~30 sec extract    ~5-15 min       20-40 min training
  work/<id>/frames   work/<id>/      work/<id>/models/
                     colmap/         gaussian/
                     sparse_text/
                            │
┌───────────────────────────▼─────────────────────────────────┐
│         Storage  (models/registry.json + work/ directory)   │
│  - .splat and .ply files per job                            │
│  - Cloud URLs (S3/GCS) if enabled                           │
│  - AI analysis results per job                              │
└─────────────────────────────────────────────────────────────┘
```

### What Runs Where

| Stage | Tool | Runs On | Approx. Time |
|-------|------|---------|--------------|
| Frame extraction | FFmpeg | Local CPU | ~30 sec |
| Quality validation | OpenCV | Local CPU | Parallel with FFmpeg |
| SfM pose estimation | COLMAP | Local (GPU-accelerated SIFT) | ~5–15 min |
| Gaussian Splat training | PyTorch + gsplat | Colab T4 / local CUDA GPU | 20–40 min |
| FastAPI server + viewer | FastAPI + Three.js | Local | n/a (always on) |
| AI Layer | YOLO + SAM + Transformers | Local GPU (optional) | ~1–2 min post-training |
| Cloud upload | boto3 / GCS | Background thread | ~1–5 min (size-dependent) |

---

## Codebase Structure

```
monosplat/
├── config/
│   └── config.yaml                   # All hyperparameters and feature flags
├── data/
│   ├── raw/                          # User input: raw video files
│   ├── processed/                    # Intermediate processed data
│   └── colmap_output/                # COLMAP sparse reconstruction output
├── docs/
│   └── pipeline.md                   # Internal pipeline design notes
├── models/
│   ├── checkpoints/                  # Mid-training .pkl snapshots
│   ├── gaussian/                     # Final .ply and .splat outputs
│   └── registry.json                 # Job status database (loaded at server startup)
├── notebooks/
│   └── monosplat_colab_gpu.ipynb     # Google Colab training notebook
├── outputs/
│   ├── logs/                         # Per-job training logs (one file per job_id)
│   ├── renders/                      # Training preview PNG renders
│   └── videos/                       # Output video exports
├── scripts/
│   ├── train_local_gpu.py            # Local GPU training entry point
│   ├── zip_for_colab.py              # Package job for Colab handoff
│   ├── mark_ready.py                 # Update registry status after Colab training
│   ├── smoke_test.py                 # Quick end-to-end sanity check
│   ├── verify_pipeline.py            # Dependency and environment check
│   └── start_server.ps1              # Windows server startup script
├── src/
│   ├── ai/
│   │   └── ai_layer.py               # YOLO + SAM + Transformers AI integration
│   ├── pipeline/
│   │   ├── server.py                 # FastAPI endpoints + Three.js viewer HTML
│   │   ├── pipeline_manager.py       # Job lifecycle: queue, dispatch, status
│   │   ├── queue_setup.py            # Redis/RQ queue with thread fallback
│   │   └── worker.py                 # Background job worker (runs pipeline stages)
│   ├── preprocessing/
│   │   ├── extract_frames.py         # FFmpeg frame extraction + quality filters
│   │   ├── colmap_runner.py          # COLMAP automation (SfM pipeline)
│   │   └── utils.py                  # Shared preprocessing utilities
│   ├── reconstruction/
│   │   ├── gaussian_model.py         # GaussianModel: learnable parameters + densification
│   │   ├── trainer.py                # GaussianTrainer: training loop, gsplat/software paths
│   │   └── loss.py                   # L1, SSIM, PSNR, LPIPS loss functions
│   └── renderer/
│       ├── renderer.py               # GaussianRenderer: gsplat-first + software fallback
│       └── camera.py                 # Camera: pinhole model with 360GS-compatible API
├── uploads/                          # Incoming video files (managed by server)
├── work/                             # Per-job working directory (created at runtime)
│   └── <job_id>/
│       ├── frames/                   # Extracted PNG frames
│       ├── colmap/                   # COLMAP database and sparse model
│       └── models/gaussian/          # Final .ply and .splat output
└── requirements.txt
```

---

## Design Deep-Dives

This section explains the key design decisions in each major source file.

---

### GaussianModel

**File:** `src/reconstruction/gaussian_model.py`

`GaussianModel` is a `torch.nn.Module` that holds all of the learnable Gaussian parameters and exposes the density control operations (clone, split, prune).

#### Parameter Representation

All six properties are stored in their *unconstrained* (pre-activation) form as `nn.Parameter` tensors so that gradient-based optimisation operates in unconstrained space:

| Stored parameter | Activation applied at access | Why |
|-----------------|------------------------------|-----|
| `_opacities` (logit) | `torch.sigmoid(_opacities)` | Constrains opacity to (0, 1) |
| `_scales` (log) | `torch.exp(_scales)` | Constrains scale to (0, ∞) |
| `_rotations` (raw quaternion) | `F.normalize(_rotations, dim=1)` | Normalises to unit quaternion |
| `_positions` | identity | Unconstrained world position |
| `_features_dc`, `_features_rest` | identity | SH coefficients, unbounded |

#### Initialisation from COLMAP Points

`create_from_points(positions, colors)` converts the COLMAP sparse point cloud into the initial Gaussian population:

1. **DC SH coefficients** — derived from the COLMAP point colour using the inverse of the SH zero-frequency evaluation: `sh_dc = (color − 0.5) / SH_C0`.
2. **Rest SH coefficients** — initialised to zero. The training loop gradually activates higher-degree SH via `oneup_sh_degree()`.
3. **Initial scale** — computed with `_knn_mean_dist`, the mean distance to the 3 nearest neighbours in the point cloud. This gives each Gaussian a size commensurate with the local point density. The result is stored as `log(mean_dist)` (log-scale, no shrink factor — matching the 360GS convention for object-mode reconstruction).
4. **Initial opacity** — `inverse_sigmoid(0.1)`. Gaussians start at 10% opacity; the optimiser pushes them to be more opaque where they explain real geometry.
5. **Initial rotation** — identity quaternion `[1, 0, 0, 0]`.

#### Memory-Efficient KNN (`_knn_mean_dist`)

Computing mean nearest-neighbour distances for a point cloud of N points naively requires an N×N distance matrix. At 80,000 Gaussians, this is 25 GB — infeasible on any consumer GPU. `_knn_mean_dist` handles this with a two-tier approach:

- **Small clouds (N ≤ 8192):** exact pairwise KNN on GPU.
- **Large clouds:** randomly sample 8,192 representative points, compute exact KNN for the sample, then assign each full-population point the mean-k distance of its nearest sample neighbour. Distance computation for the full population is chunked in groups of 4,096 rows to keep peak VRAM around 640 MB.

#### Density Control

Gaussian Splatting achieves high-quality reconstruction by adaptively adding Gaussians where they are needed and removing ones that contribute nothing. Three operations manage density:

**Clone (`densify_and_clone`):** Targets Gaussians with high position gradients (the scene is not yet well explained) AND small scales (the Gaussian is already small, so it needs a neighbour to cover more area). A copy of the Gaussian is appended in-place. The condition uses `percent_dense * scene_extent` as the size threshold, matching the 360GS implementation.

**Split (`densify_and_split`):** Targets Gaussians with high position gradients AND large scales (the Gaussian covers too much area and needs to be split into finer detail). The parent is replaced by N=2 children. Child positions are sampled from a Gaussian distribution centred on the parent using the parent's scale as the standard deviation, then rotated into world space using the parent's rotation matrix. Child scales are set to `log(parent_scale / (0.8 × N))` — slightly smaller than half the parent, matching the 360GS split formula.

**Prune (`prune_points`):** Removes Gaussians that are either too transparent (opacity < 0.005) or too large relative to the scene (scale > 0.1 × scene_extent), or that push the count beyond `max_gaussians`. All parameter tensors are sliced with `.detach()` to prevent autograd graph violations.

#### 360GS API Compatibility

The model exposes dual property aliases (`positions`/`get_xyz`, `scales`/`get_scaling`, etc.) so that it is a drop-in replacement in both the MonoSplat training loop and the LeoDarcy/360GS training loop. It also implements `get_features()` (full concatenated SH tensor) and `get_covariance()` (3D covariance in upper-triangular packed form) to match the interface expected by the CUDA rasterizer.

---

### GaussianTrainer

**File:** `src/reconstruction/trainer.py`

`GaussianTrainer` orchestrates the entire optimisation process. It selects the gsplat or software renderer based on runtime availability and runs the main loop with densification, opacity resets, checkpointing, and evaluation.

#### Dual Training Path

The trainer automatically selects the best available backend:

- **gsplat path** (`_use_gsplat_train = True`): Used when `gsplat` is installed and CUDA is available. The forward pass calls `gsplat.rasterization()` directly, which returns a `meta` dict containing `means2d` (screen-space projected Gaussian centres with gradients) and `radii` (per-Gaussian screen radii). Densification reads from `means2d.grad`, which is the original 3DGS paper's criterion — more accurate than position-gradient heuristics.

- **Software path** (`_use_gsplat_train = False`): Fallback for CPU-only machines or when gsplat is not installed. The software renderer is called instead, and densification uses `model._positions.grad.norm(dim=1)` as a proxy for 2D mean gradients. Less accurate but functional for debugging.

#### Optimizer Setup

Per-parameter Adam groups match the 360GS and original 3DGS paper learning rates:

| Parameter group | Learning rate |
|----------------|---------------|
| `_positions` | `0.00016` (position LR) |
| `_features_dc` | `0.0025` (feature LR) |
| `_features_rest` | `0.0025 / 20 = 0.000125` (higher-freq SH trained slower) |
| `_opacities` | `0.05` |
| `_scales` | `0.005` |
| `_rotations` | `0.001` |

`eps=1e-15` is used (tighter than PyTorch's default 1e-8) to prevent premature gradient saturation on very small parameters.

#### Position LR Decay

The position learning rate is decayed exponentially over the course of training:

```
lr_position(t) = base_lr × exp(−5 × t / total_iterations)
```

This matches the original 3DGS paper schedule. At the end of training, the position LR is `exp(−5) ≈ 0.007×` its starting value, encouraging the Gaussians to settle into stable final positions rather than continuing to drift.

#### SH Degree Scheduling

The model starts with `active_sh_degree = 0` (only the DC colour term, no view-dependence). Every 1,000 iterations, `oneup_sh_degree()` increments this by 1, up to the configured maximum (default: 3). This progressive activation prevents the higher-frequency SH bands from fitting noise early in training before the geometry is established.

#### NaN Detection

The trainer detects and skips iterations where `loss.backward()` would produce NaN. The first 10 NaN events are printed individually; thereafter one line is printed every 100 events. This allows training to continue past occasional numerical instabilities (typically from degenerate Gaussians before pruning removes them).

#### Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` is applied every iteration. This prevents occasional large gradient spikes from corrupting training when Gaussians temporarily overlap or produce extreme depth values.

#### Opacity Reset

Every `opacity_reset_interval` iterations (default: 3000), opacities are clamped down to `sigmoid(−4.595) ≈ 0.01`. This gives newly-added Gaussians a chance to prove their usefulness — those that do not get enough photometric support will be pruned in the next densification cycle. The Adam state for the opacity parameter group is flushed and reset after each opacity reset to avoid stale momentum terms.

#### Evaluation

Every `eval_every` iterations (default: 1000), held-out test cameras (if available) are rendered without gradient tracking and PSNR and SSIM metrics are computed. Results are appended to `self.eval_log` and printed. This lets you monitor reconstruction quality without waiting for training to complete.

---

### Loss Functions

**File:** `src/reconstruction/loss.py`

The training loss is a weighted combination of L1 and SSIM:

```
loss = (1 − λ_ssim) × L1 + λ_ssim × (1 − SSIM)
```

The default `λ_ssim = 0.2` matches the original 3DGS paper and the 360GS codebase. SSIM is computed with a Gaussian kernel of width 11 pixels and σ=1.5, with constants C1=0.0001 and C2=0.0009.

**SSIM kernel caching:** The Gaussian kernel is an `@functools.lru_cache` keyed on (window_size, channels, device_str). This avoids rebuilding the kernel on every iteration, which would create a new CUDA tensor every step.

**L1 loss** penalises absolute pixel differences, which is robust to outliers and produces sharp edges. **SSIM** penalises differences in local contrast and structure, which helps maintain texture detail and prevents the ghosting/blurring that pure L1 sometimes produces around fine structures.

**PSNR metric** is computed for evaluation (not training):

```
PSNR = 20 × log10(1.0) − 10 × log10(MSE) = −10 × log10(MSE)
```

Typical Gaussian Splatting reconstruction PSNR is 25–35 dB. Results above 30 dB are considered high quality.

**LPIPS** (Learned Perceptual Image Patch Similarity) is optionally available. It uses a pre-trained VGG network to compute perceptual similarity rather than pixel-level metrics. Disabled by default (`lambda_lpips = 0.0`). Set `> 0` in config if you have the `lpips` package installed.

---

### Renderer

**File:** `src/renderer/renderer.py`

`GaussianRenderer` implements a **gsplat-first with software fallback** design. The choice of backend is made at construction time based on the availability of `gsplat` and CUDA.

#### gsplat Backend

When `gsplat` is available, the renderer calls `gsplat.rasterization()`, which:
1. Projects Gaussian means to 2D screen space.
2. Computes per-Gaussian 2D covariance ellipses from the 3D covariance and the projection Jacobian.
3. Performs GPU tile-based rasterization with back-to-front depth sorting.
4. Returns the composited RGB image, an alpha channel, and the `meta` dict (containing `means2d`, `radii`, `gaussian_ids`) used by the trainer for gradient-based densification.

This is the standard Nerfstudio gsplat approach, which is actively maintained and avoids the manual CUDA compilation step required by the original `diff-gaussian-rasterization` extension.

#### Software Backend

When gsplat is unavailable (CPU-only machines, or gsplat not installed), the renderer falls back to a pure-PyTorch implementation:

1. **Transform:** World-space positions are multiplied by the view matrix to get camera-space positions. Points behind the camera are culled.
2. **Project covariance:** The 3D covariance (built from scale + rotation quaternion) is projected to a 2D screen covariance using the Jacobian of the perspective projection, `J @ W_mat`. A regularisation term of 0.3 is added to the diagonal to prevent degenerate near-zero covariances from causing numerical issues.
3. **Batch render:** Gaussians are sorted by depth and rendered in batches of `batch_size=5000` to limit peak memory. Each batch evaluates the 2D Gaussian kernel on a pixel grid and accumulates the alpha-composited result.

#### Spherical Harmonic Evaluation

Both paths share the `_eval_sh(degree, sh, dirs)` function, which evaluates spherical harmonic coefficients up to degree 3 for a set of view directions. The SH basis functions are hard-coded constants `SH_C0`, `SH_C1`, `SH_C2`, `SH_C3` matching the original 3DGS paper. The output is clamped to `[0, 1]` after adding the DC offset of 0.5.

---

### Camera

**File:** `src/renderer/camera.py`

`Camera` is a plain-data pinhole camera model (no PyTorch tensors) with 360GS-compatible properties. It is thread-safe by design — the pipeline spawns multiple rendering threads, and mutable tensors in cameras would cause race conditions.

Key properties:
- `fx, fy, cx, cy` — intrinsic parameters. If not provided, computed from `fov_degrees` using `fy = (height/2) / tan(fov/2)`.
- `view_matrix` — 4×4 float32 view matrix, also exposed as `world_view_transform` (360GS alias).
- `full_proj_transform` — view @ projection, used by the CUDA rasterizer.
- `FoVx, FoVy, tanfovx, tanfovy` — field of view accessors for 360GS compatibility.
- `image_width, image_height` — aliases for `width, height`.

Camera matrices are constructed using `look_at(position, target, up)` and `perspective_matrix(fov, aspect, near, far)` from `src/utils/math_utils`.

---

### Frame Extraction

**File:** `src/preprocessing/extract_frames.py`

The frame extractor is the first stage of the pipeline and is responsible for producing a clean, high-quality set of PNG frames for COLMAP.

#### Adaptive FPS

If `video_fps: null` in config, the extractor automatically chooses an extraction rate based on video duration:
- Short videos (< 30 sec) → higher FPS (more frames for coverage)
- Long videos (> 60 sec) → lower FPS (avoid hitting the 600-frame cap)

A hard cap of 600 frames prevents excessive COLMAP feature extraction time.

#### Multi-Stage Quality Filtering

After extraction, frames pass through a cascade of quality checks:

**Resolution validation** (`validate_image_resolution`): Every frame is checked with PIL before COLMAP runs. Frames smaller than 256×256px are rejected with a hard error. Sub-256px images produce near-zero SIFT features, which is the root cause of the "46 Gaussians / blank splat" failure mode.

**Blur filtering** (`filter_blurry_images`): Each frame is converted to greyscale and the variance of the Laplacian is computed. Frames below a configurable threshold (default: 80.0) are removed. Higher variance = sharper image. Raising the threshold is stricter; lowering it is more permissive.

**Feature-count filtering** (`filter_low_feature_frames`): SIFT keypoints are detected on each frame and frames with fewer than a dynamically-computed threshold (based on the average feature count across all frames) are removed. This catches cases like frames looking at blank walls or heavily overexposed regions.

**Exposure validation** (`validate_exposure`): Frames are checked for overexposure (too many saturated pixels) and underexposure (mean brightness too low). Problematic frames are warned about rather than hard-removed.

**Motion estimation** (`estimate_motion`): Optical flow (via OpenCV) is computed between consecutive frames to detect motion blur from fast camera movement. Warnings are generated if the average optical flow magnitude exceeds a threshold.

#### Frame Validation

After filtering, every remaining frame is re-opened with PIL (`validate_images`) to confirm it is not corrupt before COLMAP runs. Corrupt frames that passed extraction are silently removed.

---

### COLMAP Runner

**File:** `src/preprocessing/colmap_runner.py`

The COLMAP runner automates all four internal COLMAP stages via subprocess calls, with careful parameter tuning to maximise reconstruction quality for the input types common in MonoSplat (phone video, single-object capture, indoor scenes).

#### Why `exhaustive_matcher` Over `sequential_matcher`

The original MonoSplat v1 used `sequential_matcher`, which only matched each frame to its N nearest temporal neighbours. For phone video at 5 fps, this produced ~0.005 seconds of matching time per frame — meaning almost zero cross-view matches on anything but the simplest subjects.

The current runner uses `exhaustive_matcher` as the default, which matches every frame pair. This is the critical change that fixed the "46 Gaussians / blank splat" failure mode. `exhaustive_matcher` is slower but produces complete, accurate correspondences across the full set of frames.

Key COLMAP parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `SiftExtraction.peak_threshold` | `0.004` | Default is 0.0067; lowering it extracts more keypoints on weak-texture surfaces (dark logos, glossy products) |
| `SiftExtraction.max_num_features` | `16000` | Dense enough for complex objects |
| `SiftMatching.guided_matching` | `1` | Epipolar constraint filters false matches, especially on reflective or repeated-pattern surfaces |
| `SiftMatching.max_num_matches` | `32768` | Allows rich matches per frame pair |
| `camera_model` | `OPENCV` | Models radial distortion (`k1, k2, p1, p2`), which is significant in phone video |
| `single_camera` | `true` | Phone video: one lens model fits all frames |

#### Mapper Thresholds

Mapper thresholds (minimum inlier ratio, absolute pose refinement tolerances) are stratified by the `quality` setting in config (`low / medium / high`). Higher quality = stricter thresholds = cleaner sparse model but more chance of fewer registered images on difficult captures.

#### 3D Point Count Diagnostic

After reconstruction, the runner reads `points3D.txt` and counts the 3D points. If fewer than 500 points are found, a loud warning is printed:

```
[COLMAP] ⚠ WARNING: Only N 3D points found. This will produce an almost-empty Gaussian splat.
```

This is an early warning before the (much slower) GPU training runs on bad input data.

#### Output Layout

```
work/<job_id>/colmap/
├── database.db              # COLMAP feature and match database
├── sparse/0/                # Binary sparse model (cameras, images, points3D)
└── sparse_text/             # Text-format export
    ├── cameras.txt          # Camera intrinsics (focal length, distortion coefficients)
    ├── images.txt           # Camera extrinsics (pose per frame)
    └── points3D.txt         # Sparse 3D point cloud (used to initialise GaussianModel)
```

---

### Pipeline Manager

**File:** `src/pipeline/pipeline_manager.py`

`DynamicPipelineManager` is the job lifecycle controller. It maintains the authoritative registry of all jobs in `models/registry.json` and dispatches incoming uploads through the full pipeline.

#### Job States

Jobs transition through the following states (stored as `status` in `registry.json`):

```
uploading → extracting → colmap → training → ai_analysis → uploading_cloud → ready
                                    ↓
                             ready_for_colab    (no local GPU)
```

`error` is a terminal state reachable from any stage.

#### Queue Architecture

The manager uses `queue_setup.py` to initialise a job queue with automatic fallback:

1. **Redis + RQ (preferred):** If Redis is running, jobs are dispatched as RQ tasks. This supports multiple concurrent workers and job persistence across server restarts.
2. **Thread fallback:** If Redis is not available, jobs run in background threads via Python's `concurrent.futures.ThreadPoolExecutor`. This is the default for local development.

The queue mode in use is reported at server startup:

```
[server] Queue mode: thread   # or 'redis' if Redis is available
```

#### Registry Persistence

`registry.json` is the single source of truth for all job metadata. It is updated atomically after each stage transition. **Important:** The FastAPI server loads `registry.json` once at startup. Any manual edits made while the server is running are ignored until a restart. This is by design — atomic file I/O from multiple worker threads requires this approach to avoid corruption.

---

### Server

**File:** `src/pipeline/server.py`

The FastAPI server implements the complete HTTP surface for MonoSplat, including the video upload portal, the Three.js viewer, and all API endpoints.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Upload portal HTML |
| `POST` | `/upload` | Upload video, create job, return `job_id` |
| `GET` | `/api/jobs` | List all jobs with status |
| `GET` | `/api/jobs/{job_id}` | Single job status + metadata |
| `GET` | `/api/jobs/{job_id}/stream` | SSE live status stream (stage + progress %) |
| `GET` | `/api/jobs/{job_id}/metrics` | PSNR / SSIM / timing for completed jobs |
| `PUT` | `/api/jobs/{job_id}/meta` | Update scene notes and tags |
| `GET` | `/api/jobs/{job_id}/ai` | AI analysis results (detections, segmentation) |
| `POST` | `/api/jobs/{job_id}/ai/query` | Spatial search: query by class, description, or position |
| `POST` | `/api/jobs/{job_id}/ai/qa` | Ask a natural language question about the scene |
| `GET` | `/api/models` | List completed (READY) models |
| `GET` | `/api/models/latest` | Most recently completed model |
| `POST` | `/api/train-local/{job_id}` | Trigger local GPU training for a `ready_for_colab` job |
| `GET` | `/api/health` | Queue mode, Redis status, worker count, uptime |
| `GET` | `/splat/{job_id}` | Serve `.splat` binary for the browser renderer |
| `GET` | `/ply/{job_id}` | Serve `.ply` for download |
| `GET` | `/thumbnails/{job_id}` | Serve thumbnail PNG |
| `GET` | `/share/{job_id}` | Shareable redirect → `/viewer/{job_id}` |
| `GET` | `/viewer/{job_id}` | Inline Three.js viewer (mobile-ready, XR-enabled) |
| `GET` | `/capture-guide` | Standalone capture best-practices page |

#### SSE Progress Stream

`GET /api/jobs/{job_id}/stream` returns a `text/event-stream` that pushes live job status updates. Each event contains:
- `stage`: Current pipeline stage name
- `progress`: Integer 0–100
- `message`: Human-readable status message
- `warnings`: List of quality warnings from frame extraction

The frontend uses this to update the job card progress bar without polling.

#### Error Handling

All HTTP errors return `application/json` (never HTML). A global `StarletteHTTPException` handler ensures the frontend can always parse error responses:

```python
@app.exception_handler(StarletteHTTPException)
async def _json_http_error(request, exc):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
```

---

### AI Layer

**File:** `src/ai/ai_layer.py`

The AI Layer adds intelligent scene understanding to completed Gaussian Splat jobs. It processes the extracted frames (not the splat directly) using three model families:

#### Object Detection (YOLO)

`ObjectDetector` loads a YOLO model (default: `yolov8n.pt`, configurable to `yolov8s/m/l`) via the `ultralytics` package. It runs inference on the training frames and aggregates per-frame detections into a scene-level inventory. Each detection includes:
- Class name (e.g., "chair", "bottle", "plant")
- Confidence score
- Bounding box `[x1, y1, x2, y2]`
- Centre point `[x, y]`

The detection count is stored in `registry.json` as `ai_detections` and displayed in the job card.

#### Semantic Segmentation (SAM)

`SceneSegmenter` uses Facebook's Segment Anything Model (`facebook/sam-vit-base` by default) to produce pixel-level segmentation masks. This provides finer-grained region information than bounding boxes.

#### Spatial Search

`SpatialIndex` allows querying the scene by:
- **Class:** `"Find all chairs"` — returns all detections of the given class with positions.
- **Description:** `"Find wooden objects"` — fuzzy text match against detection class names.
- **Nearby:** `"Find objects near position (x, y, z)"` — sorts by 2D proximity to a specified point.

#### Scene QA

`SceneQA` uses a causal language model (default: `gpt2`, configurable to `gpt2-medium/large`) to answer natural language questions about the scene. Questions like "What objects are in this scene?" or "How many chairs are visible?" are answered by prompting the model with a structured description of the detected objects.

All AI components are **optional** and disabled by default (`ai_layer.enabled: false` in config). Each component degrades gracefully if its dependency is not installed.

---

## Pipeline Stages in Detail

### Stage 1 — Frame Extraction (FFmpeg)

```
Video (MP4/MOV/AVI/MKV) → FFmpeg → work/<job_id>/frames/*.png
```

FFmpeg is invoked via subprocess rather than a Python binding. This gives access to hardware-accelerated decoding on supported platforms and avoids a Python-level OpenCV dependency for the extraction step. The output is always PNG to avoid re-compression artefacts.

After extraction, the multi-stage quality filter runs (blur → feature count → exposure → resolution). Frames that fail are removed before COLMAP touches them. Warnings are stored per-job and surfaced in the UI.

### Stage 2 — Camera Pose Estimation (COLMAP SfM)

```
Frames → COLMAP → work/<job_id>/colmap/sparse_text/
                      ├── cameras.txt
                      ├── images.txt
                      └── points3D.txt
```

Four internal COLMAP substages:
1. `feature_extractor` — SIFT keypoint detection on every frame
2. `exhaustive_matcher` — keypoint matching across all frame pairs
3. `mapper` — Structure-from-Motion: triangulates 3D points and estimates camera poses
4. `model_converter` — exports binary sparse model to text format

The text-format export is used by the Gaussian trainer to load camera poses and the initial point cloud.

### Stage 3 — Gaussian Splat Training (PyTorch + gsplat, GPU required)

```
Frames + Camera Poses → GaussianTrainer
                              → work/<job_id>/models/gaussian/<job_id>.ply
                              → work/<job_id>/models/gaussian/<job_id>.splat
```

Training is an iterative process:
1. Pick a random training camera
2. Render current Gaussians from that viewpoint (gsplat or software path)
3. Compute L1 + SSIM loss against the ground-truth frame
4. Backpropagate and update all six learnable properties via Adam
5. Accumulate per-Gaussian screen-space gradients
6. Every 100 iters (between iter 500 and 15000): densify (clone + split) and prune
7. Every 3000 iters: reset opacities
8. Every 5000 iters: save checkpoint + PLY
9. Every 1000 iters: evaluate PSNR/SSIM on held-out cameras

At the end of training, the model is exported as both `.ply` and `.splat`.

### Stage 4 — AI Analysis (Optional)

```
work/<job_id>/frames/ → AI Layer → registry.json (ai_results, ai_detections)
```

Object detection and segmentation run on the training frames. Results are stored in the registry for the API to serve.

### Stage 5 — Cloud Upload (Optional)

```
work/<job_id>/models/gaussian/ → S3 / GCS → cloud_urls stored in registry.json
```

If cloud storage is enabled, the `.splat` and `.ply` files are uploaded to the configured bucket. Cloud URLs are returned by `GET /api/jobs/{job_id}`.

### Stage 6 — Browser Viewer (Three.js)

```
<job_id>.splat → FastAPI → /viewer/<job_id> → Three.js real-time render
```

The viewer is a self-contained HTML page served by FastAPI. It loads the `.splat` binary via the `/splat/{job_id}` endpoint and renders it using the `gaussian-splats-3d` Three.js library.

| Control | Action |
|---------|--------|
| Left drag | Orbit/rotate |
| Right drag | Pan |
| Scroll | Zoom |
| R | Reset camera |
| F | Fullscreen |
| V | Enter VR mode |
| T | Toggle measurement tool |
| A | Toggle annotations |

---

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA

The default `pip install torch` installs a CPU-only build. You must explicitly install the CUDA build:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True + your GPU name
```

`cu121` wheels are forward-compatible with CUDA drivers 12.1–12.5+. Install `gsplat` after PyTorch CUDA is confirmed working:

```bash
pip install gsplat
```

### 3. Install External Tools

**FFmpeg** (frame extraction):
```bash
# Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

**COLMAP** (Structure-from-Motion):
```bash
# Ubuntu
sudo apt install colmap

# macOS
brew install colmap

# Windows
# Download from https://github.com/colmap/colmap/releases and add to PATH
```

Verified working versions: **COLMAP 3.13.0** (with CUDA), **FFmpeg 8.1**

Verify:
```bash
colmap --version   # 3.8+
ffmpeg -version    # 4.x+
```

### 4. Configure the Project

Edit `config/config.yaml` to set your preferences. Defaults work for most setups. To enable optional features:

```yaml
# Enable AI Layer
ai_layer:
  enabled: true
  detection_model: "yolov8n.pt"

# Enable cloud storage
cloud_storage:
  enabled: true
  type: "s3"
  s3:
    bucket: "monosplat-jobs"
    region: "us-east-1"
```

### 5. Start the Server

```bash
uvicorn src.pipeline.server:app --reload --port 8000
```

Open `http://localhost:8000` in your browser. You should see the upload portal.

### 6. Upload and Process

1. Upload your video (MP4/MOV, 20–90 seconds recommended)
2. Give your scene a name
3. Watch the pipeline progress bar (SSE-driven, live updates)
4. If you have a local GPU, click "Start Local GPU Training" when the status reaches `ready_for_colab`
5. Otherwise, use the Colab workflow (see [GPU Training](#gpu-training))
6. Click "View 3D Scene" when the job status shows `ready`

---

## GPU Training

### Option A — Local GPU

After COLMAP finishes, the UI shows a "Start Local GPU Training" button for jobs in `ready_for_colab` status. This calls `POST /api/train-local/{job_id}`, which runs:

```bash
python scripts/train_local_gpu.py --job_id <job_id>
```

Or run directly:

```bash
python scripts/train_local_gpu.py --job_id <job_id>

# Override auto-detected profile
python scripts/train_local_gpu.py --job_id <job_id> \
    --iterations 10000 --max_gaussians 100000 --sh_degree 2

# Resume from a checkpoint
python scripts/train_local_gpu.py --job_id <job_id> \
    --resume work/<job_id>/models/checkpoints/checkpoint_005000.pkl

# Save a preview render after training
python scripts/train_local_gpu.py --job_id <job_id> --preview
```

The script auto-detects VRAM and selects a training profile:

| VRAM | GPU Examples | Iterations | Max Gaussians | Resolution | Est. Time |
|------|-------------|------------|---------------|------------|-----------|
| ≥ 20 GB | RTX 3090, 4090, A100 | 30,000 | 500,000 | 960×540 | ~20 min |
| ≥ 8 GB | RTX 3070, 3080, 4070 | 15,000 | 200,000 | 800×450 | ~30 min |
| ≥ 4 GB | RTX 3060, 2060, GTX 1650 | 7,000 | 80,000 | 640×360 | ~15–20 min |
| < 4 GB / CPU | Fallback | 1,000 | 10,000 | 480×270 | Very slow |

Times assume `gsplat` is installed and CUDA is active.

### Option B — Google Colab (Recommended for most setups)

Colab provides a free T4 (16 GB) or A100 (40 GB) GPU. For most consumer hardware, Colab is faster and easier.

| Factor | Local GTX 1650 | Google Colab (T4/A100) |
|--------|---------------|------------------------|
| Training time (gsplat) | 15–20 min | 10–20 minutes |
| Setup required | CUDA toolkit, compiler | None |
| VRAM | 4 GB | 16 GB (T4) / 40 GB (A100) |
| Cost | Free (electricity) | Free tier available |

**Colab workflow:**

1. Package the job:
   ```bash
   python scripts/zip_for_colab.py <job_id>
   ```
   This creates `<job_id>_for_colab.zip` containing the extracted frames, COLMAP poses, config, and training scripts.

2. Upload the zip to the Colab notebook (`notebooks/monosplat_colab_gpu.ipynb`) and run all cells.

3. Download the output (`.splat` and `.ply`) and place in `work/<job_id>/models/gaussian/`.

4. Update `models/registry.json`:
   ```json
   {
     "status": "ready",
     "ply_path": "D:\\path\\to\\work\\<job_id>\\models\\gaussian\\<job_id>.ply",
     "splat_path": "D:\\path\\to\\work\\<job_id>\\models\\gaussian\\<job_id>.splat",
     "num_gaussians": 80000
   }
   ```

5. Stop and restart the server, then open `http://localhost:8000/viewer/<job_id>`.

Or use the helper script:
```bash
python scripts/mark_ready.py <job_id>
```

---

## Configuration Reference

**File:** `config/config.yaml`

```yaml
project:
  name: "MonoSplat"
  version: "3.0.0"
  mode: "object"          # object | scene | panoramic

data:
  video_fps: null         # null = adaptive; set a number to override (e.g. 5)
  max_frames: 600         # hard cap on extracted frames

colmap:
  binary_path: "colmap"
  quality: "medium"       # low / medium / high
  camera_model: "OPENCV"  # handles radial distortion for phone video
  single_camera: true     # one lens model for all frames in the same video

training:
  iterations: 30000
  iterations_cpu: 1000    # hard cap when running on CPU (no GPU)
  save_every: 5000        # checkpoint every N iterations
  eval_every: 1000        # PSNR/SSIM evaluation every N iterations

  learning_rate:
    position:       0.00016
    feature:        0.0025
    opacity:        0.05
    scaling:        0.005
    rotation:       0.001
    position_final: 0.0000016  # exponential decay target

  densify_from_iter:       500    # start densification at this iteration
  densify_until_iter:     15000   # stop densification at this iteration
  densification_interval: 100     # densify every N iterations
  opacity_reset_interval: 3000    # reset opacities every N iterations
  percent_dense:           0.01   # Gaussians > 1% of scene extent are split
  densify_grad_threshold:  0.0002 # 3DGS default gradient threshold
  lambda_dssim:            0.2    # SSIM weight in combined loss

renderer:
  background_color: [1.0, 1.0, 1.0]  # white background
  sh_degree: 3                         # full SH (16 bands) for best view-dependence
  max_gaussians: 1000000               # 1M cap — browser-safe
  batch_size: 5000                     # software renderer batch size

viewer:
  window_width: 800
  window_height: 800
  target_fps: 60
  fov_degrees: 60.0
  near_plane: 0.01
  far_plane: 100.0

cloud_storage:
  enabled: false
  type: "local"             # "s3" | "gcs" | "local"
  s3:
    bucket: "monosplat-jobs"
    region: "us-east-1"
  gcs:
    bucket: "monosplat-jobs"

ai_layer:
  enabled: false
  detection_model: "yolov8n.pt"
  segmentation_model: "facebook/sam-vit-base"
  qa_model: "gpt2"
  detection_confidence: 0.5
```

---

## God Mode Features (v3.0)

### Stage 1: Capture Quality Warnings

The frame extractor runs a multi-stage quality analysis during extraction:

- **Motion detection** via optical flow (OpenCV) — warns if camera movement is too fast or too slow.
- **Exposure validation** — detects overexposed and underexposed frames.
- **Blur filtering** — Laplacian variance threshold (default: 80.0; adjustable in config).
- **Resolution hard-fail** — rejects frames smaller than 256×256px before COLMAP.

Warnings appear in the job card as a yellow banner.

### Stage 2: WebXR VR/AR Mode

The Three.js viewer includes full WebXR support:

**VR Mode:**
- Button: `🥽 VR` or press `V`
- Requires WebXR-compatible headset (Meta Quest, HTC Vive, etc.)
- Full 6DOF tracking with hand controller support

**AR Mode:**
- Button: `📱 AR`
- Requires mobile device with WebXR support
- Place 3D scenes in the real environment via AR overlay

### Stage 3: Progressive Chunk Loading

Large scenes (> 10,000 Gaussians) are automatically chunked:
- Chunks load in coarse-to-fine order (most important first)
- Real-time loading progress indicator
- Fallback to monolithic loading if chunks are unavailable

### Stage 4: SPZ Compression

Compressed `.spz` files are generated alongside `.splat`:
- Significant size reduction with minimal quality loss
- Available for download from the viewer

### Stage 5: Cloud Storage Integration

Completed jobs are automatically uploaded to cloud storage when enabled. Supported backends:

- **AWS S3:** `type: "s3"` — uses `boto3`. Supports IAM role credentials (recommended) or explicit key/secret.
- **Google Cloud Storage:** `type: "gcs"` — uses `google-cloud-storage`. Supports service account JSON.
- **Local:** `type: "local"` — copies to a local `cloud_storage/` directory. Default, no credentials required.

Cloud URLs are stored in `registry.json` and returned by the job API.

### Stage 6: Full XR Features

**Measurement Tool (`T`):**
- Click two points in the scene to measure distance
- Result displayed in scene units
- Useful for architectural and product visualisation

**Teleport Navigation:**
- Enable via `⚡ Teleport` button
- Shift+Click to teleport camera to any point in the scene
- Efficient navigation in large scenes

**Collaborative Viewing:**
- Enable via `👥 Collab` button
- Camera positions synchronise across multiple viewers (requires WebSocket server)

### Stage 7: AI Layer

Powered by YOLO, SAM, and a causal language model:

- **Object detection** — identifies common objects with confidence scores and bounding boxes
- **Semantic segmentation** — pixel-level region understanding via SAM
- **Spatial search** — `POST /api/jobs/{job_id}/ai/query` with `query_type: "class"`, `"description"`, or `"nearby"`
- **Scene QA** — `POST /api/jobs/{job_id}/ai/qa` with a natural language `question`

All AI features are disabled by default and require optional pip packages.

---

## Input Data Requirements

MonoSplat is **data-sensitive**. Most reconstruction failures are caused by incorrect capture, not code issues.

### What Works

- Real-world footage from a phone camera
- Slow, complete orbit around the subject (one step per second)
- 60–80% frame overlap between consecutive frames
- Consistent exposure locked before recording
- One continuous uninterrupted clip
- Textured subjects (edges, patterns, surface detail)

Good examples: shoes, bottles, plants, statues, room interiors, building exteriors.

### What Fails

- Logo animations, motion graphics, or screen recordings
- Videos with cuts, fades, or transitions
- Smooth, shiny, or transparent objects (plain walls, glass, mirror-finish metal)
- Fast motion causing motion blur
- 2D flat subjects or rendered CGI

These fail because COLMAP cannot find consistent SIFT feature matches between frames with near-identical or textureless surfaces.

### Common Error Symptoms

| Error | Cause | Fix |
|-------|-------|-----|
| "Could not register image" | Poor frame overlap | Walk slower, two full loops |
| "Discarding reconstruction" | Not enough valid frames | More angles, better lighting |
| Sparse or broken model | Textureless or inconsistent input | Change subject or lighting |
| Blank splat (46 Gaussians) | Too few 3D points from COLMAP | Resolution validation failure — re-upload full-res video |

---

## Capture Guide

### Recording Parameters

| Parameter | Recommendation |
|-----------|---------------|
| Duration | 35–45 seconds (~200 frames at 5 fps) |
| Motion | Slow smooth arc — one step per second |
| Frame overlap | 60–80% between consecutive frames |
| Lighting | Consistent, diffuse — avoid hard shadows |
| Exposure | Lock before recording (tap-hold on iPhone; Pro mode on Android) |
| Resolution | 1080p minimum |
| Subject framing | Fill 60–70% of the frame |

### Angle Guidelines

**For an object (shoe, bottle, plant):**
- Slow complete circle at eye level
- Second loop slightly above
- Keep object centred at consistent distance

**For a room or indoor space:**
- Walk slowly around the perimeter facing inward
- Avoid pointing at windows (overexposure destroys SIFT features)

**For architecture:**
- Walk parallel to the facade at consistent distance
- Arc around corners slowly

### Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Moving too fast | Motion blur, failed alignment | One step per second |
| Not enough angles | Holes in geometry | Two full loops minimum |
| Changing exposure | Inconsistent colours | Lock exposure before recording |
| Subject too small | Low feature density | Fill 60–70% of frame |
| Textureless subject | No SIFT features to match | Choose textured subjects |
| Video with cuts | COLMAP cannot bridge the jump | One continuous clip only |

---

## Output Formats

### `.ply` — Standard Gaussian Splat Archive

Contains positions, SH colour coefficients, opacity, scale, and rotation per Gaussian in binary PLY format. Compatible with:
- SuperSplat editor (`https://supersplat.playcanvas.com`)
- SIBR viewer (original 3DGS viewer)
- Luma AI
- Any tool supporting the standard 3DGS PLY layout

Use for archiving, further processing, or desktop apps.

### `.splat` — Browser-Optimised Binary

32 bytes per Gaussian. Each Gaussian is packed as:
- Position: 3 × float32 (12 bytes)
- Colour (SH DC term): 3 × uint8 (3 bytes)
- Opacity: 1 × uint8 (1 byte)
- Scale: 3 × uint8 (3 bytes, log-quantised)
- Rotation: 4 × uint8 (4 bytes, normalised quaternion)
- Padding: 1 byte

Direct drag-and-drop into `https://supersplat.playcanvas.com`. Served by the built-in Three.js viewer at `/viewer/{job_id}`.

Use for browser viewing, sharing, and demos.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| FFmpeg not found | Not in PATH | Install FFmpeg and add to PATH |
| COLMAP not found | Not installed | Install COLMAP and add to PATH |
| COLMAP produces no model | Poor overlap or textureless input | Two full loops, locked exposure |
| CUDA not detected despite NVIDIA GPU | PyTorch installed without CUDA build | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `loss.backward()` crash — no grad_fn | Software renderer path active (CUDA not available) | Fix CUDA/PyTorch install first |
| OOM crash during KNN init | Old code path computing full N×N distance matrix | Already fixed in `gaussian_model.py` — ensure you have the latest code |
| Training OOM | Too many Gaussians for VRAM | Lower `max_gaussians` in config or script args |
| Blank splat in browser | `.splat` not ready or registry not updated | Check job status; stop server, edit `registry.json`, restart |
| UI shows `ready_for_colab` after training | Registry not updated / server not restarted | Stop server, edit `registry.json`, restart |
| Registry edits not picked up | Server loads `registry.json` once at startup | Always stop → edit → restart |
| NaN loss during training | Degenerate Gaussians before first prune | Normal; trainer skips NaN iters automatically |
| Colab times out | Long training run | Checkpoints save every 5k iters — resume with `--resume` |
| Upload portal unreachable | Server not running | Run `uvicorn` command, check port 8000 |
| Feature matching crash (exit 3221225786) | GPU VRAM exhaustion during exhaustive matching | Reduce `max_num_features` in COLMAP config |

---

## Performance Targets

| Stage | Target Time | Notes |
|-------|------------|-------|
| Frame extraction + quality filter | ~30–60 sec | FFmpeg + OpenCV, up to 600 frames |
| COLMAP feature extraction | ~1–2 min | GPU-accelerated SIFT |
| COLMAP exhaustive matching | 2–10 min | Scales as O(N²) frame pairs |
| COLMAP sparse reconstruction | 2–5 min | Depends on scene complexity |
| Gaussian training — Colab T4 | 20–40 min | 15k–30k iterations, gsplat |
| Gaussian training — RTX 3080+ | 15–30 min | 15k–30k iterations, gsplat |
| Gaussian training — GTX 1650 | 15–20 min | 7k iterations, gsplat |
| Gaussian training — CPU fallback | Very slow (hours) | 1k iterations hard cap |
| Browser render | 30+ FPS | Three.js viewer |

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.9+ |
| Deep Learning | PyTorch | 2.1+ (cu121 build) |
| Gaussian Rasterizer | gsplat (nerfstudio-project) | 1.0+ |
| Frame Extraction | FFmpeg | 4.x+ (8.1 verified) |
| Pose Estimation (SfM) | COLMAP | 3.8+ (3.13 verified) |
| Gaussian Model | Custom `GaussianModel` (360GS-compatible API) | — |
| Gaussian Trainer | Custom `GaussianTrainer` (gsplat / software dual path) | — |
| Loss Functions | L1 + SSIM + PSNR (+ optional LPIPS) | — |
| Object Detection | YOLO (Ultralytics YOLOv8) | 8.0+ |
| Segmentation | SAM (facebook/sam-vit-base) | — |
| Scene QA | GPT-2 (HuggingFace Transformers) | 4.30+ |
| Cloud Storage | boto3 (S3) / google-cloud-storage (GCS) | latest |
| Web Server | FastAPI + Uvicorn | 0.104+ |
| Browser Viewer | Three.js (gaussian-splats-3d) | — |
| Quality Metrics | pytorch-msssim + lpips | — |
| Config | PyYAML | 6.0+ |
| 3D Formats | PLY (plyfile) + .splat binary | — |

---

## Method Comparison

### Gaussian Splatting vs NeRF

These two approaches solve the same problem — reconstruct a 3D scene from 2D images — but differ fundamentally in how they represent the scene, how they render it, and what trade-offs that creates.

#### Scene Representation

A **NeRF** (Neural Radiance Field) represents the scene as an implicit function encoded in a neural network's weights. To find out what a point in space looks like from a given direction, you query the network at that 3D coordinate and view direction. The scene has no explicit geometry — it exists only as learned numeric activations inside the network.

**Gaussian Splatting** represents the scene as a large collection of explicit 3D ellipsoids — the Gaussians. Each one is a discrete object in world space with a measurable position, size, orientation, opacity, and view-dependent colour. The scene is the union of those ellipsoids. You can iterate over them, inspect them, move them, and delete them.

```
NeRF:              f(x, y, z, θ, φ) → (RGB, density)   [neural network query]
Gaussian Splatting: {position, scale, rotation, opacity, SH_coefficients} × N   [explicit list]
```

#### Rendering

NeRF rendering requires casting a ray for every pixel and sampling 64–256 3D points along each ray, each requiring a full network forward pass. At 800×800 resolution that is roughly 50–130 million network queries per frame — which is why NeRF renders take seconds per frame, not milliseconds.

Gaussian Splatting renders by sorting all Gaussians by depth and splatting them onto the screen as 2D Gaussian blobs using the analytic projection of each 3D covariance matrix. This is a simple scan over a sorted list plus a per-pixel accumulation, which maps efficiently onto GPU tile-based rendering. MonoSplat's gsplat backend performs this with custom CUDA kernels — the result is real-time frame rates (30–60+ FPS) in a browser.

#### Training

Both approaches optimise via gradient descent against the same photometric loss. But NeRF optimises network weights, which generalise smoothly through an implicit representation. Gaussian Splatting optimises the explicit Gaussian parameters and also controls the *count* of Gaussians throughout training — splitting Gaussians that cover too much area, cloning Gaussians in under-reconstructed regions, and pruning Gaussians that contribute nothing. This adaptive density control is what allows Gaussian Splatting to converge in 15–30 minutes on a single GPU rather than several hours.

#### Editability

Because NeRF stores the scene implicitly in network weights, editing it requires re-training or complex latent-space manipulation. You cannot move an object by translating a few weights.

Because every Gaussian in a Gaussian Splat is an explicit object with a position in world space, editing is direct: move, scale, rotate, or delete individual Gaussians or groups of them. Tools like SuperSplat expose this editing directly in the browser.

#### Summary

| Property | NeRF | Gaussian Splatting |
|----------|------|--------------------|
| Scene representation | Implicit neural network (weights) | Explicit 3D Gaussian primitives (N × parameters) |
| Render method | Ray-march + neural query per sample | Depth-sort + 2D Gaussian splat |
| Render speed | Seconds per frame | Real-time: 30–60+ FPS in the browser |
| Training time | Several hours on GPU | 15–30 min on GPU (with adaptive densification) |
| Editability | Difficult — scene is implicit | Direct — Gaussians are explicit objects |
| Browser support | Requires server-side rendering or baking | Native via Three.js, zero-install |
| Memory scaling | Compact (network weights, ~50–200 MB typical) | Scales with scene complexity: N × 32 bytes per splat |
| Novel view quality | Excellent, especially for view-dependent effects | Excellent; competitive with NeRF at similar training budgets |

For MonoSplat's use case — mobile capture, browser delivery, interactive viewing — Gaussian Splatting's real-time render speed and zero-install browser viewer are decisive advantages over NeRF.

---

### Why MonoSplat vs Other Pipelines

#### vs. Raw 3DGS Reference Code

The original 3DGS paper released a research CLI. It takes images from a folder and produces a trained splat, nothing more. There is no video ingestion, no automated COLMAP pipeline, no web interface, no job tracking, no live progress, and no browser viewer without installing the SIBR desktop app.

MonoSplat wraps the same core algorithm — identical Gaussian model, densification schedule, and Adam optimiser — in a production pipeline designed for the full end-to-end workflow from phone video to shareable browser link.

| Aspect | 3DGS Reference Code | MonoSplat |
|--------|---------------------|-----------|
| Input | Folder of images (manual prep) | Upload any video via browser |
| COLMAP integration | Manual: run COLMAP yourself, point the script at the output | Automated: exhaustive_matcher, sequential pipeline, 3D-point diagnostic |
| Frame quality checks | None | Blur filter, resolution hard-fail, motion detection, exposure validation |
| Training backend | `diff-gaussian-rasterization` (manual CUDA compile) | `gsplat` (`pip install gsplat`, no compile step) |
| Progress tracking | Terminal stdout | Live SSE stream → browser progress bar |
| Viewer | SIBR desktop app (install required) | Three.js in-browser, shareable URL, mobile-ready |
| Job management | None | Async job queue (thread or Redis/RQ), registry, resume support |
| Scene metadata | None | Notes, tags, cloud URLs, AI detections stored per job |
| AI layer | None | YOLO object detection, SAM segmentation, scene QA |
| XR | None | WebXR VR/AR mode, measurement tool, collaborative viewing |

#### vs. Commercial Apps (Luma AI, Polycam)

Commercial reconstruction apps are convenient but come with real trade-offs for research, education, and production use cases that require control or privacy.

| Aspect | Commercial Apps (Luma AI, Polycam) | MonoSplat |
|--------|------------------------------------|-----------|
| Cost | Subscription or per-scene fee; free tiers have limits | Free, open source (MIT) |
| Data privacy | Video and reconstruction uploaded to vendor servers | Runs entirely locally; data never leaves your machine |
| Transparency | Black-box pipeline; no visibility into COLMAP settings, training parameters, or loss functions | Full source; every parameter documented and adjustable |
| Extensibility | Fixed feature set | Fork and extend freely |
| Academic use | Not citable as a method; no access to algorithm details | Open, auditable, reproducible; every design decision documented |
| Offline use | Requires internet connection | Fully offline after initial package install |
| Custom hardware | No control over compute tier | Run on your own GPU, Colab, or any cloud instance |

The core Gaussian training code in MonoSplat — `GaussianModel`, `GaussianTrainer`, `GaussianRenderer` — is the same algorithm used inside commercial tools. The difference is that MonoSplat exposes it entirely.

---

### Software Renderer vs gsplat

MonoSplat ships with two Gaussian rendering backends. The choice is made automatically at runtime, but understanding the difference matters for performance expectations and debugging.

#### What Each Backend Is

**gsplat** is the [nerfstudio-project's](https://github.com/nerfstudio-project/gsplat) actively-maintained CUDA library for Gaussian Splatting. It implements tile-based GPU rasterization with proper depth sorting and exposes a clean Python API. Installing it is a single `pip install gsplat` — no manual CUDA compile step, no Visual Studio Build Tools, no compiler chain required.

**The software renderer** is a pure-PyTorch fallback built into `src/renderer/renderer.py`. It implements the same algorithm in plain Python — no CUDA extensions required. It runs on CPU or GPU, requires zero additional dependencies beyond PyTorch itself, and is used automatically whenever gsplat is not installed or CUDA is unavailable.

#### How Backend Selection Works

Selection is determined in `GaussianRenderer.__init__` and `GaussianTrainer.__init__` at startup — not per-frame. The criterion is simple: if `gsplat` imports successfully **and** a CUDA device is available, gsplat is used. Otherwise the software renderer is used.

```python
# trainer.py
self._use_gsplat_train = (
    _GSPLAT is not None and self.device == "cuda"
)
print(f"[Trainer] Backend : {'gsplat' if self._use_gsplat_train else 'software renderer'}")
```

This is printed in the server log at the start of every training run so you always know which path is active.

#### What the Software Renderer Actually Does

The software renderer (`_render_software` in `renderer.py`) is a correct but deliberately simple implementation. Here is the exact sequence for every frame:

1. **Transform** — multiply all Gaussian positions by the 4×4 view matrix to get camera-space coordinates. Any Gaussian with `z_cam < 0.01` (behind the near plane) is culled.

2. **Sort by depth** — `z_cam.argsort(descending=True)` sorts all surviving Gaussians back-to-front. Alpha compositing requires strict back-to-front order.

3. **Evaluate SH colours** — for each visible Gaussian, evaluate its spherical harmonic coefficients against the current view direction using `_eval_sh()`. This gives the view-dependent colour for this camera position.

4. **Project 3D covariance to 2D** (`_project_cov3d_to_2d`) — for each Gaussian, construct the rotation matrix from its quaternion, build the 3×3 world-space covariance `M @ M^T` where `M = R @ S`, then project to a 2×2 screen covariance using the perspective Jacobian `J @ W`. A regularisation of 0.3 is added to the diagonal to prevent degenerate near-zero covariances from causing numerical issues. This gives a 2D Gaussian ellipse on screen.

5. **Compute screen bounding radius** — `radius = ceil(3 × sqrt(max_eigenvalue))`, clamped to `[1, 128]` pixels. This bounds the pixel region each Gaussian affects.

6. **Batch alpha-composite** — Gaussians are processed in batches of `batch_size=5000`. For each Gaussian, a pixel grid is constructed over its bounding box, the 2D Gaussian weight `exp(−0.5 × x^T Σ⁻¹ x)` is evaluated at every pixel in the box, and the result is alpha-composited using the running transmittance:

   ```python
   alpha   = (opacity × weight).clamp(max=0.9999)
   contrib = alpha × transmittance[patch]
   accumulated[:, patch] += contrib × color
   transmittance[patch]  *= (1.0 − alpha)
   ```

7. **Composite background** — the background colour is multiplied by the remaining transmittance (pixels not fully covered by Gaussians) and added to the accumulation buffer.

#### Render Quality

Both backends produce the same result. The software renderer is not an approximation — it implements the same depth-sorted, alpha-composited Gaussian splatting algorithm as gsplat. For a fixed scene state, pixel output is numerically equivalent between backends.

#### Densification Accuracy: the Critical Training Difference

Render quality is not the main reason to prefer gsplat. The critical difference is **densification accuracy**.

The original 3DGS paper densifies based on the gradient of each Gaussian's 2D screen-space position — `means2d.grad.norm(dim=-1)`. This identifies Gaussians that are working hard to explain under-reconstructed regions: a large screen-space gradient means the optimiser is trying to move the Gaussian significantly to reduce loss, signalling that more Gaussians are needed nearby.

gsplat exposes this directly in the `meta` dict returned by `gs.rasterization()`:

```python
rendered, meta = gs.rasterization(...)
# meta["means2d"]      — 2D screen positions with gradient graph attached
# meta["radii"]        — per-Gaussian screen radius
# meta["gaussian_ids"] — maps packed tile renders back to full Gaussian indices
```

The trainer reads `means2d.grad.norm(dim=-1)` and scatters it back to the full N-Gaussian index via `gaussian_ids`. This exactly matches the 3DGS paper's criterion.

The software renderer returns a plain image tensor — it does not expose `means2d` or `radii`. The software training path falls back to `model._positions.grad.norm(dim=1)` (3D position gradient) as a proxy. This is correlated with the correct criterion but less accurate: Gaussians that are large on-screen but have small 3D position gradients can be missed, leading to slower or less complete densification.

#### Summary

| Property | gsplat (CUDA) | Software Renderer (PyTorch) |
|----------|--------------|----------------------------|
| Installation | `pip install gsplat` | Zero — built into MonoSplat |
| Requires CUDA | Yes | No (runs on CPU or GPU) |
| Render speed | Real-time (30–60+ FPS) | Slow: ~0.5–5 sec/frame depending on Gaussian count |
| Training speed | 15–40 min (GPU) | Hours on CPU; much slower on GPU |
| Densification criterion | Accurate 2D screen-space gradient (`means2d.grad`) | Approximate 3D position gradient proxy |
| Iterations cap | Full schedule (30,000 default) | Hard-capped at 1,000 on CPU (`iterations_cpu` in config) |
| Use case | All production training and rendering | Debugging, CPU-only demo runs, environments without CUDA |

#### Why Not Use `diff-gaussian-rasterization` (the Original CUDA Extension)?

The original 3DGS paper used a custom CUDA extension called `diff-gaussian-rasterization`. MonoSplat chose gsplat instead for several concrete reasons visible in the codebase:

- **No manual compile step.** `diff-gaussian-rasterization` requires cloning the repo, having CUDA Toolkit installed, a C++ compiler (Visual Studio Build Tools on Windows), and `CUDA_HOME` set correctly. `gsplat` is a standard pip wheel.
- **Active maintenance.** The original extension's development has slowed significantly. gsplat is actively maintained by the nerfstudio project with regular updates and bug fixes.
- **Cleaner API.** `diff-gaussian-rasterization` returns densification data via `screenspace_points.grad` — a side-effect of the backward pass that requires careful grad hook wiring. gsplat returns `means2d`, `radii`, and `gaussian_ids` directly in the `meta` dict, making the densification code explicit, readable, and debuggable.
- **Same output format.** gsplat produces the same `.ply` / `.splat` binary files. Switching to gsplat required zero changes to the export pipeline.

---

## Future Scope

- Dedicated GPU worker (Runpod / Lambda Labs) for production deployments
- Redis job queue for multi-user concurrent processing
- NeRF vs Gaussian Splatting comparison viewer
- VR headset support via OpenXR
- Multi-scene stitching
- Model compression (fewer Gaussians, same perceptual quality)
- Automatic CUDA environment validation on server startup

---

## References

- **3D Gaussian Splatting for Real-Time Radiance Field Rendering** — Kerbl et al., SIGGRAPH 2023  
  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- **gsplat — Gaussian Splatting Library** — nerfstudio-project  
  https://github.com/nerfstudio-project/gsplat

- **COLMAP — Structure-from-Motion and Multi-View Stereo**  
  https://colmap.github.io/

- **gaussian-splats-3d — Three.js Gaussian Splat viewer** — mkkellogg  
  https://github.com/mkkellogg/GaussianSplats3D

- **SuperSplat — Browser-based splat viewer and editor**  
  https://supersplat.playcanvas.com

- **Segment Anything Model (SAM)** — Meta AI Research  
  https://github.com/facebookresearch/segment-anything

- **YOLOv8** — Ultralytics  
  https://github.com/ultralytics/ultralytics

---

## License

MIT License — free to use for educational, research, and commercial purposes.
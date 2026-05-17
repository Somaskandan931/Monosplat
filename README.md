# MonoSplat

> **Single-camera 3D Gaussian Splat reconstruction pipeline** — record a video, run COLMAP for camera alignment, train with PyTorch on GPU, view the 3D splat in your browser.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-required-red?logo=ffmpeg)](https://ffmpeg.org)
[![COLMAP](https://img.shields.io/badge/COLMAP-3.8+-purple)](https://colmap.github.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](.)

</div>

---

## 🎯 What Is MonoSplat?

MonoSplat is an **end-to-end pipeline** that transforms a single video into a photorealistic 3D Gaussian Splat scene you can navigate in your browser — no desktop app, no OpenGL setup required.

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Video Input │ →  │ Frame Extract│ →  │ Camera Poses│ →  │ Gaussian     │ →  │ Browser     │
│ MP4/MOV     │    │ FFmpeg       │    │ COLMAP SfM  │    │ Training     │    │ Viewer      │
│             │    │              │    │             │    │ PyTorch GPU  │    │ Three.js    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```

### ✨ Key Features

- **🎥 Single Video Input** — Just record a video with your phone
- **🔧 Fully Automated** — From upload to 3D viewer, zero manual intervention
- **🌐 Browser-Based** — View your 3D scene at `http://localhost:8000/viewer/<job_id>`
- **🚀 GPU Accelerated** — Training on Colab T4 or local CUDA GPU
- **📦 Multiple Export Formats** — `.splat` for web, `.ply` for desktop apps
- **🤖 AI-Powered** — Object detection, segmentation, and scene QA (optional)
- **🥽 WebXR Support** — VR/AR mode with measurement tools (optional)

---

## ⚡ Quick Start

### Prerequisites

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install PyTorch with CUDA support (REQUIRED for GPU training)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 4. Install FFmpeg
# Ubuntu: sudo apt install ffmpeg
# macOS:  brew install ffmpeg
# Windows: https://ffmpeg.org/download.html

# 5. Install COLMAP
# Ubuntu: sudo apt install colmap
# macOS:  brew install colmap
# Windows: https://github.com/colmap/colmap/releases
```

### Run the Server

```bash
uvicorn src.pipeline.server:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser and upload a video!

---

## 📋 Table of Contents

- [What Is MonoSplat?](#-what-is-monosplat)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Pipeline Architecture](#-pipeline-architecture)
- [GPU Training](#-gpu-training)
- [Configuration](#-configuration)
- [Capture Guide](#-capture-guide)
- [Troubleshooting](#-troubleshooting)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🔬 How It Works

### Gaussian Splatting vs Traditional ML

**Gaussian Splat training is fundamentally different from normal ML models.** A conventional ML model trains once and generalizes to new inputs. Gaussian Splatting **overfits intentionally** — the trained model *is* the scene itself.

```
Video A  →  COLMAP  →  PyTorch train  →  scene_A.splat   (represents ONLY scene A)
Video B  →  COLMAP  →  PyTorch train  →  scene_B.splat   (represents ONLY scene B)
```

Every new video requires a new training run. This is fundamental to the technology, not a limitation of this codebase.

### The Pipeline

| Stage | Tool | Purpose | Output |
|-------|------|---------|--------|
| **1. Frame Extraction** | FFmpeg | Extract keyframes from video | `work/<job_id>/frames/*.png` |
| **2. Camera Poses** | COLMAP SfM | Estimate real camera positions | `cameras.txt`, `images.txt`, `points3D.txt` |
| **3. Gaussian Training** | PyTorch + gsplat | Train 3D Gaussian scene | `<job_id>.ply`, `<job_id>.splat` |
| **4. Browser Viewer** | Three.js | Real-time 3D visualization | Interactive web viewer |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                            │
│         Upload Video  →  "Processing…"  →  View 3D Scene        │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP
┌───────────────────────────────▼─────────────────────────────────┐
│              FastAPI Backend (src/pipeline/server.py)            │
│  • Receives video upload                                         │
│  • Creates job in ModelRegistry (models/registry.json)          │
│  • Returns job_id, streams live status via SSE                  │
│  • Runs frame extraction (FFmpeg) locally                       │
│  • Serves .splat files for browser viewing                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Job handoff
┌───────────────────────────────▼─────────────────────────────────┐
│                  COLMAP (SfM Alignment — CPU)                   │
│  • Reads extracted frames                                       │
│  • Detects SIFT keypoints, matches sequentially                 │
│  • Estimates real camera positions and orientations             │
│  • Exports cameras.txt, images.txt, points3D.txt                │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Pose handoff
┌───────────────────────────────▼─────────────────────────────────┐
│              PyTorch GPU Worker (Colab / Local CUDA)            │
│  • Reads COLMAP poses + frames                                  │
│  • Runs Gaussian Splat training (GPU, 10–40 min)                │
│  • Exports .ply + .splat                                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│           Storage (models/registry.json + work/ directory)       │
│  • Stores .splat and .ply files per job                         │
│  • Served by FastAPI viewer endpoint                            │
└─────────────────────────────────────────────────────────────────┘
```

### What Runs Where

| Stage | Tool | Runs On | Time |
|-------|------|---------|------|
| Frame extraction | FFmpeg | Local CPU | ~30 sec |
| SfM pose estimation | COLMAP | Local (GPU-accelerated) | ~5–15 min |
| Gaussian Splat training | PyTorch + gsplat | Colab T4 / Local CUDA GPU | 10–40 min |
| Browser viewer | FastAPI + Three.js | Local | Real-time |

---

## 🎮 Recommended Workflow

```
1.  Record phone video (35–45 sec, slow orbit, locked exposure)
2.  Start server: uvicorn src.pipeline.server:app --reload --port 8000
3.  Upload via browser at http://localhost:8000
4.  Server auto-runs frame extraction (FFmpeg, ~30 sec)
5.  COLMAP alignment runs locally (~5–15 min)
6.  Status changes to "ready_for_colab" in the UI
7a. Option A — Local GPU: Click "Start Local GPU Training" in UI
7b. Option B — Google Colab: Zip the job, upload to Colab notebook
8.  Training completes → .splat and .ply written to work/<job_id>/models/gaussian/
9.  Update models/registry.json, restart server
10. View at http://localhost:8000/viewer/<job_id>
```

---

## 🖥️ GPU Training

### Option A — Local GPU

After COLMAP finishes, the UI shows a "Start Local GPU Training" button. You can also run directly:

```bash
# Basic usage
python scripts/train_local_gpu.py --job_id <job_id>

# Override settings
python scripts/train_local_gpu.py --job_id <job_id> \
    --iterations 10000 --max_gaussians 100000 --sh_degree 2

# Resume from checkpoint
python scripts/train_local_gpu.py --job_id <job_id> \
    --resume work/<job_id>/models/checkpoints/checkpoint_005000.pkl

# Render preview after training
python scripts/train_local_gpu.py --job_id <job_id> --preview
```

The script auto-detects your GPU VRAM and picks an optimal training profile:

| VRAM | GPU Examples | Iterations | Gaussians | Resolution | Est. Time* |
|------|-------------|------------|-----------|------------|-----------|
| ≥ 20 GB | RTX 3090, 4090, A100 | 30,000 | 500,000 | 960×540 | ~20 min |
| ≥ 8 GB | RTX 3070, 3080, 4070 | 15,000 | 200,000 | 800×450 | ~30 min |
| ≥ 4 GB | RTX 3060, 2060, GTX 1650 | 7,000 | 80,000 | 640×360 | ~15–20 min |
| < 4 GB / CPU | Fallback | 1,000 | 10,000 | 480×270 | Very slow |

*\*With gsplat CUDA rasterizer installed.*

### Option B — Google Colab (Recommended)

Unless you have an RTX 3080+ with the CUDA rasterizer, Colab is the better choice. You get a T4 (16 GB) or A100 (40 GB) for free.

| Factor | Local GTX 1650 | Google Colab (T4/A100) |
|--------|---------------|------------------------|
| Training time (no rasterizer) | 3–4 hours | 10–20 minutes |
| Training time (with rasterizer) | 15–20 min | 10–20 minutes |
| Setup required | CUDA toolkit, Build Tools | None |
| VRAM | 4 GB | 16 GB (T4) / 40 GB (A100) |
| Cost | Your electricity | Free tier |

**How to use Colab:**

1. **Zip the job folder:**
   ```bash
   python scripts/zip_for_colab.py <job_id>
   ```

2. **Upload to Colab notebook** (`notebooks/monosplat_colab_gpu.ipynb`) and run all cells

3. **Download output** and place files in `work/<job_id>/models/gaussian/`

4. **Update `models/registry.json`** and restart the server

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize the pipeline:

```yaml
# Frame extraction
data:
  video_fps: null        # null = adaptive; set number to override
  max_frames: 600        # hard cap on extracted frames

# COLMAP settings
colmap:
  binary_path: "colmap"
  quality: "medium"      # low / medium / high
  camera_model: "OPENCV"
  single_camera: true

# Training parameters
training:
  iterations: 30000
  save_every: 5000
  densify_from_iter: 500
  densify_until_iter: 15000
  densification_interval: 100
  densify_grad_threshold: 0.0002
  lambda_dssim: 0.2

# Renderer settings
renderer:
  background_color: [1.0, 1.0, 1.0]
  sh_degree: 3
  max_gaussians: 1000000
  use_gsplat: true
  batch_size: 5000

# Optional: AI Layer
ai_layer:
  enabled: false
  detection_model: "yolov8n.pt"

# Optional: Cloud Storage
cloud_storage:
  enabled: false
  type: "local"  # "s3", "gcs", or "local"
```

---

## 📸 Capture Guide

### What Works ✅

- **Real-world footage** shot with a phone camera
- **Slow complete orbit** around the subject (60–80% frame overlap)
- **Consistent exposure** and lighting throughout
- **One continuous** uninterrupted clip
- **Textured subjects** (edges, patterns, surface detail)

**Good examples:** shoe, bottle, plant, statue, room, building exterior

### What Fails ❌

- Logo animations, motion graphics, screen recordings
- Videos with cuts, fades, or transitions
- Smooth, shiny, or transparent objects (plain walls, glass, metal spheres)
- Fast motion causing motion blur
- 2D content or rendered CGI

These fail because COLMAP cannot find consistent feature matches between frames.

### Recording Tips

| Parameter | Recommendation |
|-----------|---------------|
| Duration | 35–45 seconds (~200 frames at 5 fps) |
| Motion | Slow smooth arc — one step per second |
| Frame overlap | 60–80% between consecutive frames |
| Lighting | Consistent, diffuse — avoid hard shadows |
| Exposure | Lock before recording (tap-hold on iPhone, Pro mode on Android) |
| Resolution | 1080p minimum |
| Subject framing | Fill 60–70% of the frame |

### Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Moving too fast | Motion blur, failed alignment | One step per second |
| Not enough angles | Holes in geometry | Two full loops minimum |
| Changing exposure | Inconsistent colors | Lock exposure before recording |
| Subject too small | Low feature density | Fill 60–70% of frame |
| Textureless subject | No SIFT features to match | Choose textured subjects |
| Video with cuts | COLMAP cannot bridge the jump | One continuous clip only |

---

## 🔧 Troubleshooting

### OOM crash during KNN initialization

**Symptom:**
```
RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 25600000000 bytes
```

**Cause:** The kNN computation requires a large distance matrix.

**Fix:** The pipeline uses batched GPU kNN that processes 2,048 rows at a time (~640 MB peak). If you still get OOM, reduce `max_gaussians` in your training config.

### CUDA not detected despite NVIDIA GPU

**Symptom:** `⚠ CUDA not available — running on CPU`

**Cause:** PyTorch was installed without CUDA support (the default `pip install torch`).

**Fix:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### gsplat rasterization AssertionError

**Symptom:**
```
AssertionError: torch.Size([30000, 3])
```

**Cause:** Feature tensor shape mismatch with gsplat API expectations.

**Fix:** This has been resolved in the latest code. The pipeline now automatically handles tensor shape normalization. Simply re-run your training — Cell 5 of the Colab notebook will fetch the updated code automatically.

### Registry edits not picked up by UI

**Cause:** The server loads `models/registry.json` once at startup into memory.

**Fix:** Stop the server, edit the file, restart:
```bash
# Stop server (Ctrl+C)
# Edit models/registry.json
uvicorn src.pipeline.server:app --reload --port 8000
```

### COLMAP produces no model

**Cause:** Poor frame overlap or textureless input.

**Fix:** Record with two full loops, locked exposure, and ensure your subject has visible texture.

### Browser viewer shows blank screen

**Cause:** .splat file not ready or registry not updated.

**Fix:** Wait for `ready` status in the UI, then update `models/registry.json` and restart the server.

---

## 📊 Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.9+ |
| Deep Learning | PyTorch | 2.0+ (cu121 build) |
| Frame Extraction | FFmpeg | 4.x+ (8.1 verified) |
| Pose Estimation (SfM) | COLMAP | 3.8+ (3.13 verified) |
| Gaussian Training | Custom PyTorch + gsplat | 1.5.3+ |
| Cloud Training | Google Colab (T4 GPU) | — |
| Web Server | FastAPI + Uvicorn | 0.104+ |
| Browser Viewer | Three.js (gaussian-splats-3d) | — |
| 3D Formats | PLY + .splat binary | — |
| Config | PyYAML | 6.0+ |

---

## 📄 Output Formats

### .ply (Standard Gaussian Splat Archive)
Contains positions, SH colors, opacity, scale, rotation per Gaussian. Compatible with SuperSplat, SIBR viewers, Luma AI. Use for archiving, further processing, desktop apps.

### .splat (Browser-Optimized Binary)
32 bytes per splat. Direct drag-and-drop into [SuperSplat](https://supersplat.playcanvas.com). Served by the built-in Three.js viewer at `/viewer/<job_id>`. Use for browser viewing, sharing, and demos.

---

## 🔮 Future Scope

- [ ] Dedicated GPU worker (Runpod / Lambda Labs) for production
- [ ] Job queue with Redis for multi-user support
- [ ] Cloud storage (S3 / Cloudflare R2) for persistent splat hosting
- [ ] NeRF vs Gaussian Splatting comparison viewer
- [ ] VR headset support (OpenXR)
- [ ] Multi-scene stitching
- [ ] Model compression (fewer splats, same quality)
- [ ] CUDA rasterizer auto-install on setup

---

## 📚 References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion and Multi-View Stereo
- [gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) — Three.js Gaussian Splat viewer
- [SuperSplat](https://supersplat.playcanvas.com) — Browser-based splat viewer and editor
- [gsplat](https://github.com/nerfstudio-project/gsplat) — CUDA rasterizer

---

## 📄 License

MIT License — free to use for educational and research purposes.

---

<div align="center">

**Made with ❤️ by [Somaskandan](https://github.com/Somaskandan931)**

[Report Issue](https://github.com/Somaskandan931/Monosplat/issues) · [Request Feature](https://github.com/Somaskandan931/Monosplat/issues) · [GitHub](https://github.com/Somaskandan931/Monosplat)

</div>
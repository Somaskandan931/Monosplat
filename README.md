# MonoSplat

> Single-camera 3D Gaussian Splat reconstruction pipeline — record a video,
> run COLMAP for camera alignment, train with PyTorch on GPU, view the 3D splat in your browser.
> **Now with AI-powered scene understanding, XR support, and cloud storage integration.**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)
![FFmpeg](https://img.shields.io/badge/FFmpeg-required-red)
![COLMAP](https://img.shields.io/badge/COLMAP-3.8+-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-God%20Mode-success)

---

## What Is This?

MonoSplat is an end-to-end pipeline that takes a single video input and reconstructs a
photorealistic 3D Gaussian Splat scene you can navigate in your browser — no desktop app,
no OpenGL setup required.

```
Video Input  →  Frame Extraction  →  Camera Poses  →  Gaussian Training  →  Browser Viewer
   MP4/MOV         FFmpeg              COLMAP (SfM)     PyTorch (GPU)          Three.js
                    ↓                   ↓               ↓                    ↓
              Quality Warnings    Cloud Upload    AI Analysis         XR Features
```

### Pipeline Stack

| Stage | Tool | Why |
|-------|------|-----|
| Frame extraction | **FFmpeg** | Fast, stable, broad codec support |
| Camera alignment (SfM) | **COLMAP** | Geometry-based, real camera poses from real captured video |
| Gaussian Splat training | **Custom PyTorch** | GPU-accelerated, high-quality 3D Gaussian output |
| Browser viewer | **Three.js** | Zero-install, cross-platform, real-time |
| Quality validation | **OpenCV + Custom** | Blur, motion, and exposure detection |
| Cloud storage | **S3/GCS/Local** | Persistent scene storage and sharing |
| AI Layer | **YOLO + SAM + Transformers** | Object detection, segmentation, QA |
| XR Features | **WebXR + Three.js** | VR/AR mode, measurements, collaboration |

This pipeline is:
- ✔ Geometry-based (real camera poses, not neural approximations)
- ✔ Works on real captured phone videos
- ✔ Fully open source

---

## Problem Statement

Creating high-quality 3D reconstructions traditionally requires:
- Multi-camera rigs or depth sensors
- Complex photogrammetry workflows
- Heavy rendering pipelines that are not real-time

MonoSplat addresses this by enabling:
- Single-camera capture (phone video)
- Automated reconstruction via COLMAP SfM
- Real-time browser-based visualization via Three.js

The goal is to make 3D scene capture as simple as recording a video.

---

## How Gaussian Splatting Actually Works

**Gaussian Splat training is not like a normal ML model.** A conventional ML model
trains once and runs inference on any new input. Gaussian Splatting works differently —
the trained model *is* the scene. It overfits intentionally to one specific capture.

```
Video A  →  COLMAP  →  PyTorch train  →  scene_A.splat   (only represents scene A)
Video B  →  COLMAP  →  PyTorch train  →  scene_B.splat   (only represents scene B)
```

There is no shared weights file that generalizes to new scenes. Every new video requires
a new training run. This is fundamental to the technology, not a limitation of this codebase.

---

## Product Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER (Browser)                          │
│   Upload video  →  "Processing…"  →  View 3D scene          │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│            FastAPI Backend  (src/pipeline/server.py)        │
│  - Receives video upload                                    │
│  - Creates job in ModelRegistry (models/registry.json)      │
│  - Returns job_id, streams live status via SSE              │
│  - Serves .splat files for browser viewing                  │
│  - Runs frame extraction (FFmpeg) locally                   │
└───────────────────────────┬─────────────────────────────────┘
                            │ Job handoff
┌───────────────────────────▼─────────────────────────────────┐
│         COLMAP  (SfM Alignment — CPU/local)                 │
│  - Reads extracted frames                                   │
│  - Detects SIFT keypoints, matches sequentially             │
│  - Estimates real camera positions and orientations         │
│  - Exports cameras.txt, images.txt, points3D.txt            │
└───────────────────────────┬─────────────────────────────────┘
                            │ Pose handoff
┌───────────────────────────▼─────────────────────────────────┐
│         PyTorch GPU Worker  (Colab / local CUDA GPU)        │
│  - Reads COLMAP poses + frames                              │
│  - Runs Gaussian Splat training (GPU, 20–40 min)            │
│  - Exports .ply + .splat                                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│         Storage  (models/registry.json + work/ directory)   │
│  - Stores .splat and .ply files per job                     │
│  - Served by FastAPI viewer endpoint                        │
└─────────────────────────────────────────────────────────────┘
```

### What Runs Where

| Stage | Tool | Runs On | Notes |
|-------|------|---------|-------|
| Frame extraction | FFmpeg | Local | ~30 sec, CPU is fine |
| SfM pose estimation | COLMAP | Local | ~5–15 min, GPU-accelerated SIFT |
| Gaussian Splat training | PyTorch | Colab T4 / local CUDA GPU | GPU required, 20–40 min |
| FastAPI server + viewer | FastAPI + Three.js | Local | Serves files, no GPU needed |

### Recommended Developer Workflow

```
1.  Record phone video (35–45 sec, slow orbit, locked exposure)
2.  Start server locally → upload via browser at http://localhost:8000
3.  Server auto-runs frame extraction (FFmpeg, ~30 sec)
4.  COLMAP alignment runs locally (~5–15 min)
5.  Status changes to "ready_for_colab" in the UI
6a. Option A — Local GPU: click "Start Local GPU Training" in the UI
6b. Option B — Google Colab: zip the job, upload to Colab notebook
7.  Training completes → .splat and .ply written to work/<job_id>/models/gaussian/
8.  Update models/registry.json, restart server
9.  View at http://localhost:8000/viewer/<job_id>
```

---

## 🚀 Quick Start (Execution Guide)

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (required for GPU training)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is working
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2. Install External Tools

```bash
# FFmpeg (required for frame extraction)
# Ubuntu:  sudo apt install ffmpeg
# macOS:   brew install ffmpeg
# Windows: https://ffmpeg.org/download.html

# COLMAP (required for SfM pose estimation)
# Ubuntu:  sudo apt install colmap
# macOS:   brew install colmap
# Windows: https://github.com/colmap/colmap/releases
```

### 3. Configure the Project

Edit `config/config.yaml` to enable/disable features:

```yaml
# Enable cloud storage (Stage 5)
cloud_storage:
  enabled: false  # Set to true to enable S3/GCS uploads
  type: "local"   # "s3", "gcs", or "local"

# Enable AI Layer (Stage 7)
ai_layer:
  enabled: false  # Set to true to enable AI analysis
  detection_model: "yolov8n.pt"
```

### 4. Start the Server

```bash
uvicorn src.pipeline.server:app --reload --port 8000
```

### 5. Upload and Process

1. Open browser to `http://localhost:8000`
2. Upload your video (MP4/MOV, 20-90 seconds recommended)
3. The pipeline runs automatically:
   - Frame extraction with quality validation
   - COLMAP pose estimation
   - Gaussian training (GPU required)
   - AI analysis (if enabled)
   - Cloud upload (if enabled)
4. View the 3D scene in the browser viewer

### 6. Use XR Features

In the viewer, use these controls:
- **🥽 VR** - Enter VR mode (requires WebXR-compatible headset)
- **📱 AR** - Enter AR mode (mobile device with WebXR support)
- **📏 Measure** - Click two points to measure distance (T key)
- **⚡ Teleport** - Shift+Click to teleport camera
- **👥 Collab** - Enable collaborative viewing mode
- **R** - Reset camera
- **F** - Fullscreen
- **H** - Toggle controls
- **M** - Toggle metrics
- **A** - Toggle annotations

---

## 🎯 God Mode Features (New in v3.0)

### Stage 1: Capture Quality Warnings

The pipeline now analyzes your video during frame extraction and warns about quality issues:

- **Motion detection**: Warns if camera movement is too fast or too slow
- **Exposure validation**: Detects overexposed and underexposed frames
- **Blur filtering**: Identifies and removes blurry frames

Warnings appear in the job card as a yellow banner with specific recommendations.

**Configuration:**
```yaml
# Blur threshold (default: 80.0)
# Higher = more permissive, Lower = stricter
```

### Stage 2: WebXR VR/AR Mode

The viewer now supports WebXR for immersive VR and AR experiences.

**VR Mode:**
- Click the "🥽 VR" button or press **V**
- Requires a WebXR-compatible headset (Meta Quest, HTC Vive, etc.)
- Full 6DOF tracking with hand controller support

**AR Mode:**
- Click the "📱 AR" button
- Requires a mobile device with WebXR support
- Place 3D scenes in your real environment

### Stage 3: Progressive Chunk Loading

Large scenes are now loaded progressively for faster initial viewing:

- Scenes with >10,000 Gaussians are automatically chunked
- Chunks load in order of importance (coarse LOD first)
- Fallback to monolithic loading if chunks unavailable
- Real-time progress indicator during loading

**Configuration:**
```yaml
# Chunk size (default: 50,000 splats per chunk)
# Adjust based on your network bandwidth
```

### Stage 4: SPZ Compression

Compressed splat format for efficient storage and transfer:

- `.spz` files are automatically generated alongside `.splat`
- Significant size reduction with minimal quality loss
- Download `.spz` from the viewer for compressed storage

### Stage 5: Cloud Storage Integration

Upload scenes to cloud storage for persistence and sharing:

**Supported backends:**
- **AWS S3** - Set `type: "s3"` in config
- **Google Cloud Storage** - Set `type: "gcs"` in config
- **Local filesystem** - Set `type: "local"` (default)

**Configuration:**
```yaml
cloud_storage:
  enabled: true
  type: "s3"
  s3:
    bucket: "monosplat-jobs"
    region: "us-east-1"
    aws_access_key_id: "YOUR_KEY"  # Optional, use IAM roles in production
    aws_secret_access_key: "YOUR_SECRET"
```

**Usage:**
After enabling, all completed jobs are automatically uploaded to cloud storage. Cloud URLs are stored in the job metadata for easy sharing.

### Stage 6: Full XR Features

Enhanced viewer with professional XR capabilities:

**Measurement Tool:**
- Click "📏 Measure" or press **T**
- Click two points in the scene
- Distance is displayed in scene units
- Useful for architectural and product visualization

**Teleport Navigation:**
- Click "⚡ Teleport" to enable
- Shift+Click anywhere to teleport camera
- Quick navigation in large scenes

**Collaborative Viewing:**
- Click "👥 Collab" to enable
- Camera positions sync across multiple viewers (requires WebSocket server)
- Great for remote collaboration and presentations

**Keyboard Shortcuts:**
- **V** - Enter VR mode
- **T** - Toggle measurement tool
- **C** - Toggle collaborative mode
- **R** - Reset camera
- **F** - Fullscreen
- **H** - Toggle controls HUD
- **M** - Toggle metrics panel
- **A** - Toggle annotations
- **Space** - Toggle auto-rotate

### Stage 7: AI Layer

Intelligent scene understanding powered by modern AI models:

**Object Detection:**
- Automatically detects objects in your scene (using YOLO)
- Identifies common objects: chairs, tables, bottles, plants, etc.
- Detection count displayed in job metrics

**Semantic Segmentation:**
- Pixel-level scene understanding (using SAM)
- Identifies distinct regions and objects
- Useful for scene editing and analysis

**Spatial Search:**
- Query scenes by object class: "Find all chairs"
- Natural language search: "Find wooden objects"
- Spatial queries: "Find objects near position (x,y,z)"

**Scene QA:**
- Ask natural language questions about your scene
- Examples:
  - "What objects are in this scene?"
  - "How many chairs are visible?"
  - "Describe the lighting conditions"

**Configuration:**
```yaml
ai_layer:
  enabled: true
  detection_model: "yolov8n.pt"  # yolov8n.pt, yolov8s.pt, yolov8m.pt
  segmentation_model: "facebook/sam-vit-base"
  qa_model: "gpt2"
  detection_confidence: 0.5
```

**API Endpoints:**
```bash
# Get AI results for a job
GET /api/jobs/{job_id}/ai

# Query the scene
POST /api/jobs/{job_id}/ai/query
{
  "query": "chair",
  "query_type": "class"  # "class", "description", "nearby"
}

# Ask a question
POST /api/jobs/{job_id}/ai/qa
{
  "question": "What objects are in this scene?"
}
```

---

## Setup

### Prerequisites

```bash
# Python dependencies
pip install -r requirements.txt

# PyTorch with CUDA support (required for GPU training)
# The default pip install of torch is CPU-only — you need the CUDA build:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is working
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True + your GPU name

# FFmpeg (required for frame extraction)
# Ubuntu:  sudo apt install ffmpeg
# macOS:   brew install ffmpeg
# Windows: https://ffmpeg.org/download.html
#          or via conda: conda install -c conda-forge ffmpeg

# COLMAP (required for SfM pose estimation)
# Ubuntu:  sudo apt install colmap
# macOS:   brew install colmap
# Windows: https://github.com/colmap/colmap/releases
#          or via conda: conda install -c conda-forge colmap
```

### Verify Tool Versions

```powershell
# Windows PowerShell
colmap help 2>&1 | Select-Object -First 3      # should show COLMAP 3.8+
ffmpeg -version 2>&1 | Select-Object -First 1  # should show ffmpeg version 4.x+
```

```bash
# Linux / macOS
colmap --version    # 3.8+
ffmpeg -version     # 4.x+
```

Verified working versions: **COLMAP 3.13.0** (with CUDA), **FFmpeg 8.1**

### Quick Start

```bash
uvicorn src.pipeline.server:app --reload --port 8000
# Open browser → http://localhost:8000
# Upload your video → pipeline runs automatically
```

---

## GPU Training

### Option A — Local GPU

After COLMAP finishes the UI shows a "Start Local GPU Training" button (for jobs in
`ready_for_colab` status). Clicking it calls `POST /api/train-local/<job_id>` which
launches:

```bash
python scripts/train_local_gpu.py --job_id 
```

You can also run it directly:

```bash
python scripts/train_local_gpu.py --job_id 

# Override profile settings if needed
python scripts/train_local_gpu.py --job_id  \
    --iterations 10000 --max_gaussians 100000 --sh_degree 2

# Resume from checkpoint
python scripts/train_local_gpu.py --job_id  \
    --resume work//models/checkpoints/checkpoint_005000.pkl

# Render a preview image after training
python scripts/train_local_gpu.py --job_id  --preview
```

The script auto-detects your GPU VRAM and picks a training profile:

| VRAM | GPU Examples | Iterations | Gaussians | Resolution | Est. Time* |
|------|-------------|------------|-----------|------------|-----------|
| ≥ 20 GB | RTX 3090, 4090, A100 | 30,000 | 500,000 | 960×540 | ~20 min |
| ≥ 8 GB | RTX 3070, 3080, 4070 | 15,000 | 200,000 | 800×450 | ~30 min |
| ≥ 4 GB | RTX 3060, 2060, GTX 1650 | 7,000 | 80,000 | 640×360 | ~15–20 min |
| < 4 GB / CPU | Fallback | 1,000 | 10,000 | 480×270 | Very slow |

*With `diff-gaussian-rasterization` installed. Without it, the 4 GB tier takes 3–4 hours.

### Option B — Google Colab (Recommended for most setups)

Unless you have an RTX 3080+ with the CUDA rasterizer installed, Colab is the better
choice. You get a T4 (16 GB) or A100 (40 GB) for free with no setup.

| Factor | Local GTX 1650 | Google Colab (T4/A100) |
|--------|---------------|------------------------|
| Training time (no rasterizer) | 3–4 hours | 10–20 minutes |
| Training time (with rasterizer) | 15–20 min | 10–20 minutes |
| Setup required | CUDA toolkit, Build Tools, compiler | None |
| VRAM | 4 GB | 16 GB (T4) / 40 GB (A100) |
| Cost | Free (your electricity) | Free tier available |

**How to use Colab:**

1. Zip the job folder:
   ```bash
   python scripts/zip_for_colab.py <job_id>
   ```
2. Upload the zip to `notebooks/monosplat_colab_gpu.ipynb` and run all cells
3. Download the output and place files in `work/<job_id>/models/gaussian/`
4. Update `models/registry.json` and restart the server (see below)

---

## diff-gaussian-rasterization (Optional CUDA Rasterizer)

This C++ CUDA extension speeds up rendering ~100× and is the difference between
3–4 hours and 15–20 minutes on a GTX 1650.

**Requirements before installing:**
- Visual Studio Build Tools with "Desktop development with C++"
  Download: `https://aka.ms/vs/17/release/vs_BuildTools.exe`
- CUDA Toolkit 12.1 — Download: `https://developer.nvidia.com/cuda-toolkit`
- `CUDA_HOME` environment variable set

**Windows (PowerShell):**
```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:PATH += ";$env:CUDA_HOME\bin"

cd D:\PycharmProjects
git clone --recurse-submodules https://github.com/graphdeco-inria/diff-gaussian-rasterization
cd diff-gaussian-rasterization
python setup.py install
```

**Linux / macOS:**
```bash
git clone --recurse-submodules https://github.com/graphdeco-inria/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

If `CUDA_HOME` is not set, find your CUDA install:
```powershell
# Windows — find nvcc.exe
Get-ChildItem "C:\","D:\" -Recurse -Filter "nvcc.exe" -ErrorAction SilentlyContinue | Select-Object FullName
```

---

## Updating the Registry After Training

The server loads `models/registry.json` **once on startup** into memory. Edits to the
file while the server is running are ignored. You must stop the server, edit the file,
then restart.

**Correct workflow:**

```powershell
# 1. Stop the server (Ctrl+C in the terminal running uvicorn, or:)
Get-Process python | Stop-Process

# 2. Edit models/registry.json
#    Set status → "ready", fill in ply_path and splat_path

# 3. Restart
uvicorn src.pipeline.server:app --reload --port 8000

# 4. Verify
Invoke-RestMethod http://localhost:8000/api/jobs
```

**Example registry entry after successful training:**

```json
"f0bdec2a7e62": {
  "job_id": "f0bdec2a7e62",
  "item_name": "vase",
  "status": "ready",
  "progress": 100,
  "message": "Training complete. Model ready to view.",
  "ply_path": "D:\\PycharmProjects\\monosplat\\work\\f0bdec2a7e62\\models\\gaussian\\f0bdec2a7e62.ply",
  "splat_path": "D:\\PycharmProjects\\monosplat\\work\\f0bdec2a7e62\\models\\gaussian\\f0bdec2a7e62.splat",
  "num_images": 124,
  "num_gaussians": 10000,
  "error": null
}
```

> **Note:** Use full absolute paths for `ply_path` and `splat_path` on Windows.
> The output files are always written to `work/<job_id>/models/gaussian/`.

---

## ⚠️ Input Data Requirements

This pipeline is **data-sensitive**. Most failures are caused by incorrect input videos,
not code issues.

### What Works ✅

- Real-world footage shot with a phone camera
- Slow complete orbit around the subject (60–80% frame overlap)
- Consistent exposure and lighting throughout
- One continuous uninterrupted clip
- Textured subjects (edges, patterns, surface detail)

Good examples: shoe, bottle, plant, statue, room, building exterior

### What Fails ❌

- Logo animations, motion graphics, screen recordings
- Videos with cuts, fades, or transitions
- Smooth, shiny, or transparent objects (plain walls, glass, metal spheres)
- Fast motion causing motion blur
- 2D content or rendered CGI

These fail because COLMAP cannot find consistent feature matches between frames.

### Common Error Symptoms

| Error | Cause | Fix |
|-------|-------|-----|
| "Could not register image" | Poor frame overlap | Walk slower, two full loops |
| "Discarding reconstruction" | Not enough valid frames | More angles, better lighting |
| Sparse / broken model | Textureless or inconsistent input | Change subject or environment |

---

## Capture Guide

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

### Angle Guidelines

**For an object (shoe, bottle, plant):**
- Slow complete circle at eye level
- Second loop slightly above
- Keep object centered and maintain consistent distance

**For a room or indoor space:**
- Walk slowly around the perimeter facing inward
- Avoid pointing at windows (overexposure)

**For architecture:**
- Walk parallel to the facade at consistent distance
- Arc around corners slowly

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

## Pipeline Stages in Detail

### Stage 1 — Frame Extraction (FFmpeg)

```
Video (MP4/MOV) → FFmpeg → work/<job_id>/frames/*.png
```

- Adaptive fps based on video duration (configurable, default null = auto)
- Hard cap at 600 frames (config default)
- Validates every extracted frame — corrupted files removed before COLMAP runs

### Stage 2 — Camera Pose Estimation (COLMAP SfM)

```
Frames → COLMAP → work/<job_id>/colmap/sparse_text/
                      ├── cameras.txt      # camera intrinsics
                      ├── images.txt       # camera extrinsics (poses)
                      └── points3D.txt     # sparse 3D point cloud
```

- GPU-accelerated SIFT keypoint detection
- Sequential matching with overlap=10 (much lighter than exhaustive)
- Triangulates 3D points and estimates real camera positions via Structure-from-Motion
- Exports text-format sparse model for use in Gaussian training

### Stage 3 — Gaussian Splat Training (PyTorch — GPU required)

```
Frames + Camera Poses → GaussianTrainer
                              → work/<job_id>/models/gaussian/<job_id>.ply
                              → work/<job_id>/models/gaussian/<job_id>.splat
```

- Initializes Gaussians from the COLMAP sparse point cloud
- Trains using L1 + SSIM photometric loss (lambda_dssim=0.2)
- Densifies (clone/split) and prunes Gaussians on a schedule
- Saves checkpoints every 5,000 iterations for resume support
- Exports `.ply` (archive) and `.splat` (browser-optimized, 32 bytes per splat)

### Stage 4 — Browser Viewer (Three.js)

```
<job_id>.splat → FastAPI → /viewer/<job_id> → Three.js real-time render
```

| Control | Action |
|---------|--------|
| Left drag | Rotate |
| Right drag | Pan |
| Scroll | Zoom |
| R | Reset camera |

---

## Configuration Reference

All parameters in `config/config.yaml`:

```yaml
data:
  video_fps: null        # null = adaptive (duration-based); set a number to override
  max_frames: 600        # hard cap on extracted frames

colmap:
  binary_path: "colmap"
  quality: "medium"      # low / medium / high
  camera_model: "OPENCV"
  single_camera: true

training:
  iterations: 30000
  save_every: 5000
  densify_from_iter: 500
  densify_until_iter: 15000
  densification_interval: 100
  densify_grad_threshold: 0.0002
  lambda_dssim: 0.2

renderer:
  background_color: [1.0, 1.0, 1.0]
  sh_degree: 3
  max_gaussians: 1000000
  use_cuda_rasterizer: true
  batch_size: 5000
```

---

## Output Formats

### .ply (standard Gaussian Splat archive)
Contains positions, SH colors, opacity, scale, rotation per Gaussian.
Compatible with SuperSplat, SIBR viewers, Luma AI.
Use for archiving, further processing, desktop apps.

### .splat (browser-optimized binary)
32 bytes per splat. Direct drag-and-drop into https://supersplat.playcanvas.com.
Served by the built-in Three.js viewer at `/viewer/<job_id>`.
Use for browser viewing, sharing, and demos.

---

## Known Issues and Fixes

### OOM crash during KNN initialization (GTX 1650 / 4 GB VRAM)
**Symptom:**
```
RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 25600000000 bytes
```
**Cause:** The original `_knn_mean_dist` computed a full 80,000×80,000 distance matrix
(25 GB) on CPU.

**Fix (applied in `scripts/train_local_gpu.py`):** Batched GPU kNN — processes 2,048
rows at a time (~640 MB peak), runs on CUDA instead of CPU:
```python
for start in range(0, N, 2048):
    chunk = pts[start:end]
    D = torch.cdist(chunk, pts)   # only 2048 rows at once
    ...
```

### CUDA not detected despite NVIDIA GPU being present
**Symptom:** `⚠ CUDA not available — running on CPU`

**Cause:** PyTorch was installed without CUDA support (the default `pip install torch`).

**Fix:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Note: cu121 wheels work with CUDA drivers 12.1–12.5+ (forward compatible).

### CUDA_HOME not set (rasterizer build fails)
**Symptom:** `OSError: CUDA_HOME environment variable is not set`

**Fix (Windows PowerShell):**
```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:PATH += ";$env:CUDA_HOME\bin"
python setup.py install
```

### loss.backward() crash — no grad_fn
**Symptom:**
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```
**Cause:** Software renderer returns a tensor with no gradient graph.
Downstream effect of CUDA not being available.
**Fix:** Fix the CUDA/PyTorch install first (see above).

### Registry edits not picked up by the UI
**Cause:** The server loads `models/registry.json` once at startup into memory.
**Fix:** Stop the server, edit the file, restart.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| FFmpeg not found | Not in PATH | Install FFmpeg and add to PATH |
| COLMAP not found | Not installed | Install COLMAP and add to PATH |
| COLMAP produces no model | Poor overlap or textureless input | Two full loops, locked exposure |
| Feature matching crash (exit 3221225786) | GPU VRAM exhaustion during exhaustive matching | Pipeline uses sequential_matcher by default |
| Blurry reconstruction | Motion blur | Lock exposure, walk slower |
| Training OOM | Too many Gaussians | Lower `max_gaussians` in config or script args |
| Colab times out | Long training run | Checkpoints every 5k iters — resume with `--resume` |
| Browser viewer blank | .splat not ready or registry not updated | Wait for `ready` status; check registry.json |
| UI shows `ready_for_colab` after training | Registry not updated / server not restarted | Stop server, edit registry.json, restart |
| Upload portal unreachable | Server not running | Run uvicorn command, check port 8000 |

---

## Performance Targets

| Stage | Target Time | Notes |
|-------|------------|-------|
| Frame extraction | ~30 sec | FFmpeg, up to 600 frames |
| COLMAP feature extraction | ~1 min | GPU-accelerated SIFT |
| COLMAP sequential matching | 1–3 min | overlap=10 |
| COLMAP sparse reconstruction | 2–5 min | Depends on scene complexity |
| Gaussian training — Colab T4 | 20–40 min | 15k–30k iterations |
| Gaussian training — GTX 1650 (no rasterizer) | 3–4 hrs | 7k iterations, software renderer |
| Gaussian training — GTX 1650 (with rasterizer) | 15–20 min | 7k iterations, CUDA rasterizer |
| Browser render | 30+ FPS | Three.js viewer |

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.9+ |
| Deep Learning | PyTorch | 2.0+ (cu121 build) |
| Frame Extraction | FFmpeg | 4.x+ (8.1 verified) |
| Pose Estimation (SfM) | COLMAP | 3.8+ (3.13 verified) |
| Gaussian Training | Custom PyTorch GaussianTrainer | — |
| Cloud Training | Google Colab (T4 GPU) | — |
| Web Server | FastAPI + Uvicorn | 0.104+ |
| Browser Viewer | Three.js (gaussian-splats-3d) | — |
| 3D Formats | PLY + .splat binary | — |
| Config | PyYAML | 6.0+ |

---

## Method Comparison

### Gaussian Splatting vs NeRF

| Property | NeRF | Gaussian Splatting |
|----------|------|--------------------|
| Scene representation | Implicit neural network | Explicit 3D Gaussian primitives |
| Render speed | Seconds per frame | Real-time (30–60+ FPS) |
| Training time | Hours | 15–30 minutes (GPU) |
| Editability | Difficult | Easy — Gaussians are explicit objects |
| Browser support | Requires server-side rendering | Native Three.js, zero install |

### Why Not Use Hardware Depth Sensors?

| Limitation | Impact |
|-----------|--------|
| High cost | Limits accessibility |
| Complex setup | Calibration and synchronization required |
| Reduced portability | Not suitable for casual mobile capture |
| Overkill for visual tasks | Unnecessary for photorealistic rendering |

---

## Why MonoSplat vs Other Pipelines

### vs. Raw 3DGS Reference Code

| Aspect | 3DGS Reference Code | MonoSplat |
|--------|---------------------|-----------|
| Usability | Research CLI-only | Full web upload portal + live job tracking |
| COLMAP integration | Manual | Automated sequential matching |
| Viewer | SIBR desktop app | Three.js browser, shareable URL |
| Job management | None | Async queue, SSE status, registry |

### vs. Commercial Apps (Luma AI, Polycam)

| Aspect | Commercial Apps | MonoSplat |
|--------|----------------|-----------|
| Cost | Subscription / per-scene | Free, open source |
| Privacy | Data uploaded to third party | Runs locally |
| Extensibility | Black box | Full source, customizable |
| Academic use | Not citable | Open, auditable, reproducible |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| PSNR | Image reconstruction quality (higher = better) |
| SSIM | Structural similarity (0–1, higher = better) |
| FPS | Viewer render performance |
| Training Time | End-to-end efficiency |
| Gaussians | Model complexity / detail level |

### Limitations

- Sensitive to motion blur and dynamic scenes
- Struggles with reflective and transparent surfaces
- Requires good camera coverage for complete reconstruction
- Every new scene requires a new training run (by design — the model IS the scene)

---

## Future Scope

- Dedicated GPU worker (Runpod / Lambda Labs) for production
- Job queue with Redis for multi-user support
- Cloud storage (S3 / Cloudflare R2) for persistent splat hosting
- NeRF vs Gaussian Splatting comparison viewer
- VR headset support (OpenXR)
- Multi-scene stitching
- Model compression (fewer splats, same quality)
- CUDA rasterizer auto-install on setup

---

## Deliverables

| Item | Status |
|------|--------|
| Full modular source code | ✅ done |
| FFmpeg frame extractor with image validation | ✅ done |
| COLMAP automated SfM pipeline (sequential matching) | ✅ done |
| Custom PyTorch Gaussian training with .splat export | ✅ done |
| FastAPI server with Three.js browser viewer | ✅ done |
| Web upload portal (single video, live job tracking) | ✅ done |
| Local GPU training script with VRAM auto-detection | ✅ done |
| Google Colab GPU training notebook | ✅ done |
| Unified YAML config | ✅ done |
| Pipeline documentation | ✅ done |
| **Stage 1: Capture quality warnings (blur, motion, exposure)** | ✅ done |
| **Stage 2: WebXR VR/AR mode** | ✅ done |
| **Stage 3: Progressive chunk loading** | ✅ done |
| **Stage 4: SPZ compression** | ✅ done |
| **Stage 5: Cloud storage (S3/GCS/Local)** | ✅ done |
| **Stage 6: Full XR features (measure, teleport, collab)** | ✅ done |
| **Stage 7: AI Layer (detection, segmentation, QA)** | ✅ done |

---

## References

- 3D Gaussian Splatting for Real-Time Radiance Field Rendering — Kerbl et al., SIGGRAPH 2023
  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- COLMAP — Structure-from-Motion and Multi-View Stereo
  https://colmap.github.io/
- gaussian-splats-3d — Three.js Gaussian Splat viewer
  https://github.com/mkkellogg/GaussianSplats3D
- SuperSplat — Browser-based splat viewer and editor
  https://supersplat.playcanvas.com
- diff-gaussian-rasterization — CUDA rasterizer
  https://github.com/graphdeco-inria/diff-gaussian-rasterization

---

## License

MIT License — free to use for educational and research purposes.
# MonoSplat

> Single-camera 3D Gaussian Splat reconstruction pipeline — record a video, use RealityScan for alignment, train with Lichtfeld on GPU, view the 3D splat in your browser.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)
![FFmpeg](https://img.shields.io/badge/FFmpeg-required-red)
![RealityScan](https://img.shields.io/badge/RealityScan-SfM-purple)
![Lichtfeld](https://img.shields.io/badge/Lichtfeld-training-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What Is This?

MonoSplat is an end-to-end pipeline that takes a single video input and reconstructs a photorealistic 3D Gaussian Splat scene you can navigate in your browser — no desktop app, no OpenGL setup required.

```
Video Input  →  Frame Extraction  →  Camera Poses      →  Gaussian Training  →  Browser Viewer
   MP4/MOV         FFmpeg            RealityScan (SfM)      Lichtfeld              Three.js
```

### Pipeline Stack

| Stage | Tool | Why |
|-------|------|-----|
| Frame extraction | **FFmpeg** | Fast, stable, broad codec support |
| Camera alignment (SfM) | **RealityScan** | Geometry-based, real camera poses from real captured video |
| Gaussian Splat training | **Lichtfeld** | GPU-accelerated, high-quality 3D Gaussian output |
| Browser viewer | Three.js | Zero-install, cross-platform, real-time |

This pipeline is:
- ✔ Geometry-based
- ✔ Uses real camera poses
- ✔ Works on real captured videos

---

## Problem Statement

Creating high-quality 3D reconstructions traditionally requires:
- Multi-camera rigs or depth sensors
- Complex photogrammetry workflows
- Heavy rendering pipelines that are not real-time

This creates a barrier for students, developers, and creators who want to explore spatial computing.

MonoSplat addresses this by enabling:
- Single-camera capture (phone video)
- Automated reconstruction pipeline via RealityScan SfM
- Real-time browser-based visualization via Lichtfeld + Three.js

The goal is to make 3D scene capture as simple as recording a video.

---

## How Gaussian Splatting Actually Works

**Gaussian Splat training is not like a normal machine learning model.** A conventional ML model trains once and runs inference on any new input. Gaussian Splatting works differently — the trained model *is* the scene. It overfits intentionally to one specific capture.

```
Video A  →  RealityScan  →  Lichtfeld train  →  scene_A.splat   (only represents scene A)
Video B  →  RealityScan  →  Lichtfeld train  →  scene_B.splat   (only represents scene B)
```

There is no shared weights file that generalizes to new scenes. Every new video requires a new training run. This is fundamental to the technology, not a limitation of this codebase.

From the user's perspective:
```
Upload video  →  "Processing…"  →  View 3D scene
```

The RealityScan alignment and Lichtfeld training happen invisibly in the background.

---

## Product Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER (Browser)                          │
│   Upload video  →  "Processing…"  →  View 3D scene          │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│            FastAPI Backend  (server.py)                     │
│  - Receives video upload                                    │
│  - Creates job in ModelRegistry (registry.json)             │
│  - Returns job_id, streams live status via SSE              │
│  - Serves .splat files for browser viewing                  │
│  - Runs frame extraction (FFmpeg) locally (CPU)             │
└───────────────────────────┬─────────────────────────────────┘
                            │ Job handoff
┌───────────────────────────▼─────────────────────────────────┐
│         RealityScan  (SfM Alignment — CPU/local)            │
│  - Reads extracted frames                                   │
│  - Estimates real camera positions and orientations         │
│  - Exports cameras.txt, images.txt, points3D.txt            │
│  - Geometry-based, works on real captured video             │
└───────────────────────────┬─────────────────────────────────┘
                            │ Pose handoff
┌───────────────────────────▼─────────────────────────────────┐
│         Lichtfeld GPU Worker  (Colab / Runpod / Lambda)     │
│  - Reads RealityScan poses + frames                         │
│  - Runs Gaussian Splat training (GPU, 20–40 min)            │
│  - Exports .ply + .splat                                    │
│  - Returns output to local machine                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│         Storage  (local models/gaussian/ or cloud)          │
│  - Stores .splat files per job                              │
│  - Served by FastAPI viewer endpoint                        │
└─────────────────────────────────────────────────────────────┘
```

### What Runs Where

| Stage | Tool | Runs On | Why |
|---|---|---|---|
| Frame extraction | FFmpeg | PyCharm / local | Fast, CPU is fine, ~30 sec |
| SfM pose estimation | RealityScan | PyCharm / local | One-time per scene, CPU works, 5–15 min |
| Gaussian Splat training | Lichtfeld | Colab T4 GPU | GPU required — 20–40 min on GPU |
| FastAPI server + viewer | FastAPI + Three.js | PyCharm / local | Serves files, no GPU needed |

### Recommended Developer Workflow

```
1.  Record phone video
2.  Start server locally (PyCharm) → upload via browser portal
3.  Server auto-runs frame extraction (FFmpeg, ~30 sec)
4.  RealityScan alignment runs locally (CPU, ~5–15 min)
5.  Upload project zip to Colab
6.  Lichtfeld trains on Colab T4 GPU (~30 min)
7.  Download .splat back to local machine
8.  Place .splat in models/gaussian/
9.  Server serves it at http://localhost:8000/viewer/<job_id>
10. Build downstream products using the .splat file
```

---

## Deployment Modes

### 1. Local Developer Mode (Current)
- Runs entirely on local machine (PyCharm)
- FFmpeg + RealityScan run locally; Lichtfeld runs on Colab
- Ideal for development, debugging, and research

### 2. Web Application Mode (Target)
- Fully automated pipeline
- No developer interaction required
- Real-time job tracking via web UI

### 3. Distributed System Mode (Production)
- Local server → job queue → cloud GPU workers → return .splat
- CPU stages run locally (FFmpeg + RealityScan)
- Lichtfeld training runs on cloud (Colab / Runpod / Lambda Labs)
- Scalable multi-user system with job scheduling

---

## Design Decisions

### Why RealityScan for SfM?
RealityScan provides geometry-based camera pose estimation using real captured video — no neural approximation. It produces accurate real camera positions and orientations that Lichtfeld's training loop requires for correct scene alignment.

### Why Lichtfeld for Training?
Lichtfeld is a GPU-accelerated Gaussian Splat training framework that produces high-quality `.ply` and `.splat` outputs optimized for real-time browser rendering. It integrates cleanly with RealityScan's pose output format.

### Why FFmpeg for Frame Extraction?
FFmpeg is significantly faster and more stable for video decoding and frame extraction, especially for longer videos and varied codecs (H.264, H.265, ProRes, etc.). No OpenCV dependency.

### Why Browser-Based Viewer?
A Three.js-based viewer allows zero installation, cross-platform access, and easy sharing.

### Why CPU Mode for Alignment by Default?
RealityScan's SfM pipeline works reliably on CPU for ≤200 frames and completes in 5–15 minutes — acceptable for a per-scene one-time cost. GPU acceleration is available for Colab environments.

---

## Design Decision: Software vs Hardware

### Approach Used

MonoSplat adopts a software-driven pipeline using single-camera RGB input (standard phone video):
- **FFmpeg** for frame extraction
- **RealityScan** for Structure-from-Motion (geometry-based, real camera poses)
- **Lichtfeld** for GPU Gaussian Splat training
- **Three.js** for browser rendering

### Why Not Use Hardware?

| Limitation | Impact |
|-----------|--------|
| High cost | Limits accessibility and scalability |
| Complex setup | Requires calibration and synchronization |
| Reduced portability | Not suitable for casual or mobile capture |
| Overkill for visual tasks | Unnecessary for photorealistic rendering |

### Trade-off Analysis

| Aspect | Software Pipeline (FFmpeg + RealityScan + Lichtfeld) | Hardware-Based Pipeline |
|--------|------------------------------------------------------|------------------------|
| Cost | Low (free tools) | High |
| Ease of Use | Simple | Complex |
| Accuracy | Moderate–High | Very High |
| Realism | High | High |
| Flexibility | High | Limited |

---

## ⚠️ Important: Input Data Requirements (Read Before Running)

This pipeline is **data-sensitive**. Most failures are caused by incorrect input videos — not code issues.

### What Works ✅

MonoSplat relies on geometry-based Structure-from-Motion (SfM) via RealityScan.
For successful reconstruction, your input video must contain:

- Real-world scenes (not animations)
- Consistent camera motion around the subject
- High texture (edges, patterns, details)
- Strong frame-to-frame overlap (60–80%)

Examples of good input:
- Walking around an object (shoe, bottle, statue)
- Drone orbit around a building
- Handheld scan of a person or room

---

### What Fails ❌

The following will almost always fail during SfM alignment:

- Logo animations or motion graphics
- 2D content / screen recordings
- Videos with cuts, fades, or transitions
- Smooth or textureless objects
- Fast motion causing motion blur

These fail because SfM cannot find enough consistent feature matches between frames.

---

### Why This Matters

Gaussian Splatting is **not a general ML model**.

```
Video → SfM (camera poses) → Gaussian Training → Scene-specific model
```

- The model is trained per scene
- It requires accurate camera poses
- If SfM fails → training will fail

---

### Common Error Symptoms

| Error | Cause |
|-------|-------|
| "Could not register image" | Poor feature overlap |
| "Discarding reconstruction" | Not enough valid frames |
| Sparse / broken model | Textureless or inconsistent input |

---

### Key Insight

> ✔ Good data + simple pipeline = SUCCESS
> ❌ Bad data + perfect pipeline = FAILURE

---

### Alternative (Advanced)

If working with synthetic or rendered content (e.g., Blender):

- SfM-based pipeline will not work
- You must use known camera poses instead

---

This is the #1 reason Gaussian Splat pipelines fail — choosing the wrong input.

---

## Capture Guide

This is the most critical section. The pipeline's SfM stage (COLMAP) relies entirely on finding consistent visual features across frames. Bad input video = failed reconstruction, regardless of how the code is configured.

---

### What Kind of Video Is Accepted

Your video must be:

- **Real-world footage** — not animation, screen recording, CGI, or motion graphics
- **Shot with a single fixed camera** — phone camera is perfect
- **Well-lit with consistent exposure** — no auto-exposure flickering between frames
- **One continuous uninterrupted shot** — no cuts, fades, or scene changes
- **Slow and steady movement** — fast motion causes blur which destroys SIFT features
- **Textured subject** — the scene must have visible edges, patterns, and surface detail

Things that will always fail: plain white walls, shiny metallic spheres, glass objects, TV screens, logos, smooth or textureless surfaces. These fail because COLMAP cannot find enough consistent feature matches between frames.

---

### At What Angle It Should Be Recorded

The golden rule is **60–80% overlap between consecutive frames**. COLMAP needs to see the subject from many overlapping angles to triangulate 3D points.

**For an object (shoe, bottle, plant, statue):**
- Walk in a slow complete circle around it
- Keep the object centered in frame the entire time
- Do at least 2 full loops — one at eye level, one slightly above
- Maintain consistent distance — do not zoom in or out

**For a room or indoor space:**
- Walk slowly around the perimeter facing inward
- Overlap every position with the previous one
- Avoid pointing directly at windows — overexposure destroys features

**For architecture or building exterior:**
- Walk parallel to the facade at a consistent distance
- Arc around corners slowly
- Do not back away suddenly or change elevation abruptly

**Angles that will fail:**
- Straight-on flat shot of a wall — no parallax, COLMAP cannot triangulate
- Top-down directly overhead — poor for SfM
- Random handheld walking with the camera swinging — too much blur and jump cuts

---

### Practical Phone Recording Tips

Since the pipeline extracts 200 frames at 5 fps, a **40-second video gives exactly 200 frames** — the ideal target length.

| Parameter | Recommendation |
|-----------|---------------|
| Duration | 35–45 seconds (≈ 200 frames at 5 fps) |
| Motion | Slow, smooth arc — roughly one step per second |
| Frame overlap | 60–80% between consecutive frames |
| Lighting | Consistent, diffuse — avoid hard shadows |
| Exposure | Lock exposure before recording (tap-hold on iPhone, Pro mode on Android) |
| Resolution | 1080p minimum |
| Subject framing | Fill 60–70% of the frame |

**Before you record:** tap and hold on your subject to lock focus and exposure. This prevents the camera from adjusting brightness mid-shot, which would cause inconsistent frame colors and confuse COLMAP's feature matching.

---

### Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Moving too fast | Motion blur, failed alignment | Walk at one step per second |
| Not enough angles | Holes in geometry, incomplete model | Two full loops minimum |
| Changing exposure | Inconsistent colors between frames | Lock exposure before recording |
| Background clutter | Noisy, unstable reconstruction | Use simple, static background |
| Subject too small in frame | Poor feature density, low detail | Fill 60–70% of frame |
| Textureless subject | No SIFT features to match | Choose subjects with visible surface detail |
| Video with cuts or fades | COLMAP cannot bridge the jump | Use one continuous uninterrupted clip |

---

## Setup

### Prerequisites

```bash
# FFmpeg (required for frame extraction)
# Ubuntu:  sudo apt install ffmpeg
# macOS:   brew install ffmpeg
# Windows: https://ffmpeg.org/download.html

# COLMAP (required for SfM pose estimation)
# Ubuntu:  sudo apt install colmap
# macOS:   brew install colmap
# Windows: https://github.com/colmap/colmap/releases

# Python dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Start the local server
uvicorn src.pipeline.server:app --reload --port 8000

# Open browser → http://localhost:8000
# Upload your video → pipeline runs automatically
```

---

## Pipeline Stages in Detail

### Stage 1 — Frame Extraction (FFmpeg)

```
Video (MP4/MOV) → FFmpeg → work/<job_id>/frames/*.png
```

- Extracts frames at 5 fps (configurable)
- Hard cap at 200 frames for tractable COLMAP alignment
- Validates every extracted frame — corrupted files are removed before COLMAP runs

---

### Stage 2 — Camera Pose Estimation (COLMAP SfM)

```
Frames → COLMAP → work/<job_id>/colmap/sparse_text/
                      ├── cameras.txt      # camera intrinsics
                      ├── images.txt       # camera extrinsics (poses)
                      └── points3D.txt     # sparse 3D point cloud
```

- Detects SIFT keypoints in every frame (GPU-accelerated)
- Matches features sequentially across adjacent frames (overlap=10)
- Triangulates 3D points and estimates real camera positions and orientations via Structure-from-Motion
- Exports text-format sparse model for use in Gaussian training

---

### Stage 3 — Gaussian Splat Training (PyTorch — GPU required)

```
Frames + Camera Poses → GaussianTrainer (Colab T4 GPU)
                              → work/<job_id>/models/gaussian/<job_id>.ply
                              → work/<job_id>/models/gaussian/<job_id>.splat
```

- On desktop (CPU): pipeline stops here with `ready_for_colab` status — zip the work folder and upload to Colab
- On Colab (GPU): initializes Gaussians from the COLMAP sparse point cloud, trains using L1 + SSIM photometric loss, densifies (clone/split) and prunes Gaussians throughout training
- Exports `.ply` (archive format) and `.splat` (browser-optimized binary, 32 bytes per splat)

---

### Stage 4 — Browser Viewer (Three.js)

```
<job_id>.splat → FastAPI → /viewer/<job_id> → Three.js real-time render
```

| Input | Action |
|-------|--------|
| Left mouse drag | Rotate |
| Right mouse drag | Pan |
| Scroll wheel | Zoom |
| R | Reset camera |

---

## Output Formats

### .ply (standard Gaussian Splat)
Compatible with SuperSplat, SIBR, Luma AI. Contains positions, SH colors, opacity, scale, rotation per Gaussian. Use for archiving, further processing, desktop apps.

### .splat (browser-optimized binary)
32 bytes per splat. Direct drag-and-drop into https://supersplat.playcanvas.com. Served by the built-in Three.js viewer at `/viewer/<job_id>`. Use for browser viewing, sharing, demos.

---

## Configuration Reference

All parameters in `config/config.yaml`:

```yaml
data:
  video_fps: 5           # Frames extracted per second
  max_frames: 200        # Hard cap on extracted frames

colmap:
  binary_path: "colmap"       # CLI executable name
  quality: "medium"           # low / medium / high
  camera_model: "OPENCV"
  single_camera: true

training:
  output_dir: "models/gaussian"
  checkpoint_dir: "models/checkpoints"
  iterations: 30000
  save_every: 5000
  densify_from_iter: 500
  densify_until_iter: 15000

renderer:
  max_gaussians: 3000000
  sh_degree: 3
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| FFmpeg not found | Not in PATH | Install FFmpeg and add to PATH |
| COLMAP not found | Not installed | Install COLMAP and add to PATH |
| COLMAP produces no model | Poor overlap or textureless input | Capture at 5 fps; walk two full loops |
| Feature matching crash (exit 3221225786) | GPU VRAM exhaustion during exhaustive matching | Pipeline uses sequential_matcher by default — check colmap_runner.py |
| Blurry reconstruction | Motion blur | Lock exposure; walk slower |
| Training OOM on Colab | Too many Gaussians | Lower `max_gaussians` in config |
| Colab times out | Long training | Checkpoints save every 5k iters — resume with `--resume` |
| Browser viewer blank | .splat not ready | Wait for job to reach `ready` in portal |
| Upload portal unreachable | Server not running | Run uvicorn command, check port 8000 |

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Frame extraction | ~30 sec | FFmpeg, up to 200 frames |
| COLMAP feature extraction | ~1 min | GPU-accelerated SIFT, 200 frames |
| COLMAP sequential matching | 1–3 min | overlap=10, far lighter than exhaustive |
| COLMAP sparse reconstruction | 2–5 min | Depends on scene complexity |
| Gaussian training (T4 GPU) | 20–40 min | 15k–30k iterations on Colab |
| Browser render FPS | 30+ FPS | Three.js viewer |
| .splat file size | Under 500 MB | 3M splats ~96 MB |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Deep Learning | PyTorch 2.0+ |
| Frame Extraction | FFmpeg |
| Pose Estimation (SfM) | COLMAP |
| Gaussian Training | Custom PyTorch (GaussianTrainer) |
| Cloud Training | Google Colab (T4 GPU) |
| Web Server | FastAPI + Uvicorn |
| Browser Viewer | Three.js (gaussian-splats-3d) |
| 3D Model Formats | PLY + .splat binary |
| Config | PyYAML |

---

## Method Comparison

### Gaussian Splatting vs NeRF

| Property | NeRF | Gaussian Splatting |
|----------|------|--------------------|
| Scene representation | Implicit neural network | Explicit 3D Gaussian primitives |
| Render speed | Seconds per frame | Real-time (20–60+ FPS) |
| Training time | Hours | 15–30 minutes |
| Editability | Difficult | Easy — Gaussians are explicit objects |

### SfM Tool Comparison

| Tool | Geometry-based | Real Camera Poses | Works on Video | Notes |
|------|---------------|-------------------|----------------|-------|
| **COLMAP** (used) | ✅ | ✅ | ✅ | Open source, battle-tested |
| RealityCapture | ✅ | ✅ | ✅ | Commercial, pay-per-use |
| NeRF-based pose estimation | ❌ | Approximated | ✅ | Neural, less accurate |

---

## Why MonoSplat Is Better Than Other Pipelines

Most Gaussian Splat pipelines make trade-offs that hurt accessibility, accuracy, or deployability. MonoSplat is designed to avoid all three.

### vs. NeRF-based Pipelines (Instant-NGP, Nerfstudio)

| Aspect | NeRF Pipelines | MonoSplat |
|--------|---------------|-----------|
| Render speed | Seconds per frame (offline) | Real-time 30+ FPS in browser |
| Training time | 1–6 hours | 20–40 minutes (GPU) |
| Scene representation | Implicit neural network | Explicit 3D Gaussians (editable) |
| Browser support | Requires heavy server-side rendering | Native Three.js viewer, zero install |
| Editability | Cannot edit scene geometry | Gaussians are explicit — can be pruned, filtered, exported |

NeRF produces a neural function, not a scene you can interact with. MonoSplat produces an explicit `.splat` file you can open, share, and embed anywhere.

---

### vs. Photogrammetry-only Pipelines (Meshroom, Metashape)

| Aspect | Photogrammetry Pipelines | MonoSplat |
|--------|-------------------------|-----------|
| Output | Dense mesh + texture | Real-time Gaussian Splat |
| Render quality | Baked texture, flat lighting | View-dependent color, photorealistic |
| Runtime viewer | Heavy desktop app required | Browser — no install |
| Pipeline automation | Manual multi-step workflow | Single video upload → browser view |
| Real-time FPS | Not real-time | 30+ FPS |

Meshes require UV unwrapping, texture baking, and a rendering engine. MonoSplat skips all of that and renders directly in the browser at real-time speeds.

---

### vs. Raw Gaussian Splat Implementations (3DGS reference code)

| Aspect | 3DGS Reference Code | MonoSplat |
|--------|---------------------|-----------|
| Usability | Research code, CLI-only | Full web upload portal + live job tracking |
| Deployment | Local only | Local + Colab GPU + cloud-ready architecture |
| COLMAP integration | Manual | Automated (sequential matching, zero manual steps) |
| Viewer | SIBR desktop viewer | Three.js browser viewer, shareable URL |
| Job management | None | Async job queue, SSE status streaming, registry |

The original 3DGS paper code requires expert setup. MonoSplat wraps the entire pipeline into a product: upload a video, get a shareable 3D scene.

---

### vs. Luma AI / Polycam (Commercial Apps)

| Aspect | Commercial Apps | MonoSplat |
|--------|----------------|-----------|
| Cost | Subscription / per-scene fees | Free (open source) |
| Control | Black box | Full source, customizable |
| Privacy | Data uploaded to third party | Runs locally on your machine |
| Extensibility | No | Plug in your own models, export formats, viewers |
| Academic use | Not citable | Fully open, auditable, reproducible |

Commercial tools are a black box. MonoSplat is fully open — every stage is inspectable, modifiable, and citable in academic work.

---

### Summary

MonoSplat is the only pipeline that combines:
- ✅ Geometry-based SfM (real camera poses, not neural approximations)
- ✅ Real-time browser viewer (no desktop app, no install)
- ✅ Automated end-to-end pipeline (video in → splat out)
- ✅ CPU/GPU split architecture (local preprocessing, cloud training)
- ✅ Fully open source and academically reproducible

---

## Evaluation

| Metric | Description |
|--------|-------------|
| PSNR | Image reconstruction quality |
| SSIM | Structural similarity |
| FPS | Viewer performance |
| Training Time | End-to-end efficiency |
| Completeness | Visual coverage of scene |

### Limitations

- Sensitive to motion blur and dynamic scenes
- Struggles with reflective and transparent surfaces
- Requires good camera coverage for complete reconstruction
- Every new scene requires a new training run (by design — the model IS the scene)

---

## Key Learnings

- Capture quality has a greater impact than model complexity
- COLMAP SfM alignment is the most failure-prone stage — good overlap is critical
- Gaussian Splatting enables real-time rendering unlike NeRF
- Data preprocessing and alignment are critical for training stability
- Per-scene training is not a limitation — the model encodes the scene, not general rules

---

## Future Scope

- Dedicated GPU worker (Runpod / Lambda Labs) for production training
- Job queue with Redis for concurrent multi-user support
- Cloud storage (S3 / Cloudflare R2) for persistent splat hosting
- NeRF vs Gaussian Splatting side-by-side comparison viewer
- VR headset support (OpenXR)
- Multi-scene stitching
- Model compression (fewer splats, same quality)
- AR overlay on live camera feed
- CUDA rasterizer upgrade (60+ FPS from current baseline)

---

## Deliverables

| Item | Status |
|------|--------|
| Full modular source code | done |
| FFmpeg frame extractor with image validation | done |
| COLMAP automated SfM pipeline (sequential matching) | done |
| Custom PyTorch Gaussian training with .splat export | done |
| FastAPI server with Three.js browser viewer | done |
| Web upload portal (single video, live job tracking) | done |
| Dynamic hot-reload pipeline | done |
| Google Colab GPU training notebook | done |
| Unified YAML config | done |
| Pipeline documentation | done |

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
- diff-gaussian-rasterization — CUDA rasterizer (drop-in upgrade)
  https://github.com/graphdeco-inria/diff-gaussian-rasterization

---

## License

MIT License — free to use for educational and research purposes.
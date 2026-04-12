# Pipeline Documentation

Pipeline: **FFmpeg → RealityScan (SfM) → Lichtfeld (GPU training)**

This pipeline is:
- ✔ Geometry-based
- ✔ Uses real camera poses
- ✔ Works on real captured videos

---

## Stage 1 — Data Acquisition

**Goal:** Collect multi-view images of the target scene.

**Best practices:**
- Capture 50–200 images / frames
- Walk in a smooth arc around the subject
- Maintain 60–80% overlap between consecutive frames
- Keep consistent exposure and focus

**Tools:** `src/preprocessing/extract_frames.py` (FFmpeg)

---

## Stage 2 — Frame Extraction (FFmpeg)

**Goal:** Extract individual frames from the captured video at a consistent rate.

FFmpeg is used over OpenCV for:
- Broader codec support (H.264, H.265, ProRes, HEVC, etc.)
- Faster extraction (hardware-accelerated on supported platforms)
- No OpenCV dependency

**Parameters:**
- Default: 5 fps extraction, hard cap at 200 frames
- Output: `work/<job_id>/frames/output_0001.png`, `output_0002.png`, …
- Automatic validation: corrupted frames are removed before RealityScan

**Tools:** `src/preprocessing/extract_frames.py`

---

## Stage 3 — Pose Estimation (RealityScan SfM)

**Goal:** Estimate the real camera position and orientation for every frame.

RealityScan uses geometry-based Structure-from-Motion (SfM) — no neural approximation.
It produces real camera poses from the actual geometry visible in the captured video.

RealityScan runs 4 internal stages:

1. **Feature Detection** — Keypoints detected in each frame
2. **Feature Matching** — Keypoints matched across all frame pairs
3. **Sparse Reconstruction (SfM)** — Triangulates 3D points, estimates real camera poses
4. **Model Export** — Exports COLMAP-compatible text format

**Output files:**
```
work/<job_id>/realityscan/
└── sparse_text/
    ├── cameras.txt      # Camera intrinsics (focal length, distortion)
    ├── images.txt       # Camera extrinsics (real poses per frame)
    └── points3D.txt     # Sparse 3D point cloud
```

**Note:** The output is COLMAP-compatible text format, so the downstream
Lichtfeld training step reads it directly without conversion.

**Tools:** `src/preprocessing/realityscan_runner.py`

**Install RealityCapture CLI:**
- Download: https://www.capturingreality.com/
- Windows: add `C:\Program Files\Capturing Reality\RealityCapture` to PATH
- Linux/macOS: set `realityscan.binary_path` in `config/config.yaml`

---

## Stage 4 — Gaussian Splat Training (Lichtfeld)

**Goal:** Optimize millions of 3D Gaussians to represent the scene.

Lichtfeld initializes Gaussians from the RealityScan sparse point cloud,
then trains using the real camera poses to match ground-truth frames.

Each Gaussian has 5 learnable properties:
- **Position** (x, y, z) — where it is in world space
- **Color** (SH coefficients) — view-dependent color
- **Opacity** — how transparent it is
- **Scale** (sx, sy, sz) — size along each axis
- **Rotation** (quaternion) — orientation

**Training loop:**
1. Pick a random training camera (from RealityScan poses)
2. Render the current Gaussians from that viewpoint
3. Compare to ground-truth frame (L1 + SSIM loss)
4. Backpropagate and update Gaussian parameters
5. Periodically: densify (add Gaussians) and prune (remove useless ones)

**Densification:**
- **Clone** Gaussians with large gradients but small size (under-reconstructed regions)
- **Split** Gaussians that are too large (over-reconstructed regions)
- **Prune** Gaussians that are too transparent

**Runs on:** Colab T4 GPU (~20–40 min). CPU fallback available for testing only.

**Tools:** `src/reconstruction/gaussian_model.py`, `src/reconstruction/trainer.py`

---

## Stage 5 — Interactive Viewer

**Goal:** Real-time rendering with WASD navigation in the browser.

**Render pipeline:**
1. Sort Gaussians by depth (back to front)
2. For each Gaussian: project 3D covariance to 2D screen ellipse
3. Alpha-composite Gaussians onto the canvas
4. Serve via FastAPI → Three.js browser viewer

**Tools:** `src/renderer/renderer.py`, `src/pipeline/server.py`

---

## Desktop → Colab Handoff

On a CPU machine (your desktop), the pipeline runs Stages 1–3 and stops
at `ready_for_colab` status. You then:

```
1. Run: python scripts/zip_for_colab.py <job_id>
2. Upload <job_id>_for_colab.zip to Colab
3. Run the Colab notebook — Lichtfeld trains on GPU
4. Download <job_id>.splat and <job_id>.ply
5. Place in work/<job_id>/models/gaussian/
6. Update models/registry.json: set status → "ready"
7. Open http://localhost:8000/viewer/<job_id>
```

The zip contains:
```
work/<job_id>/frames/                      ← FFmpeg extracted frames
work/<job_id>/realityscan/sparse_text/    ← RealityScan camera poses
config/config.yaml                         ← Lichtfeld training config
src/ + scripts/                            ← Lichtfeld training code
```

---

## Configuration Reference (`config/config.yaml`)

| Section          | Key                     | Default        | Description                              |
|------------------|-------------------------|----------------|------------------------------------------|
| `data`           | `video_fps`             | 5              | Frames/sec to extract from video         |
| `realityscan`    | `binary_path`           | RealityCapture | CLI executable name or full path         |
| `realityscan`    | `quality`               | medium         | SfM quality: low / medium / high         |
| `lichtfeld`      | `iterations`            | 30000          | Total Lichtfeld training steps           |
| `lichtfeld`      | `save_every`            | 5000           | Save checkpoint every N iters            |
| `lichtfeld.lr`   | `position`              | 0.00016        | Learning rate for Gaussian positions     |
| `lichtfeld`      | `densify_from_iter`     | 500            | Start densification at this step         |
| `lichtfeld`      | `densify_until_iter`    | 15000          | Stop densification at this step          |
| `renderer`       | `max_gaussians`         | 3,000,000      | Maximum Gaussian count                   |
| `viewer`         | `target_fps`            | 60             | Target render FPS                        |
| `viewer`         | `movement_speed`        | 0.05           | WASD movement speed                      |
| `viewer`         | `mouse_sensitivity`     | 0.003          | Mouse look sensitivity                   |
| `colab_overrides`| `iterations`            | 5000           | Override for Colab T4 (fits in 16 GB)   |
| `colab_overrides`| `max_gaussians`         | 500000         | Reduced for Colab VRAM                  |

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| FFmpeg not found | Not in PATH | Install FFmpeg and add to PATH |
| RealityScan not found | Binary not on PATH | Set `realityscan.binary_path` in config.yaml |
| RealityScan produces no model | Poor overlap | Capture at 5 fps; walk two full loops |
| Lichtfeld OOM on Colab | Too many Gaussians | Reduce `max_gaussians` in colab_overrides |
| Colab times out | Long training | Checkpoints save every 5k iters — resume with `--resume` |
| Browser viewer blank | .splat not ready | Wait for job to reach `ready` in portal |

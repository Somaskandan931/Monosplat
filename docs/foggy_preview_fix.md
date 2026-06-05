# Foggy Preview — Root Cause Analysis & Fix

## The Symptom

```
Before extent = 5.18
After extent  = 0.0468
cameras_extent = 0.1000
Preview at 500 iterations: solid grey fog
Training: 14+ hours, no improvement
```

---

## Root Cause Chain (3 linked bugs)

### Bug 1 — `normalize_scene.py`: Wrong scale denominator

**File:** `src/preprocessing/normalize_scene.py`

**Old code:**
```python
all_positions = np.concatenate([centers - centroid, point_xyz - centroid])
max_val = np.max(np.abs(all_positions))   # ← pulled in by COLMAP outliers
scale   = 1.0 / max_val
```

**Why it fails:**  
Your `points3D.txt` has 83,719 points. The max distance from centroid is **113.76 m** (P99 is 61.30 m, P50 is 4.35 m). COLMAP routinely creates a handful of extreme outlier points from mismatched feature tracks.

`max_val = 108.04` → `scale = 0.009256`

Camera centres that were at ~5.18 m radius got multiplied by 0.009256 → **0.048 m radius**.

**Fixed code:**
```python
camera_offsets = centers - centroid
camera_radius  = np.max(np.linalg.norm(camera_offsets, axis=1))
scale          = 1.0 / camera_radius   # cameras fill the unit sphere exactly
```

After fix: camera radius = 5.18 → scale = 0.193 → cameras_extent ≈ **1.0** ✓

---

### Bug 2 — `gaussian_model.py`: Scale clamp relative to wrong scene size

**File:** `src/reconstruction/gaussian_model.py` → `initialise_from_pcd()`

**Old code:**
```python
scales = torch.log(dist).unsqueeze(-1).repeat(1, 3)
scales = torch.clamp(scales, min=-4.0, max=0.0)   # max=0 → exp(0) = 1.0 world-unit
```

**Why it fails:**  
After Bug 1, the scene was compressed to 0.048 world-units wide.  
`exp(0) = 1.0 world-unit` = **2100% of scene diameter**.  
Every Gaussian is a giant translucent blob covering the entire scene → solid grey fog.

**Fixed code:**
```python
# max_log_scale = log(cameras_extent * 0.1)
# cameras_extent floored to 1.0 in train.py → max_log_scale = log(0.1) ≈ -2.3
# Each Gaussian starts at exp(-2.3) ≈ 0.1 world-units = 10% of scene radius
self._max_log_scale = math.log(max(spatial_lr_scale * 0.1, 1e-4))
scales = torch.clamp(scales, min=-4.0, max=self._max_log_scale)
```

---

### Bug 3 — `trainer.py`: `cameras_extent` too small → wrong prune threshold

**File:** `src/reconstruction/trainer.py` → `_maybe_densify()`

**Old code (pre-existing guard — correct):**
```python
extent = max(self.scene.cameras_extent, 1.0)
```

This guard existed but was **not enough on its own** because Bug 1 was feeding it `cameras_extent = 0.048` (before the 0.1 floor in train.py). After Bug 1 is fixed, this guard becomes a true safety net for any future normalization drift.

---

## Fix Summary

| File | Change | Effect |
|------|--------|--------|
| `normalize_scene.py` | Scale from camera radius, not max_abs of all points | `cameras_extent` goes from 0.048 → **~1.0** |
| `normalize_scene.py` | P99 outlier filter on point cloud before scaling | Removes 838 extreme COLMAP artifacts |
| `gaussian_model.py` | `max_log_scale = log(cameras_extent * 0.1)` | Initial Gaussians are 10% of scene, not 2100% |
| `gaussian_model.py` | Same ceiling applied in `_densify_and_split` | Cloned/split Gaussians obey same size limit |
| `trainer.py` | Added `[DENSIFY] before=/ after=` logging | Confirms densification is working |
| `trainer.py` | Preview every 250 iters (was 500) | Faster validation feedback |
| `trainer.py` | Checkpoint every 500 iters (was config-only) | Colab disconnect safety |
| `colab/train.py` | `MAX_INIT_GAUSSIANS = 20_000` (was 60_000) | Prevents Colab T4 OOM at init |

---

## What to Expect After Fix

```
cameras_extent (raw)    ≈ 0.8–1.2   (was 0.048)
cameras_extent (clamped)= 1.0
Initial Gaussians       : 20,000    (was 51,269)
Preview at iter 250     : recognizable scene structure
Preview at iter 500     : clear edges and colour
[DENSIFY] iter=600  before=20000  after=22431  delta=+2431
Training time (T4)      : 4–6 hrs  (was 14+)
```

---

## Verification Checklist

After applying fixes, check the training log for:

1. `[normalize_scene] camera_radius=X.XX  scale=Y.YY` — scale should be 0.1–0.5
2. `cameras_extent raw=0.9X → clamped=1.0` — raw should be close to 1.0 now
3. `[DENSIFY] iter=600 before=20000 after=22000+` — after must be > before
4. Preview at 250: not grey fog
5. Preview at 1000: recognizable structure

If `[DENSIFY] before=N after=N` (no change), `densify_and_prune` is not cloning — check `densify_grad_threshold` in `config.yaml` (should be 0.0005 for T4).

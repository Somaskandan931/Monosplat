"""
src/dataset/loader.py
---------------------
Load and serve training data: camera intrinsics, extrinsics, and images.

The dataset is built from COLMAP reconstruction output:
  - cameras.txt  → camera intrinsic parameters
  - images.txt   → camera poses (rotation + translation)
  - frames/      → the actual JPEG frames

Each call to __getitem__ returns one training view:
  {
    "image":      (3, H, W) float32 tensor [0, 1]
    "R":          (3, 3) rotation matrix (world→camera)
    "t":          (3,)   translation     (world→camera)
    "K":          (3, 3) intrinsic matrix
    "width":      int
    "height":     int
    "camera_id":  int
    "name":       str
  }

FIXES APPLIED:
  [FIX-1] _view_index_cache: O(1) lookup by view object (via id()).
          Trainer._compute_loss called self.scene.views.index(viewpoint)
          which is O(N) per training step — 100k iterations × 200 views
          means 20M list scans. Now resolved in O(1).
  [FIX-2] self.images is now a public attribute so train.py can inject
          normalized images (from normalize_scene) back into the dataset
          after construction, ensuring Camera.from_colmap in Trainer._render
          uses the corrected tvec values.
  [FIX-3] __getitem__ reads extrinsics from self.images (the authoritative
          normalized dict), not from the stale view object. This ensures
          training images and Trainer cameras share the same pose.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Optional

from src.utils.colmap_utils import (
    load_colmap_model, ColmapCamera, ColmapImage
)
from src.utils.image_utils import load_image_rgb


class ColmapDataset(Dataset):
    """
    PyTorch Dataset for COLMAP-registered frames.

    Args:
        frames_dir:   Directory containing extracted frames (*.jpg).
        sparse_dir:   COLMAP sparse model directory (contains cameras.txt etc.)
        image_width:  Resize width (0 = use COLMAP's native size).
        image_height: Resize height (0 = use COLMAP's native size).
    """

    def __init__(
        self,
        frames_dir:   str,
        sparse_dir:   str,
        image_width:  int = 0,
        image_height: int = 0,
    ):
        self.frames_dir   = Path(frames_dir)
        self.sparse_dir   = Path(sparse_dir)
        self.image_width  = image_width
        self.image_height = image_height

        # FIX-2: self.images is public — train.py replaces it with normalized images
        self.cameras, self.images, self.points3D = load_colmap_model(str(sparse_dir))

        # Build list of valid views (registered AND frame file exists on disk)
        self.views: List[ColmapImage] = []
        for img_data in self.images.values():
            frame_path = self.frames_dir / img_data.name
            if frame_path.exists():
                self.views.append(img_data)
            else:
                # Try finding by stem if extension differs
                stem    = Path(img_data.name).stem
                matches = list(self.frames_dir.glob(f"{stem}.*"))
                if matches:
                    img_data.name = matches[0].name
                    self.views.append(img_data)

        self.views.sort(key=lambda x: x.image_id)

        print(f"✅ Dataset: {len(self.views)} training views loaded")

        if len(self.views) == 0:
            raise RuntimeError(
                "No matching frames found!\n"
                f"  COLMAP images dir used: {frames_dir}\n"
                f"  Frames directory: {self.frames_dir}\n"
                "Make sure frame filenames match COLMAP's images.txt names."
            )

        # FIX-1: build O(1) view → index cache
        self._view_index_cache: Dict[int, int] = {
            id(v): i for i, v in enumerate(self.views)
        }

        # Output resolution
        sample_cam    = self.cameras[self.views[0].camera_id]
        self.width    = image_width  if image_width  > 0 else sample_cam.width
        self.height   = image_height if image_height > 0 else sample_cam.height

    # ------------------------------------------------------------------
    # O(1) view → index (FIX-1)
    # ------------------------------------------------------------------

    def view_index(self, view: ColmapImage) -> int:
        """Return the integer index of *view* in self.views in O(1)."""
        idx = self._view_index_cache.get(id(view))
        if idx is None:
            # Fallback: slow search (only if view replaced externally)
            idx = self.views.index(view)
        return idx

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> dict:
        """Return one training view as a dict of tensors."""
        view = self.views[idx]
        cam  = self.cameras[view.camera_id]

        # FIX-3: read from self.images (normalized) not the raw view object
        live_view = self.images.get(view.image_id, view)

        # ── Load image ────────────────────────────────────────────────────────
        img_path = self.frames_dir / view.name
        image_np = load_image_rgb(str(img_path), width=self.width, height=self.height)
        image    = torch.from_numpy(image_np).permute(2, 0, 1)   # (3, H, W)

        # ── Camera intrinsics → 3×3 matrix ───────────────────────────────────
        scale_x = self.width  / cam.width  if self.width  > 0 else 1.0
        scale_y = self.height / cam.height if self.height > 0 else 1.0

        K = torch.tensor([
            [cam.fx * scale_x,            0,  cam.cx * scale_x],
            [           0,     cam.fy * scale_y,  cam.cy * scale_y],
            [           0,                0,                1       ],
        ], dtype=torch.float32)

        # ── Camera extrinsics (from live/normalized image) ────────────────────
        R = torch.from_numpy(live_view.rotation_matrix()).float()  # (3, 3)
        t = torch.from_numpy(live_view.tvec).float()               # (3,)

        return {
            "image":     image,
            "R":         R,
            "t":         t,
            "K":         K,
            "width":     self.width,
            "height":    self.height,
            "camera_id": view.camera_id,
            "name":      view.name,
        }

    def get_all_camera_centers(self) -> np.ndarray:
        """Return all camera centre positions in world space. Shape: (N, 3)."""
        # FIX-3: use live (possibly normalized) images
        return np.array(
            [self.images[v.image_id].camera_center() for v in self.views],
            dtype=np.float32,
        )

    def get_train_cameras(self) -> list:
        """Return list of training view objects. Called by Trainer."""
        return self.views
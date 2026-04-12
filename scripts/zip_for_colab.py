"""
scripts/zip_for_colab.py
Helper script to zip a completed COLMAP job for upload to Colab.

Usage:
    python scripts/zip_for_colab.py <job_id>

Example:
    python scripts/zip_for_colab.py e5dea152323b
"""

import argparse
import zipfile
from pathlib import Path

def zip_job_for_colab(job_id: str, work_dir: str = "work") -> str:
    """Zip frames and colmap directories for a job ID."""
    work_path = Path(work_dir)
    job_path = work_path / job_id

    frames_dir = job_path / "frames"
    colmap_dir = job_path / "colmap"

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")

    zip_name = f"{job_id}_for_colab.zip"

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add frames
        for f in frames_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f"work/{job_id}/frames/{f.name}")

        # Add COLMAP output (sparse_text only — the three text files)
        sparse_text = colmap_dir / "sparse_text"
        if sparse_text.exists():
            for f in sparse_text.rglob("*"):
                if f.is_file():
                    zf.write(f, f"work/{job_id}/colmap/sparse_text/{f.name}")

        # Add config
        config_path = Path("config/config.yaml")
        if config_path.exists():
            zf.write(config_path, "config/config.yaml")

        # Add src/ and scripts/ so Colab can run scripts/train.py
        # without needing to upload the full project separately.
        for folder in ["src", "scripts"]:
            folder_path = Path(folder)
            if folder_path.exists():
                for f in folder_path.rglob("*"):
                    if f.is_file() and "__pycache__" not in str(f):
                        zf.write(f, str(f))

    print(f"✅ Created {zip_name}")
    print(f"   Size: {Path(zip_name).stat().st_size / 1e6:.1f} MB")
    print(f"\nUpload this file to Colab and run the notebook.")
    return zip_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip COLMAP output for Colab upload")
    parser.add_argument("job_id", help="Job ID from models/registry.json")
    parser.add_argument("--work_dir", default="work", help="Work directory (default: work)")
    args = parser.parse_args()

    zip_job_for_colab(args.job_id, args.work_dir)
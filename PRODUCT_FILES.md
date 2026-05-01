# MonoSplat Product Submission Files

Keep these files for the working product submission:

- `README.md` - product explanation, setup, capture guide, and architecture.
- `requirements.txt` - Python dependencies.
- `config/config.yaml` - pipeline configuration.
- `src/` - FastAPI server, preprocessing, COLMAP runner, training, renderer, and utilities.
- `web/index.html` - browser upload portal and job dashboard.
- `scripts/zip_for_colab.py` - packages a COLMAP-ready job for GPU training.
- `scripts/train.py` - standalone GPU training entry point for Colab/Kaggle/local CUDA.
- `scripts/smoke_test.py` - verifies product imports and server routes.
- `scripts/start_server.ps1` - starts the local FastAPI product server on Windows.
- `notebooks/monosplat_colab_gpu.ipynb` - recommended GPU training notebook.

Optional during development, not required in the submission package:

- `notebooks/monosplat_kaggle.ipynb`
- `outputs/`, `uploads/`, `work/`, and generated model files
- `__pycache__/`, `.idea/`, and zip backups

Run check:

```powershell
python scripts/smoke_test.py
```

Run product:

```powershell
.\scripts\start_server.ps1
```

Then open `http://127.0.0.1:8000`.

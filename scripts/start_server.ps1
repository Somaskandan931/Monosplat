$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
python -m uvicorn src.pipeline.server:app --host 127.0.0.1 --port 8000 --reload

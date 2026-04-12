#!/bin/bash
# =============================================================================
# start_dynamic.sh — start the full Dynamic Gaussian XR system
#
# Usage:
#   bash scripts/start_dynamic.sh
# =============================================================================

set -euo pipefail
GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║       Gaussian XR — Dynamic Store System         ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Upload Portal  →  http://localhost:8000         ║"
echo "║                                                  ║"
echo "║  HOW TO USE:                                     ║"
echo "║  1. Open http://localhost:8000 in your browser   ║"
echo "║  2. Enter a product name                         ║"
echo "║  3. Upload 30-60 photos or a short video         ║"
echo "║  4. Wait for the 3D model to appear              ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# Ensure required directories exist
mkdir -p uploads models/gaussian models/checkpoints work outputs/renders outputs/logs

# Start the 3D viewer in a separate terminal window (best-effort)
echo -e "${GREEN}[1/2] Starting dynamic 3D viewer…${NC}"
if command -v gnome-terminal &>/dev/null; then
    gnome-terminal -- bash -c "python -m src.pipeline.dynamic_viewer; exec bash" &
elif command -v xterm &>/dev/null; then
    xterm -title "Gaussian XR Viewer" -e "python -m src.pipeline.dynamic_viewer" &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e "tell app \"Terminal\" to do script \"cd $(pwd) && python -m src.pipeline.dynamic_viewer\"" &
else
    # Windows Git Bash / WSL fallback: run in background
    python -m src.pipeline.dynamic_viewer &
fi

sleep 2

# Start the web server (blocks — main process)
echo -e "${GREEN}[2/2] Starting web portal at http://localhost:8000 …${NC}"
uvicorn src.pipeline.server:app --host 0.0.0.0 --port 8000 --reload
#!/bin/bash
# ════════════════════════════════════════════
#  TextileVision AI — Quick Start Script
# ════════════════════════════════════════════

set -e
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ████████╗███████╗██╗  ██╗████████╗██╗██╗     ███████╗"
echo "     ██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝██║██║     ██╔════╝"
echo "     ██║   █████╗   ╚███╔╝    ██║   ██║██║     █████╗  "
echo "     ██║   ██╔══╝   ██╔██╗    ██║   ██║██║     ██╔══╝  "
echo "     ██║   ███████╗██╔╝ ██╗   ██║   ██║███████╗███████╗"
echo "     ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚══════╝╚══════╝"
echo -e "${NC}"
echo -e "${GREEN}  TextileVision AI — Quality Inspection System${NC}"
echo ""

# Navigate to backend
cd "$(dirname "$0")/backend"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo -e "${YELLOW}Python3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Install dependencies
echo -e "${CYAN}[1/3] Installing dependencies...${NC}"
pip install -r requirements.txt --quiet

# Check for camera
echo -e "${CYAN}[2/3] Checking camera...${NC}"
if ls /dev/video* &>/dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Camera device found${NC}"
else
    echo -e "${YELLOW}  ⚠ No camera found — Demo mode will activate${NC}"
fi

# Start server
echo -e "${CYAN}[3/3] Starting TextileVision AI...${NC}"
echo ""
echo -e "${GREEN}  Dashboard: http://localhost:8000${NC}"
echo -e "${GREEN}  API Docs:  http://localhost:8000/docs${NC}"
echo -e "${GREEN}  Stats API: http://localhost:8000/api/stats${NC}"
echo ""
echo -e "  Press Ctrl+C to stop"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

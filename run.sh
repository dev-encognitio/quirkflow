#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure common tool locations are in PATH
export PATH="/home/sandbox/.local/bin:/home/sandbox/tools/bin:$PATH"

# Ensure workspace root is in PYTHONPATH
export PYTHONPATH="/home/sandbox/ContentCreation:${PYTHONPATH:-}"

export STUDIO_HOST="${STUDIO_HOST:-127.0.0.1}"
export STUDIO_PORT="${STUDIO_PORT:-8400}"

echo "=========================================================="
echo " Starting Autonomous Video Studio on http://${STUDIO_HOST}:${STUDIO_PORT}"
echo "=========================================================="

exec uv run "${SCRIPT_DIR}/server.py"

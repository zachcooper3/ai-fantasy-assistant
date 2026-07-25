#!/usr/bin/env bash
# Runs the backend (uvicorn --reload) and frontend (next dev) together in one
# terminal, so you don't need to juggle two windows every session. Ctrl+C
# stops both cleanly.
#
# Usage (from the repo root, Git Bash on Windows or any bash on Mac/Linux):
#   ./dev.sh
#
# If you'd rather see backend and frontend logs in separate windows (e.g.
# for easier debugging), just run the two commands from the README manually
# instead — this script is a convenience, not a requirement.

set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "No venv/ found — run the first-time setup in README.md first:"
  echo "  py -m venv venv"
  exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
  echo "frontend/node_modules not found — run 'npm install' in frontend/ first."
  exit 1
fi

echo "Starting backend (uvicorn --reload) on http://localhost:8000 ..."
source venv/Scripts/activate
uvicorn backend.app.main:app --reload &
BACKEND_PID=$!

echo "Starting frontend (next dev) on http://localhost:3000 ..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

cleanup() {
  echo ""
  echo "Stopping backend and frontend ..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

wait

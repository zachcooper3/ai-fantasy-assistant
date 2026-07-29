#!/usr/bin/env bash
# Runs the backend (uvicorn) and frontend (next dev) together in one
# terminal, so you don't need to juggle two windows every session. Ctrl+C
# stops both cleanly.
#
# Usage (from the repo root, Git Bash on Windows or any bash on Mac/Linux):
#   ./dev.sh              # no auto-reload (default — see below)
#   DEV_RELOAD=1 ./dev.sh # auto-reload backend on code changes, for active development
#
# --reload defaults OFF, not on, as of 2026-07-28. Confirmed live: on
# Windows, uvicorn's --reload runs the app inside a child process spawned
# by its reload supervisor, and the first real ChromaDB query (the first
# "Get pick" of a session) makes ChromaDB's embedding backend try to spawn
# its own worker — a grandchild process attaching across two layers of
# supervision, which crashed with a Windows multiprocessing WinError 87
# mid-draft. Since this app's actual purpose is to run live *during* a
# draft (where you're not editing code anyway), --reload's only real value
# is while actively developing — hence opt-in via DEV_RELOAD=1, not the
# default. If you hit the same crash with DEV_RELOAD=1 set, that confirms
# it's --reload; drop it for that session.
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

if [ -n "$DEV_RELOAD" ]; then
  echo "Starting backend (uvicorn --reload, DEV_RELOAD=1) on http://localhost:8000 ..."
  RELOAD_FLAG="--reload"
else
  echo "Starting backend (no auto-reload — set DEV_RELOAD=1 to enable) on http://localhost:8000 ..."
  RELOAD_FLAG=""
fi
source venv/Scripts/activate
uvicorn backend.app.main:app $RELOAD_FLAG &
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

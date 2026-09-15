#!/usr/bin/env bash
set -euo pipefail

mode="${1:-docker}"

case "$mode" in
  backend)
    exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  frontend)
    cd frontend
    exec npm run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT:-5173}"
    ;;
  docker)
    exec docker compose up --build
    ;;
  *)
    echo "Usage: ./start.sh [backend|frontend|docker]" >&2
    exit 2
    ;;
esac

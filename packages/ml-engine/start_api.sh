#!/usr/bin/env bash
# Run from packages/ml-engine/
# Usage: ./start_api.sh
# Then open: http://localhost:8000/docs

set -e

cd "$(dirname "$0")"

echo "Starting NeuroForge API..."
echo "Docs: http://localhost:8000/docs"
echo ""

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
#!/usr/bin/env bash
set -e

echo "=== NeuroForge Day 8 Exit Gate ==="

# 1. Fast tests only (skip slow integration tests that train models)
echo ""
echo "→ Running Day 8 fast tests..."
python -m pytest tests/test_day8_api.py -v --tb=short -m "not slow"

# 2. Count fast tests
FAST_COUNT=$(python -m pytest tests/test_day8_api.py -m "not slow" --collect-only -q 2>&1 | grep "::" | wc -l | tr -d ' ')
echo "→ Fast test count: $FAST_COUNT (expect ~20)"

# 3. Full suite (all days, still no slow)
echo ""
echo "→ Running full suite (no slow)..."
python -m pytest --tb=short -q -m "not slow"

# 4. Smoke test: server starts and /health responds
echo ""
echo "→ Smoke testing /health endpoint..."
uvicorn api.main:app --host 127.0.0.1 --port 8765 &
SERVER_PID=$!
sleep 2

STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/health)
kill $SERVER_PID 2>/dev/null

if [ "$STATUS" = "200" ]; then
    echo "  ✅ /health returned 200"
else
    echo "  ❌ /health returned $STATUS"
    exit 1
fi

# 5. Optionally run slow tests (trains a real model)
echo ""
echo "→ Running slow integration tests (trains a model — ~10s)..."
python -m pytest tests/test_day8_api.py -v --tb=short -m "slow"

echo ""
echo "=== Day 8 complete ✅ ==="
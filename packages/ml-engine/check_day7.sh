#!/usr/bin/env bash
set -e

echo "=== NeuroForge Day 7 Exit Gate ==="

# 1. Run tests
echo ""
echo "→ Running pytest (Day 7 tests)..."
python -m pytest tests/test_day7_experiment_tracking.py -v --tb=short

# 2. Count new tests

NEW_TESTS=$(python -m pytest tests/test_day7_experiment_tracking.py --collect-only -q 2>&1 | grep "::" | wc -l | tr -d ' ')
echo ""
echo "→ Day 7 test count: $NEW_TESTS (expect ~22)"

# 3. Full suite must still be green
echo ""
echo "→ Running full test suite..."
python -m pytest --tb=short -q

# 4. Run the demo script
echo ""
echo "→ Running compare_experiments.py..."
python compare_experiments.py

# 5. Verify artifacts on disk
echo ""
echo "→ Checking experiment files on disk..."
SNAP_COUNT=$(find experiments_demo/runs -name "snapshot.json" 2>/dev/null | wc -l)
echo "   snapshot.json files found: $SNAP_COUNT (expect 2)"

if [ "$SNAP_COUNT" -eq 2 ]; then
    echo "   ✅ Persistence verified"
else
    echo "   ❌ Expected 2 snapshot files, got $SNAP_COUNT"
    exit 1
fi

echo ""
echo "=== Day 7 complete ✅ ==="
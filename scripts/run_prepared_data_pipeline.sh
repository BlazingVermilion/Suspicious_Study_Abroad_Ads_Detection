#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python}"
"$PYTHON" scripts/run_prepared_data_pipeline.py "$@"

#!/usr/bin/env bash
# Helper script to invoke the NISAR L-band cloud worker
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHONPATH="${REPO_ROOT}/backend" "${REPO_ROOT}/.venv/bin/python3" "${REPO_ROOT}/cloud-worker/nisar/worker.py" "$@"

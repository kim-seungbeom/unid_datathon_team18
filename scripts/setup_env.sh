#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

case "${1:---help}" in
  --local)
    PROJECT_PYTHON="${PROJECT_PYTHON:-python3.12}"
    if [ ! -x .venv/bin/python ]; then
      "$PROJECT_PYTHON" -m venv .venv
    fi
    .venv/bin/python -m pip install -r requirements-dev.txt
    .venv/bin/python scripts/check_environment.py
    echo 'Local setup complete. Activate with: source .venv/bin/activate'
    ;;
  --server)
    PROJECT_PYTHON="${PROJECT_PYTHON:-/home/ubuntu/venv/bin/python}"
    PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-/home/ubuntu/data/battery_data_light}"
    PROJECT_BASELINE_ROOT="${PROJECT_BASELINE_ROOT:-/home/ubuntu/data/baseline}"
    if [ ! -x "$PROJECT_PYTHON" ]; then
      echo 'Organizer Python environment not found.' >&2
      echo 'Ask whether a teammate finished the official setup.sh before running it yourself.' >&2
      exit 1
    fi
    PROJECT_LAYOUT_READY=1
    for PROJECT_SPLIT in train validation test; do
      if [ ! -f "$PROJECT_DATA_ROOT/$PROJECT_SPLIT/$PROJECT_SPLIT.csv" ] || \
         [ ! -d "$PROJECT_DATA_ROOT/$PROJECT_SPLIT/images" ]; then
        echo "Missing expected CSV or images directory for split: $PROJECT_SPLIT" >&2
        PROJECT_LAYOUT_READY=0
      fi
    done
    if [ "$PROJECT_LAYOUT_READY" -ne 1 ]; then
      echo 'Check the official setup.sh status with your team. No data was downloaded or opened.' >&2
      exit 1
    fi
    echo 'Expected shared data paths are present (contents not checked).'
    # The organizer archive can have different layouts; the team pipeline does not import it.
    if [ -d "$PROJECT_BASELINE_ROOT" ]; then
      echo 'Optional organizer baseline directory is present (contents not checked).'
    else
      echo 'Optional organizer baseline directory not found; it is not required by the team pipeline.'
    fi
    "$PROJECT_PYTHON" scripts/check_environment.py --require_cuda
    echo 'Server check only: no packages were installed or changed.'
    ;;
  *)
    echo 'Usage: bash scripts/setup_env.sh --local | --server'
    echo '--local: install into the repository .venv; synthetic CPU work only.'
    echo '--server: check shared paths, organizer venv, and CUDA without changing packages.'
    ;;
esac

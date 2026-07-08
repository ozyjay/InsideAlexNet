#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"

echo "Setting up How Does a Neural Network See?"
echo "Project: ${PROJECT_ROOT}"
echo "Virtual environment: ${VENV_DIR}"

if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${VENV_DIR}" ]]; then
  echo "Warning: another virtual environment is active: ${VIRTUAL_ENV}"
  echo "Deactivate it first if this is not intentional. Continuing with project .venv."
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating local virtual environment..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Using existing local virtual environment."
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Error: virtual environment Python was not found at ${VENV_PYTHON}" >&2
  exit 1
fi

# Safety check: all installs below must go into the project venv, not global/user site-packages.
PIP_PREFIX="$(${VENV_PYTHON} -m pip --version)"
echo "Using pip: ${PIP_PREFIX}"

case "${PIP_PREFIX}" in
  *"${VENV_DIR}"*) ;;
  *)
    echo "Error: pip does not appear to point inside ${VENV_DIR}. Aborting to avoid global installs." >&2
    exit 1
    ;;
esac

echo "Upgrading pip inside the virtual environment..."
"${VENV_PYTHON}" -m pip install --upgrade pip

echo "Installing project requirements inside .venv..."
"${VENV_PYTHON}" -m pip install -r requirements.txt

echo "Running local setup check..."
"${VENV_PYTHON}" scripts/check_setup.py

echo
cat <<MSG
Setup complete.

Activate the environment with:
  source .venv/bin/activate

Run the demo with:
  bash scripts/run_dev.sh

Or explicitly:
  .venv/bin/uvicorn app:app --host 127.0.0.1 --port 3450
MSG

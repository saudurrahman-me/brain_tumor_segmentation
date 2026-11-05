#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Environment created. Activate it with: source .venv/bin/activate"
echo "On Windows, create and activate the venv in PowerShell, then install requirements.txt."

#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up Apple Silicon development environment..."

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "❌ This setup script must run on Apple Silicon (arm64)."
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "Installing TA-Lib via Homebrew..."
brew install ta-lib

export TA_INCLUDE_PATH=/opt/homebrew/include
export TA_LIBRARY_PATH=/opt/homebrew/lib
export ARCHFLAGS="-arch arm64"
export OPENBLAS=/opt/homebrew/opt/openblas

cd "${PROJECT_ROOT}"

echo "Removing existing virtual environment (if present)..."
rm -rf .venv

# Use Homebrew Python 3.13 which satisfies hftbacktest>=2.0.0 (requires Python >=3.10)
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.13}"

echo "Creating virtual environment with ${PYTHON_BIN}..."
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

echo "Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing data-collector dependencies..."
pip install -r requirements.txt

echo "Installing signal-engine in editable mode..."
pushd signal-engine >/dev/null
pip install -e .
popd >/dev/null

echo "Verifying key imports..."
python - <<'PYCODE'
import numpy
import pandas
import talib
print("All critical imports succeeded.")
PYCODE

echo ""
echo "✅ Apple Silicon setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest signal-engine/tests/unit/ -v"
echo "  pytest tests/ -v"

#!/usr/bin/env bash
set -euo pipefail

# Optional module setup (for HPC environments)
if command -v module >/dev/null 2>&1; then
  if [ "${USE_MODULES:-0}" = "1" ]; then
    module purge
    module load BaseGPU/2025 || true
    module load BasePy/2025 || true
    module load python3/3.11 || true
  fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/torch-env"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (customize via TORCH_SPEC or TORCH_INDEX_URL)
# Examples:
#   TORCH_SPEC="torch==2.2.2 torchvision==0.17.2" ./setup_torch_env.sh
#   TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ./setup_torch_env.sh
if [ -n "${TORCH_SPEC:-}" ]; then
  python -m pip install ${TORCH_SPEC}
else
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
  else
    # default to CPU wheels
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# Core scientific stack
python -m pip install numpy pandas matplotlib joblib scipy scikit-learn tqdm pyyaml

# Install gmlp_project in editable mode if present
if [ -d "${REPO_ROOT}/gmlp_project" ]; then
  python -m pip install -e "${REPO_ROOT}/gmlp_project"
fi

echo "✅ torch-env ready. Activate with: source ${VENV_DIR}/bin/activate"

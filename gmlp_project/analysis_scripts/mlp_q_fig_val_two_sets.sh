#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=8
#PBS -l gpunum_job=0
#PBS -o ~/q2_dual_analysis_log_val.out
#PBS -e ~/q2_dual_analysis_log_val.err
#PBS -r n
#PBS -V

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONO_ROOT=""

# 1) Honor REPO_ROOT if provided (can point to repo root or gmlp_project).
if [ -n "${REPO_ROOT:-}" ]; then
  if [ -d "$REPO_ROOT/.git" ] || [ -d "$REPO_ROOT/gmlp_project" ]; then
    MONO_ROOT="$REPO_ROOT"
  elif [ -d "$REPO_ROOT/scripts" ] && [ -d "$REPO_ROOT/data" ]; then
    MONO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
  fi
fi

# 2) Prefer submit directory if available (PBS copies this script into a jobfile dir).
if [ -z "$MONO_ROOT" ]; then
  START_DIR="${PBS_O_WORKDIR:-$SCRIPT_DIR}"
  dir="$START_DIR"
  while [ "$dir" != "/" ]; do
    if [ -d "$dir/.git" ] || [ -d "$dir/gmlp_project" ]; then
      MONO_ROOT="$dir"
      break
    fi
    if [ -d "$dir/scripts" ] && [ -d "$dir/data" ]; then
      MONO_ROOT="$(cd "$dir/.." && pwd)"
      break
    fi
    dir="$(dirname "$dir")"
  done
fi

# 3) Fallback to script dir (jobfile dir); warn if repo not found.
if [ -z "$MONO_ROOT" ]; then
  MONO_ROOT="$SCRIPT_DIR"
fi

if [ ! -d "$MONO_ROOT/gmlp_project" ]; then
  echo "❌ Repo root not found. MONO_ROOT=$MONO_ROOT"
  echo "   Set REPO_ROOT to the repository root (the directory containing gmlp_project)."
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$MONO_ROOT}"
export PYTHONPATH="${MONO_ROOT}:${PYTHONPATH:-}"
echo "🚀 Starting MLP dual q_inv figure job (VALIDATION)"
echo "Job ID: $PBS_JOBID"
echo "Host: $(hostname)"
echo "Time: $(date)"

module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
source ${MONO_ROOT}/torch-env/bin/activate
python3 -m pip install -q joblib


# User-editable params
ALL_LAYERS="${ALL_LAYERS:-1}"       # default: all layers
LAYER_INDEX="${LAYER_INDEX:-0}"     # 0..4 for MLP layers (out_bn excluded)
OUTDIR="${OUTDIR:-./thesis/mlp_qinv_out/denoise_val_L500}" 
TITLE="${TITLE:-q_inv comparison (MLP Validation)}"

cd ${MONO_ROOT}

# Build layer args
if [ -n "$ALL_LAYERS" ] && [ "$ALL_LAYERS" != "0" ]; then
  LAYER_ARGS=(--all-layers)
else
  LAYER_ARGS=(--layer-index "$LAYER_INDEX")
fi

python analysis_scripts/mlp_qinv_calc_fig_two_sets.py \
  --spin_file_a1 ${MONO_ROOT}/mlp_output/denoise/diff/job_L500_0:885987.sqd/run_mnist_lambda500_val_20251218-010238/mlp_denoise_spinA_mnist_lambda500_val_seedA123.pkl\
  --spin_file_b1 ${MONO_ROOT}/mlp_output/denoise/diff/job_L500_0:885987.sqd/run_mnist_lambda500_val_20251218-010238/mlp_denoise_spinB_mnist_lambda500_val_seedB456.pkl\
  ${METRICS_A1:+--metrics_a1 "$METRICS_A1"} \
  ${METRICS_B1:+--metrics_b1 "$METRICS_B1"} \
  --label1       MLP_MNIST_DIFF_VAL \
  --spin_file_a2 ${MONO_ROOT}/mlp_output/denoise/same/job_L500_0:885985.sqd/run_same_mnist_lambda500_val_20251217-232945/mlp_same_model_spinA_mnist_lambda500_val_seed123_trA2025.pkl\
  --spin_file_b2 ${MONO_ROOT}/mlp_output/denoise/same/job_L500_0:885985.sqd/run_same_mnist_lambda500_val_20251217-232945/mlp_same_model_spinB_mnist_lambda500_val_seed123_trB4242.pkl\
  ${METRICS_A2:+--metrics_a2 "$METRICS_A2"} \
  ${METRICS_B2:+--metrics_b2 "$METRICS_B2"} \
  --label2       MLP_MNIST_SAME_VAL \
  "${LAYER_ARGS[@]}" \
  --title        "${TITLE}" \
  --output-dir   "${OUTDIR}"

exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"
exit $exit_code

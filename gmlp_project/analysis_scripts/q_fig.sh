#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m eb
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l cpunum_job=36
#PBS -l gpunum_job=1
#PBS -o ~/q2_analysis_log.out
#PBS -e ~/q2_analysis_log.err
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
#------- Program execution -----------

echo "🚀 Starting gMLP PyTorch Job"
echo "======================================"
echo "Job ID: $PBS_JOBID"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo ""

# === モジュール環境のセットアップ ===
echo "📦 Loading Python & GPU modules..."
module purge
module load BaseGPU/2025
module load BasePy/2025
module load python3/3.11
module load cudnn

# === 仮想環境をアクティベート ===
source ${MONO_ROOT}/torch-env/bin/activate

echo "🔍 Python version:"
which python
python --version

# === CUDA 環境確認 ===
echo "🎯 CUDA info:"
nvcc --version || echo "nvcc not found"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# === プロジェクトディレクトリへ移動 ===
cd ${MONO_ROOT}
echo "📁 Current directory: $(pwd)"






# === 実行ログ確認用 ===
echo "Running script: q_inv_calc_mul.py"


# === 実行 ===
python analysis_scripts/qinv_calc_fig_lossacc.py\
    --spin_file_a  ${MONO_ROOT}/output/recomputed_spins/random/gmlp_diff_model_p4_mnist_input_fashion/gmlp_spinA_train_D256_F1536_L10_M1000_seedA123_recomputed_A.pkl\
    --spin_file_b  ${MONO_ROOT}/output/recomputed_spins/random/gmlp_diff_model_p4_mnist_input_fashion/gmlp_spinB_train_D256_F1536_L10_M1000_seedB456_recomputed_B.pkl\
    --metrics-a  \
    --metrics-b  \
    --output-dir  ./thesis/randomnet\
    
    
    
    
    

# === 終了確認 ===
exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"


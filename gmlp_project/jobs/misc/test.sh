#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m be
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:10:00
#PBS -l gpunum_job=8
#PBS -o ../logs/test.out
#PBS -e ../logs/test.err
#PBS -r n

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
module load cudnncd

# === 仮想環境をアクティベート ===
source /sqfs/work/cm9029/${USER_ID}/torch-env/bin/activate

echo "🔍 Python version:"
which python
python --version

# === CUDA 環境確認 ===
echo "🎯 CUDA info:"
nvcc --version || echo "nvcc not found"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# === プロジェクトディレクトリへ移動 ===
cd ~/workspace/gmlp_project
echo "📁 Current directory: $(pwd)"

# === 実行ログ確認用 ===
echo "Running script: scripts/pca_mnist.py"


# === 実行 ===
python scripts/pca_mnist.py




# === 終了確認 ===
exit_code=$?
echo "🏁 Exit code: $exit_code"
echo "Done at: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ MLP PyTorch Job Completed Successfully!"
else
    echo "❌ MLP Job Failed. Check logs/test.err"
    tail -20 logs/test.err
fi
#!/bin/bash
#------- qsub option -----------
#PBS -q DBG
#PBS --group=cm9029
#PBS -m abe
#PBS -M fukami@cp.cmc.osaka-u.ac.jp
#PBS -l elapstim_req=00:05:00
#PBS -l cpunum_job=4
#PBS -o ~/overlap_metrics.out
#PBS -e ~/overlap_metrics.err
#PBS -r n

set -euo pipefail

echo "🚀 Starting overlap metrics demo"
echo "Job ID : ${PBS_JOBID:-manual}"
echo "Host   : $(hostname)"
echo "Time   : $(date)"
echo ""

module purge
module load BasePy/2025
module load python3/3.11

source /sqfs/work/cm9029/${USER_ID}/torch-env/bin/activate

REPO_ROOT="/sqfs/work/cm9029/${USER_ID}"
SCRATCH_BASE="/sqfs/ssd/cm9029/${USER_ID}"
SCRATCH_JOB_DIR="${SCRATCH_BASE}/overlap_metrics_${PBS_JOBID:-manual_$$}"
mkdir -p "$SCRATCH_JOB_DIR"
cleanup() {
  rm -rf "$SCRATCH_JOB_DIR"
}
trap cleanup EXIT
cd "$SCRATCH_JOB_DIR"
echo "📁 Scratch dir: $(pwd)"

python "$REPO_ROOT/analysis_scripts/overlap_metrics.py" --demo

status=$?
echo ""
echo "Job finished with status $status at $(date)"
exit $status

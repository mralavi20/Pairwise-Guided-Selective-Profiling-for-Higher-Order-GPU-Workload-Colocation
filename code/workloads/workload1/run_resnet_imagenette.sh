#!/bin/bash
#SBATCH --job-name=resnet_imagenette
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

GPU_TYPE="${1:-}"
RESULTS_FOLDER="${2:-results}"

if [ -z "${GPU_TYPE}" ]; then
    echo "Usage: sbatch --gres=gpu:GPU_TYPE:1 $0 GPU_TYPE [RESULTS_FOLDER]"
    echo "Example: sbatch --gres=gpu:A100:1 $0 A100"
    echo "Example: sbatch --gres=gpu:A100:1 $0 A100 results2"
    exit 1
fi

WORKLOAD_DIR="$HOME/workloads/workload1"
RESULTS_DIR="${WORKLOAD_DIR}/${RESULTS_FOLDER}/${GPU_TYPE}"

mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/resnet_imagenette_${GPU_TYPE}_${SLURM_JOB_ID:-nojobid}.out"
exec > "${LOG_FILE}" 2>&1

echo "============================================================"
echo " ResNet / Imagenette workload"
echo " GPU type   : ${GPU_TYPE}"
echo " Workload   : ${WORKLOAD_DIR}"
echo " Results    : ${RESULTS_DIR}"
echo " SLURM_JOBID: ${SLURM_JOB_ID}"
echo "============================================================"

JOB_START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
JOB_START_EPOCH="$(date +%s)"
echo "Job start time : ${JOB_START_HUMAN}"

module load python/3.10.18
source "$HOME/venvs/workloads/bin/activate"

cd "${RESULTS_DIR}"

python3 "${WORKLOAD_DIR}/code/resnet50.py" \
  --arch resnet50 \
  --use-imagenette \
  --imagenette-size 320px \
  --batch-size 16 \
  --epochs 50 \
  --workers 2 \
  --log-file "metrics_resnet50_imagenette_${GPU_TYPE}.json" \
  --csv-file "metrics_resnet50_imagenette_${GPU_TYPE}.csv" \
  "$HOME/data/imagenette"

PY_EXIT_CODE=$?

JOB_END_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
JOB_END_EPOCH="$(date +%s)"
echo "Job end time   : ${JOB_END_HUMAN}"

JOB_RUNTIME_SEC=$((JOB_END_EPOCH - JOB_START_EPOCH))
R_H=$((JOB_RUNTIME_SEC / 3600))
R_M=$(((JOB_RUNTIME_SEC % 3600) / 60))
R_S=$((JOB_RUNTIME_SEC % 60))

printf 'Job elapsed    : %02d:%02d:%02d (hh:mm:ss)\n' "$R_H" "$R_M" "$R_S"
echo "Python exit code: ${PY_EXIT_CODE}"

echo "============================================================"
echo " Job finished."
echo "============================================================"

exit "${PY_EXIT_CODE}"

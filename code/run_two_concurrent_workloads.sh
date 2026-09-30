#!/bin/bash
#
# Generic pairwise concurrent workload runner
#
# Usage:
#   sbatch --gres=gpu:GPU_TYPE:1 run_two_concurrent_workloads.sh GPU_TYPE WORKLOAD_A WORKLOAD_B
#
# Examples:
#   sbatch --gres=gpu:A100:1 run_two_concurrent_workloads.sh A100 workload1 workload3
#   sbatch --gres=gpu:a100:1 run_two_concurrent_workloads.sh a100 w2 w5
#   sbatch --gres=gpu:H100:1 run_two_concurrent_workloads.sh H100 resnet recoder
#   sbatch --gres=gpu:A100:1 run_two_concurrent_workloads.sh A100 w1 w1
#
# Accepted workload names:
#   w1 / workload1 / resnet / resnet50 / imagenette / resnet_imagenette
#   w2 / workload2 / transformer / multi30k / transformer_multi30k
#   w3 / workload3 / a3c / pong / a3c_pong
#   w4 / workload4 / cyclegan / monet / monet2photo / cyclegan_monet
#   w5 / workload5 / recoder / ml20m / movielens / recoder_ml20m

#SBATCH --job-name=two_concurrent_2x
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

set -euo pipefail

LAST_PID=""

usage() {
    cat <<EOF
Usage:
  sbatch --gres=gpu:GPU_TYPE:1 $0 GPU_TYPE WORKLOAD_A WORKLOAD_B [RESULTS_FOLDER]

Examples:
  sbatch --gres=gpu:A100:1 $0 A100 workload1 workload3
  sbatch --gres=gpu:a100:1 $0 a100 w2 w5
  sbatch --gres=gpu:H100:1 $0 H100 resnet recoder
  sbatch --gres=gpu:A100:1 $0 A100 w1 w1
  sbatch --gres=gpu:a100:1 $0 a100 workload1 workload3 results2
EOF
}

normalize_workload() {
    local x="${1,,}"
    case "${x}" in
        1|w1|workload1|resnet|resnet50|imagenette|resnet_imagenette)
            echo "w1"
            ;;
        2|w2|workload2|transformer|multi30k|transformer_multi30k)
            echo "w2"
            ;;
        3|w3|workload3|a3c|pong|a3c_pong|pongnoframeskip-v4)
            echo "w3"
            ;;
        4|w4|workload4|cyclegan|monet|monet2photo|cyclegan_monet)
            echo "w4"
            ;;
        5|w5|workload5|recoder|ml20m|movielens|movielens-20m|recoder_ml20m)
            echo "w5"
            ;;
        *)
            return 1
            ;;
    esac
}

workload_title() {
    case "$1" in
        w1) echo "ResNet50 / Imagenette" ;;
        w2) echo "Transformer / Multi30k" ;;
        w3) echo "A3C / PongNoFrameskip-v4" ;;
        w4) echo "CycleGAN / monet2photo" ;;
        w5) echo "Recoder / MovieLens-20M AE" ;;
        *)  echo "Unknown workload" ;;
    esac
}

workload_dir() {
    case "$1" in
        w1) echo "$HOME/workloads/workload1" ;;
        w2) echo "$HOME/workloads/workload2" ;;
        w3) echo "$HOME/workloads/workload3" ;;
        w4) echo "$HOME/workloads/workload4" ;;
        w5) echo "$HOME/workloads/workload5" ;;
        *)  return 1 ;;
    esac
}

launch_workload() {
    local wk="$1"
    local run_dir="$2"
    local port="$3"
    local slot_tag="$4"
    local wk_dir
    wk_dir="$(workload_dir "$wk")"

    mkdir -p "${run_dir}"

    case "${wk}" in
        w1)
            (
                set -e
                cd "${run_dir}"

                export MASTER_ADDR=127.0.0.1
                export MASTER_PORT="${port}"
                unset WORLD_SIZE RANK LOCAL_RANK

                python3 "${wk_dir}/code/resnet50.py" \
                    --arch resnet50 \
                    --use-imagenette \
                    --imagenette-size 320px \
                    --batch-size 16 \
                    --epochs 50 \
                    --workers 2 \
                    --log-file "metrics_resnet50_imagenette_${GPU_TYPE}_${PAIR_ID}_${slot_tag}.json" \
                    --csv-file "metrics_resnet50_imagenette_${GPU_TYPE}_${PAIR_ID}_${slot_tag}.csv" \
                    "$HOME/data/imagenette"
            ) > "${run_dir}/stdout_stderr.log" 2>&1 &
            LAST_PID=$!
            ;;

        w2)
            (
                set -e
                cd "${run_dir}"

                python3 "${wk_dir}/code/translate_transformer.py" \
                    --epochs 50 \
                    --batch-size 16 \
                    --workers 2 \
                    --data-cache-dir "$HOME/data/multi30k" \
                    --log-file "metrics_transformer_multi30k_${GPU_TYPE}_${PAIR_ID}_${slot_tag}.json" \
                    --csv-file "metrics_transformer_multi30k_${GPU_TYPE}_${PAIR_ID}_${slot_tag}.csv"
            ) > "${run_dir}/stdout_stderr.log" 2>&1 &
            LAST_PID=$!
            ;;

        w3)
            (
                set -e
                cd "${run_dir}"

                python3 "${wk_dir}/code/main.py" \
                    --env PongNoFrameskip-v4 \
                    --workers 2 \
                    --gpu-ids 0 \
                    --num-steps 20 \
                    --max-train-episodes 200 \
                    --max-test-episodes 50 \
                    --log-dir "logs" \
                    --save-model-dir "trained_models" \
                    --env-config "${wk_dir}/code/config.json"
            ) > "${run_dir}/stdout_stderr.log" 2>&1 &
            LAST_PID=$!
            ;;

        w4)
            (
                set -e
                cd "${run_dir}"
                mkdir -p "logs_metrics"

                python3 "${wk_dir}/code/train.py" \
                    --dataroot "${wk_dir}/code/datasets/monet2photo" \
                    --name "monet2photo_cyclegan_${GPU_TYPE}_${PAIR_ID}_${slot_tag}" \
                    --model cycle_gan \
                    --batch_size 4 \
                    --no_html \
                    --metrics_dir "logs_metrics" \
                    --run_tag "${GPU_TYPE}_bs4_iters5000_${PAIR_ID}_${slot_tag}" \
                    --max_train_iters 5000
            ) > "${run_dir}/stdout_stderr.log" 2>&1 &
            LAST_PID=$!
            ;;

        w5)
            (
                set -e
                cd "${run_dir}"
                mkdir -p "models" "logs_metrics"

                python3 "${wk_dir}/code/scripts/ml-20m/train.py" \
                    --data_dir "${wk_dir}/code/data/ml-20m/pro_sg/" \
                    --model_dir "models" \
                    --metrics_dir "logs_metrics" \
                    --run_tag "${GPU_TYPE}_1gpu_${PAIR_ID}_${slot_tag}" \
                    --num_epochs 500 \
                    --num_data_workers 2
            ) > "${run_dir}/stdout_stderr.log" 2>&1 &
            LAST_PID=$!
            ;;

        *)
            echo "ERROR: Unsupported workload key: ${wk}" >&2
            return 1
            ;;
    esac
}

GPU_TYPE="${1:-}"
RAW_WA="${2:-}"
RAW_WB="${3:-}"
RESULTS_FOLDER="${4:-results}"

if [ -z "${GPU_TYPE}" ] || [ -z "${RAW_WA}" ] || [ -z "${RAW_WB}" ]; then
    usage
    exit 1
fi

if ! WA_NORM="$(normalize_workload "${RAW_WA}")"; then
    echo "ERROR: Unknown workload A: ${RAW_WA}"
    usage
    exit 1
fi

if ! WB_NORM="$(normalize_workload "${RAW_WB}")"; then
    echo "ERROR: Unknown workload B: ${RAW_WB}"
    usage
    exit 1
fi

if [ "${WA_NORM#w}" -le "${WB_NORM#w}" ]; then
    WA="${WA_NORM}"
    WB="${WB_NORM}"
else
    WA="${WB_NORM}"
    WB="${WA_NORM}"
fi

PAIR_ID="${WA}_${WB}"

PAIRWISE_DIR="$HOME/workloads/two_concurrent_workloads"
RUN_ID="job_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${PAIRWISE_DIR}/${RESULTS_FOLDER}/${PAIR_ID}/${GPU_TYPE}/${RUN_ID}"

RUN_A_DIR="${RESULTS_DIR}/runA_${WA}"
RUN_B_DIR="${RESULTS_DIR}/runB_${WB}"

mkdir -p "${RUN_A_DIR}" "${RUN_B_DIR}"

LOG_FILE="${RESULTS_DIR}/pair_${PAIR_ID}_${GPU_TYPE}_${SLURM_JOB_ID:-manual}.out"
exec > "${LOG_FILE}" 2>&1

ACTUAL_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
ACTUAL_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
ACTUAL_GPU_PCI="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"

ACTUAL_GPU_NAME="${ACTUAL_GPU_NAME:-unknown}"
ACTUAL_GPU_UUID="${ACTUAL_GPU_UUID:-unknown}"
ACTUAL_GPU_PCI="${ACTUAL_GPU_PCI:-unknown}"

echo "Actual GPU name    : ${ACTUAL_GPU_NAME}"
echo "Actual GPU UUID    : ${ACTUAL_GPU_UUID}"
echo "Actual GPU PCI bus : ${ACTUAL_GPU_PCI}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

METADATA_FILE="${RESULTS_DIR}/run_metadata.txt"

JOB_NUM_FOR_PORT="${SLURM_JOB_ID:-$$}"
BASE_PORT=$((20000 + (JOB_NUM_FOR_PORT % 10000) * 2))
PORT_A="${BASE_PORT}"
PORT_B=$((BASE_PORT + 1))

trap 'echo "Received interrupt/termination signal. Killing child processes..."; jobs -pr | xargs -r kill || true' INT TERM

echo "============================================================"
echo " Pairwise concurrent workload runner (identical allowed)"
echo " GPU type        : ${GPU_TYPE}"
echo " Raw inputs      : ${RAW_WA} + ${RAW_WB}"
echo " Canonical pair  : ${WA} + ${WB}"
echo " Workload A      : $(workload_title "${WA}")"
echo " Workload B      : $(workload_title "${WB}")"
echo " Pair ID         : ${PAIR_ID}"
echo " Results dir     : ${RESULTS_DIR}"
echo " Run A dir       : ${RUN_A_DIR}"
echo " Run B dir       : ${RUN_B_DIR}"
echo " SLURM_JOBID     : ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

JOB_START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
JOB_START_EPOCH="$(date +%s)"

echo "Job start time   : ${JOB_START_HUMAN}"
echo "Port A           : ${PORT_A}"
echo "Port B           : ${PORT_B}"

cat > "${METADATA_FILE}" <<EOF
gpu_type=${GPU_TYPE}
raw_workload_a=${RAW_WA}
raw_workload_b=${RAW_WB}
canonical_workload_a=${WA}
canonical_workload_b=${WB}
canonical_title_a=$(workload_title "${WA}")
canonical_title_b=$(workload_title "${WB}")
pair_id=${PAIR_ID}
results_dir=${RESULTS_DIR}
run_a_dir=${RUN_A_DIR}
run_b_dir=${RUN_B_DIR}
slurm_job_id=${SLURM_JOB_ID:-N/A}
job_start_human=${JOB_START_HUMAN}
job_start_epoch=${JOB_START_EPOCH}
port_a=${PORT_A}
port_b=${PORT_B}
EOF

module load python/3.10.18
source "$HOME/venvs/workloads/bin/activate"

echo "Python executable : $(command -v python || true)"
echo "Python3 executable: $(command -v python3 || true)"

echo
echo "Launching workload A (${WA}: $(workload_title "${WA}")) ..."
launch_workload "${WA}" "${RUN_A_DIR}" "${PORT_A}" "runA"
PID_A="${LAST_PID}"
echo "Started workload A with PID=${PID_A}"

echo
echo "Launching workload B (${WB}: $(workload_title "${WB}")) ..."
launch_workload "${WB}" "${RUN_B_DIR}" "${PORT_B}" "runB"
PID_B="${LAST_PID}"
echo "Started workload B with PID=${PID_B}"

echo
echo "Both workloads started. Waiting for them to finish..."
echo

set +e
wait "${PID_A}"
STATUS_A=$?
wait "${PID_B}"
STATUS_B=$?
set -e

echo "Workload A exit code : ${STATUS_A}"
echo "Workload B exit code : ${STATUS_B}"

if [ "${STATUS_A}" -ne 0 ] || [ "${STATUS_B}" -ne 0 ]; then
    PY_EXIT_CODE=1
else
    PY_EXIT_CODE=0
fi

JOB_END_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
JOB_END_EPOCH="$(date +%s)"

JOB_RUNTIME_SEC=$((JOB_END_EPOCH - JOB_START_EPOCH))
R_H=$((JOB_RUNTIME_SEC / 3600))
R_M=$(((JOB_RUNTIME_SEC % 3600) / 60))
R_S=$((JOB_RUNTIME_SEC % 60))

echo "Job end time        : ${JOB_END_HUMAN}"
printf 'Job elapsed         : %02d:%02d:%02d (hh:mm:ss)\n' "$R_H" "$R_M" "$R_S"
echo "Overall exit code   : ${PY_EXIT_CODE}"

cat >> "${METADATA_FILE}" <<EOF
job_end_human=${JOB_END_HUMAN}
job_end_epoch=${JOB_END_EPOCH}
job_runtime_sec=${JOB_RUNTIME_SEC}
workload_a_exit_code=${STATUS_A}
workload_b_exit_code=${STATUS_B}
overall_exit_code=${PY_EXIT_CODE}
EOF

echo "============================================================"
echo " Job finished."
echo "============================================================"

exit "${PY_EXIT_CODE}"

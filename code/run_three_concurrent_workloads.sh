#!/bin/bash
#
# Generic triple concurrent workload runner
#
# Usage:
#   sbatch --gres=gpu:GPU_TYPE:1 run_three_concurrent_workloads.sh GPU_TYPE WORKLOAD_A WORKLOAD_B WORKLOAD_C
#
# Examples:
#   sbatch --gres=gpu:A100:1 run_three_concurrent_workloads.sh A100 workload1 workload3 workload5
#   sbatch --gres=gpu:a100:1 run_three_concurrent_workloads.sh a100 w2 w5 w1
#   sbatch --gres=gpu:H100:1 run_three_concurrent_workloads.sh H100 resnet recoder a3c
#   sbatch --gres=gpu:A100:1 run_three_concurrent_workloads.sh A100 w1 w1 w1
#   sbatch --gres=gpu:A100:1 run_three_concurrent_workloads.sh A100 w1 w1 w3
#
# Accepted workload names:
#   w1 / workload1 / resnet / resnet50 / imagenette / resnet_imagenette
#   w2 / workload2 / transformer / multi30k / transformer_multi30k
#   w3 / workload3 / a3c / pong / a3c_pong
#   w4 / workload4 / cyclegan / monet / monet2photo / cyclegan_monet
#   w5 / workload5 / recoder / ml20m / movielens / recoder_ml20m

#SBATCH --job-name=three_concurrent_3x
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32000M
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

set -euo pipefail

LAST_PID=""

usage() {
    cat <<EOF
Usage:
  sbatch --gres=gpu:GPU_TYPE:1 $0 GPU_TYPE WORKLOAD_A WORKLOAD_B WORKLOAD_C [RESULTS_FOLDER]

Examples:
  sbatch --gres=gpu:A100:1 $0 A100 workload1 workload3 workload5
  sbatch --gres=gpu:a100:1 $0 a100 w2 w5 w1
  sbatch --gres=gpu:H100:1 $0 H100 resnet recoder a3c
  sbatch --gres=gpu:A100:1 $0 A100 w1 w1 w3
  sbatch --gres=gpu:a100:1 $0 a100 workload1 workload3 workload5 results2
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
                    --log-file "metrics_resnet50_imagenette_${GPU_TYPE}_${TRIPLE_ID}_${slot_tag}.json" \
                    --csv-file "metrics_resnet50_imagenette_${GPU_TYPE}_${TRIPLE_ID}_${slot_tag}.csv" \
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
                    --log-file "metrics_transformer_multi30k_${GPU_TYPE}_${TRIPLE_ID}_${slot_tag}.json" \
                    --csv-file "metrics_transformer_multi30k_${GPU_TYPE}_${TRIPLE_ID}_${slot_tag}.csv"
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
                    --name "monet2photo_cyclegan_${GPU_TYPE}_${TRIPLE_ID}_${slot_tag}" \
                    --model cycle_gan \
                    --batch_size 4 \
                    --no_html \
                    --metrics_dir "logs_metrics" \
                    --run_tag "${GPU_TYPE}_bs4_iters5000_${TRIPLE_ID}_${slot_tag}" \
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
                    --run_tag "${GPU_TYPE}_1gpu_${TRIPLE_ID}_${slot_tag}" \
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
RAW_WC="${4:-}"
RESULTS_FOLDER="${5:-results}"

if [ -z "${GPU_TYPE}" ] || [ -z "${RAW_WA}" ] || [ -z "${RAW_WB}" ] || [ -z "${RAW_WC}" ]; then
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

if ! WC_NORM="$(normalize_workload "${RAW_WC}")"; then
    echo "ERROR: Unknown workload C: ${RAW_WC}"
    usage
    exit 1
fi

mapfile -t SORTED_WORKLOADS < <(
    printf "%s\n" "${WA_NORM}" "${WB_NORM}" "${WC_NORM}" | sed 's/^w//' | sort -n | sed 's/^/w/'
)

WA="${SORTED_WORKLOADS[0]}"
WB="${SORTED_WORKLOADS[1]}"
WC="${SORTED_WORKLOADS[2]}"

TRIPLE_ID="${WA}_${WB}_${WC}"

TRIPLEWISE_DIR="$HOME/workloads/three_concurrent_workloads"
RUN_ID="job_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${TRIPLEWISE_DIR}/${RESULTS_FOLDER}/${TRIPLE_ID}/${GPU_TYPE}/${RUN_ID}"

RUN_A_DIR="${RESULTS_DIR}/runA_${WA}"
RUN_B_DIR="${RESULTS_DIR}/runB_${WB}"
RUN_C_DIR="${RESULTS_DIR}/runC_${WC}"

mkdir -p "${RUN_A_DIR}" "${RUN_B_DIR}" "${RUN_C_DIR}"

LOG_FILE="${RESULTS_DIR}/triple_${TRIPLE_ID}_${GPU_TYPE}_${SLURM_JOB_ID:-manual}.out"
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
BASE_PORT=$((20000 + (JOB_NUM_FOR_PORT % 10000) * 3))
PORT_A="${BASE_PORT}"
PORT_B=$((BASE_PORT + 1))
PORT_C=$((BASE_PORT + 2))

trap 'echo "Received interrupt/termination signal. Killing child processes..."; jobs -pr | xargs -r kill || true' INT TERM

echo "============================================================"
echo " Triple concurrent workload runner (identical allowed)"
echo " GPU type        : ${GPU_TYPE}"
echo " Raw inputs      : ${RAW_WA} + ${RAW_WB} + ${RAW_WC}"
echo " Canonical triple: ${WA} + ${WB} + ${WC}"
echo " Workload A      : $(workload_title "${WA}")"
echo " Workload B      : $(workload_title "${WB}")"
echo " Workload C      : $(workload_title "${WC}")"
echo " Triple ID       : ${TRIPLE_ID}"
echo " Results dir     : ${RESULTS_DIR}"
echo " Run A dir       : ${RUN_A_DIR}"
echo " Run B dir       : ${RUN_B_DIR}"
echo " Run C dir       : ${RUN_C_DIR}"
echo " SLURM_JOBID     : ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

JOB_START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
JOB_START_EPOCH="$(date +%s)"

echo "Job start time   : ${JOB_START_HUMAN}"
echo "Port A           : ${PORT_A}"
echo "Port B           : ${PORT_B}"
echo "Port C           : ${PORT_C}"

cat > "${METADATA_FILE}" <<EOF
gpu_type=${GPU_TYPE}
raw_workload_a=${RAW_WA}
raw_workload_b=${RAW_WB}
raw_workload_c=${RAW_WC}
canonical_workload_a=${WA}
canonical_workload_b=${WB}
canonical_workload_c=${WC}
canonical_title_a=$(workload_title "${WA}")
canonical_title_b=$(workload_title "${WB}")
canonical_title_c=$(workload_title "${WC}")
triple_id=${TRIPLE_ID}
results_dir=${RESULTS_DIR}
run_a_dir=${RUN_A_DIR}
run_b_dir=${RUN_B_DIR}
run_c_dir=${RUN_C_DIR}
slurm_job_id=${SLURM_JOB_ID:-N/A}
job_start_human=${JOB_START_HUMAN}
job_start_epoch=${JOB_START_EPOCH}
port_a=${PORT_A}
port_b=${PORT_B}
port_c=${PORT_C}
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
echo "Launching workload C (${WC}: $(workload_title "${WC}")) ..."
launch_workload "${WC}" "${RUN_C_DIR}" "${PORT_C}" "runC"
PID_C="${LAST_PID}"
echo "Started workload C with PID=${PID_C}"

echo
echo "All three workloads started. Waiting for them to finish..."
echo

set +e
wait "${PID_A}"
STATUS_A=$?
wait "${PID_B}"
STATUS_B=$?
wait "${PID_C}"
STATUS_C=$?
set -e

echo "Workload A exit code : ${STATUS_A}"
echo "Workload B exit code : ${STATUS_B}"
echo "Workload C exit code : ${STATUS_C}"

if [ "${STATUS_A}" -ne 0 ] || [ "${STATUS_B}" -ne 0 ] || [ "${STATUS_C}" -ne 0 ]; then
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
workload_c_exit_code=${STATUS_C}
overall_exit_code=${PY_EXIT_CODE}
EOF

echo "============================================================"
echo " Job finished."
echo "============================================================"

exit "${PY_EXIT_CODE}"

#!/bin/bash

# Script to run Python CNN training by reconnecting to an existing SLURM job allocation.

# --- Configuration ---
JOB_ID=""
TRAINING_JOB_ID=""

# --- Argument Parser (Unchanged) ---
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --job_id)
            if [[ -n "$2" && "$2" != --* ]]; then
                JOB_ID="$2"; shift
            else
                echo "Error: --job_id requires a value."; exit 1
            fi
            ;;
        --load_id)
            if [[ -n "$2" && "$2" != --* ]]; then
                TRAINING_JOB_ID="$2"; shift
            fi
            ;;
        *)
            echo "Unknown parameter passed: $1"; exit 1
            ;;
    esac
    shift
done

if [[ -z "$JOB_ID" ]]; then
    echo "Error: You must provide the --job_id of the sleeping job."; exit 1
fi

# --- Resource and Model Configuration (Unchanged) ---
GPUS_PER_NODE=4
TOTAL_CPUS=64
CPUS_PER_TASK=$((TOTAL_CPUS / GPUS_PER_NODE))
MODEL_PARAMS="D L ro rr p Tlow NCH3CN plummer_shape"
LOG_SCALE_PARAMS="D L NCH3CN"
NODE_LOCAL_DIR="/local/nvme/${JOB_ID}_fits_data"
CURRENT_DIR="$(pwd)"
SCRIPT_PATH="${CURRENT_DIR}/ResNet3D_DDP.py"
VENV_PATH="/p/project/pasta/jusuf-radmc/.radmc_venv_2024/bin/activate"
LOG_FILE="slurm_output/cnn_run_ddp_${JOB_ID}.log"
MODULES_TO_LOAD="Stages/2024 GCCcore/.12.3.0 Python/3.11.3"

# =================================================================================
# JOB EXECUTION LOGIC
# =================================================================================
echo -e "\n\n==================== RECONNECT RUN STARTED: $(date '+%Y-%m-%d %H:%M:%S') ====================\n" >> "${LOG_FILE}"
echo "--- Reconnecting to Job ${JOB_ID} ---"

# --- CRITICAL NETWORK SETUP (Done ONCE on the login node) ---
# srun will propagate these to all tasks.
#export MASTER_ADDR=$(scontrol show hostnames $(scontrol show job ${JOB_ID} | grep -o 'NodeList=[^ ]*' | cut -d'=' -f2) | head -n 1)
#export MASTER_PORT=54123
export NCCL_SOCKET_IFNAME=ib0
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMP_DIR:-/p/scratch/westai0043}/mplcache_${JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

#echo "Master Address determined to be: ${MASTER_ADDR}"
#echo "Master Port set to: ${MASTER_PORT}"

# --- The Final, Simplified `srun` command ---
# It now launches PYTHON directly, not torchrun.
srun --jobid=${JOB_ID} \
     --ntasks=${GPUS_PER_NODE} \
     --gpus-per-task=1 \
     --gpu-bind=single:1 \
     --cpus-per-task=${CPUS_PER_TASK} \
     --export=ALL,NCCL_SOCKET_IFNAME,MPLBACKEND,MPLCONFIGDIR \
bash -c '
set -euo pipefail

# 0) Clean any inherited rendezvous env
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RENDEZVOUS_ENDPOINT C10D_USE_IPV6 GLOO_USE_IPV6 TP_USE_IPV6 || true

# --- 1) Determine master IPv4 on rank 0 and share it ---
TMP_FILE="/tmp/master_addr_${SLURM_JOB_ID}.txt"

if [ "${SLURM_PROCID}" -eq 0 ]; then
    # Prefer the actual IPv4 on ib0 (robust vs DNS)
    if MASTER_IP=$(ip -4 -o addr show dev ib0 2>/dev/null | awk "{print \$4}" | cut -d/ -f1 | head -n1); then
        :
    else
        # Fallback: resolve first node in nodelist, force IPv4
        HOST_FIRST="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$SLURM_NODELIST}" | head -n1)"
        MASTER_IP=$(getent ahostsv4 "${HOST_FIRST}i" | awk "{print \$1; exit}") || true
        if [ -z "${MASTER_IP:-}" ]; then
            MASTER_IP=$(ping -4 -c 1 "${HOST_FIRST}i" | head -n1 | grep -oE "([0-9]{1,3}\.){3}[0-9]{1,3}") || true
        fi
    fi

    if [ -z "${MASTER_IP:-}" ]; then
        echo "[Task 0] FATAL: Could not determine MASTER_IP" >&2
        exit 1
    fi

    echo "${MASTER_IP}" > "${TMP_FILE}"
    echo "[Task 0] Master address resolved to: ${MASTER_IP}"
fi

# Wait up to 20s for the file to appear and be non-empty
for _ in $(seq 1 20); do
    [ -s "${TMP_FILE}" ] && break
    sleep 1
done
if [ ! -s "${TMP_FILE}" ]; then
    echo "[Task ${SLURM_PROCID}] FATAL: MASTER_ADDR file not found/empty: ${TMP_FILE}" >&2
    exit 1
fi

MASTER_ADDR="$(cat "${TMP_FILE}")"
MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

# 2) Export only interface + rank hints (no MASTER_*)
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export TP_SOCKET_IFNAME=ib0
export GLOO_USE_IPV6=0
export TP_USE_IPV6=0
export C10D_USE_IPV6=0

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"

echo "[Rank ${RANK}] Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}  (WORLD_SIZE=${WORLD_SIZE}, LOCAL_RANK=${LOCAL_RANK})"


# --- 3) Environment setup (UNQUOTED to load multiple modules) ---
module --force purge >/dev/null 2>&1 || true
module load '"${MODULES_TO_LOAD}"'
#                  ^^^^^^^^^^^^^^^^^  DO NOT QUOTE in your script. Use: module load ${MODULES_TO_LOAD}
# If your wrapper insists on quotes above, replace this exact line in your script with:
#   module load ${MODULES_TO_LOAD}

source '"${VENV_PATH}"'

# --- 4) Launch Python ---
echo "[Rank ${RANK}] Launching Python script..."
python -u '"${SCRIPT_PATH}"' \
    --rdzv-addr "${MASTER_ADDR}" \
    --rdzv-port "${MASTER_PORT}" \
    --data-dir '"${NODE_LOCAL_DIR}"' \
    --wavelength-stride 1 \
    --load-preprocessed False \
    --use-local-nvme True \
    --batch-size 32 \
    --num-workers 8 \
    --model-depth 18 \
    --num-epochs 20 \
    --model_params '"${MODEL_PARAMS}"' \
    --log-scale-params '"${LOG_SCALE_PARAMS}"' \
    --job_id ${SLURM_JOB_ID} \
    --use_attention_heads False \
    --data-subset-fraction 0.8 \
    --add-noise-level 0.01 \
    --snr-threshold 5.0 \
    --use-cauchy-noise True \
    --use-ddp True \
    '"${TRAINING_JOB_ID:+--load_id "$TRAINING_JOB_ID"}"'
' 2>&1 | tee -a "${LOG_FILE}"


SRUN_EXIT_CODE=$?
echo "--- srun reconnect command finished with exit code: ${SRUN_EXIT_CODE} ---"
date
exit $SRUN_EXIT_CODE
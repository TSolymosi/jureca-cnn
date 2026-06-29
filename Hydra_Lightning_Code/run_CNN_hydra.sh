#!/bin/bash

# Hydra version of CNN launcher using a sleeper job

# ------------------ Parse arguments ------------------
JOB_ID="14077158"
TRAINING_JOB_ID=""
RUN_MODE="" # sweep for Optuna multirun, empty for single run

LOAD_OPTION=""   # best, last, or a filename

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --job_id) JOB_ID="$2"; shift ;;
        --load_id) TRAINING_JOB_ID="$2"; shift ;;
        --load_option) LOAD_OPTION="$2"; shift ;;
        -m|--sweep) RUN_MODE="sweep" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done



# ------------------ Parameters ------------------
CPUS_FOR_PYTHON=8
GPUS_PER_NODE=4
SLURM_NNODES=1

#MODEL_PARAMS="D L ro rr p Tlow NCH3CN plummer_shape"
#LOG_SCALE_PARAMS="D L NCH3CN"
FOLDER_NAME="first_test"
SCALING_PARAMS_PATH="/p/scratch/westai0101/CNN_HL_tobias/Parameters/${JOB_ID}/label_scaling.pt"
NODE_LOCAL_DIR="/local/nvme/${JOB_ID}_fits_data"

#VENV_PATH="/p/scratch/westai0043/CNN_timon/testing/sc_venv_template/venv/bin/activate"
VENV_PATH="/p/project1/pasta/jusuf-radmc/.radmc_venv_2024/bin/activate"
SCRIPT_PATH="/p/scratch/westai0101/CNN_HL_tobias/src/train.py"
LOG_DIR="slurm_output/cnn_run"
LOG_FILE="${LOG_DIR}/hydra_run_${JOB_ID}.log"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"
# Redirect ALL subsequent output to both terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1


echo -e "\n\n==================== RUN STARTED: $(date '+%Y-%m-%d %H:%M:%S') ====================\n"

# ------------------ Modules & Env ------------------
export HYDRA_FULL_ERROR=1
# --- Configuration ---
# The absolute path to your project's root directory.
# The `pwd` command automatically gets the current directory.
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
#export PYTHONPATH=/p/scratch/westai0043/CNN_timon/Hydra_Lightning/src:$PYTHONPATH
echo "Project Root: $PROJECT_ROOT"
echo "PYTHONPATH set to: $PYTHONPATH"

# Specify matplotlib cache directory
mkdir -p "${PROJECT_ROOT}/_pycache_/.matplotlib_cache"
export MPLCONFIGDIR="${PROJECT_ROOT}/_pycache_/.matplotlib_cache"




#module load Stages/2025 GCC PyTorch torchvision Python/3.12.3 || { echo 'Module load failed'; exit 1; }

PYTHON_BIN="${VENV_PATH%/activate}/python"

# ------------------ Hydra overrides ------------------
ARGS=(
  "data.data_dir=${NODE_LOCAL_DIR}"
  #"data.model_params=[$(tr ' ' , <<<"${MODEL_PARAMS}")]"
  #"data.log_scale_params=[$(tr ' ' , <<<"${LOG_SCALE_PARAMS}")]"
  "data.model_params=['M', 'D','L','ro','p','Tlow','NCH3CN']" #, 'plummer_shape'
  "data.log_scale_params=['M','D','L','NCH3CN']"
  "model.model_depth=18"
  "data.use_local_nvme=True"
  "data.batch_size=4"
  "data.num_workers=8"
  "data.data_subset_fraction=0.9995"
  "data.scaling_params_path=${SCALING_PARAMS_PATH}"
  "trainer.accelerator=gpu"
  "trainer.devices=${GPUS_PER_NODE}"
  "trainer.num_nodes=${SLURM_NNODES}"
  "trainer.max_epochs=250"
  "+job_id=${JOB_ID}"
  "data.use_cauchy_noise=True"
  "data.mask_13co=True"
  "model.num_mixtures=7"
  #"data.add_noise_level=0.0"
)

[[ -n "${FOLDER_NAME}" ]]     && ARGS+=("+folder_name=${FOLDER_NAME}")
[[ -n "${TRAINING_JOB_ID}" ]] && ARGS+=("+load_id=${TRAINING_JOB_ID}")
[[ -n "${LOAD_OPTION}" ]] && ARGS+=("+load_option=${LOAD_OPTION}")


# ------------------ NCCL setup (single-node friendly) ------------------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29577
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset NCCL_ASYNC_ERROR_HANDLING




# ------------------ Run ------------------
echo -e "\n\n==================== HYDRA RUN STARTED: $(date '+%Y-%m-%d %H:%M:%S') ====================\n" >> "${LOG_FILE}"



if [[ "$RUN_MODE" == "sweep" ]]; then
    echo "[INFO] Launching Optuna sweep (multirun mode)..."
    srun --jobid="${JOB_ID}" --cpu-bind=none \
         --ntasks-per-node="${GPUS_PER_NODE}" \
         --gpus-per-task=1 \
         --cpus-per-task="${CPUS_FOR_PYTHON}" \
         "${PYTHON_BIN}" "${SCRIPT_PATH}" -m "${ARGS[@]}" #2>&1 | tee -a "${LOG_FILE}"
else
    echo "[INFO] Launching single run..."
    srun --jobid="${JOB_ID}" --cpu-bind=none \
         --ntasks-per-node="${GPUS_PER_NODE}" \
         --gpus-per-task=1 \
         --cpus-per-task="${CPUS_FOR_PYTHON}" \
         "${PYTHON_BIN}" "${SCRIPT_PATH}" "${ARGS[@]}" #2>&1 | tee -a "${LOG_FILE}"
fi

exit $?

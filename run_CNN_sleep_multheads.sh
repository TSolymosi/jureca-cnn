#!/bin/bash

# Script to run the Python CNN training within an existing SLURM job allocation
# Loading modules and activating venv inside srun's bash -c

# --- Configuration - Needs to be updated before running! ---
JOB_ID="" # Default value of the actual JOB ID of the sleeping job
TRAINING_JOB_ID="" # Default value of the actual JOB ID of the trained job to continue
# Example usage: ./run_CNN_sleep_multheads.sh --job_id 12345678 --load_id 87654321
# Set the job ID and load ID from command line arguments if provided

# Argument parser
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --job_id)
            if [[ -n "$2" && "$2" != --* ]]; then
                JOB_ID="$2"
                shift
            else
                echo "Error: --job_id requires a value."
                exit 1
            fi
            ;;
        --load_id)
            if [[ -n "$2" && "$2" != --* ]]; then
                TRAINING_JOB_ID="$2"
                shift
            else
                TRAINING_JOB_ID=""  # Explicitly clear if no value provided
            fi
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

CPUS_FOR_PYTHON=8 # Match or be less than --cpus-per-task in the sleep job

# ---Parameters to train---
#D=3.054e+09_L=6.190e+04_ri=50_ro=1596_rr=205_p=1.18_np=100000_edr=1_rvar=0.33_phivar=1.4_LTE_Tlow=86.4_Thigh=800_NCH3CN=1.0e-10_vin=0.1_incl=130_phi=334_dist=3000_0.5arcsec
MODEL_PARAMS="D L ro rr p Tlow NCH3CN plummer_shape" # Parameters to train on  plummer_shape
LOG_SCALE_PARAMS="D L NCH3CN" # Parameters to log scale during training

# --- Define Paths ---
NODE_LOCAL_DIR="/local/nvme/${JOB_ID}_fits_data"
#ORIGINAL_FILE_LIST="${ORIGINAL_FITS_SCRATCH_DIR}/file_list.txt"
#SCALING_PARAMS_PATH="/p/scratch/pasta/production_run/CNN_Python/scaling_params.pt" # Correct path
SCRIPT_DIR="/p/project/pasta/jusuf-radmc/jureca-cnn"
SCRIPT_NAME="ResNet3D.py"

CURRENT_DIR="$(pwd)" # Current working directory
SCRIPT_PATH="${CURRENT_DIR}/${SCRIPT_NAME}"
VENV_PATH="/p/project/pasta/jusuf-radmc/.radmc_venv_2024/bin/activate" # Path to your venv activation script
SAVE_DIR="${CURRENT_DIR}/${TRAINING_JOB_ID}_model_checkpoints" # Where checkpoints are saved/loaded from

# Log file for the Python script named after the job ID
LOG_FILE="slurm_output/cnn_run_${JOB_ID}.log"


# --- Checkpoint Configuration ---
# Set this variable to the specific checkpoint file you want to resume from.
# Leave empty ("") to train from scratch.
# Example: RESUME_CHECKPOINT_PATH="${SAVE_DIR}/checkpoint_epoch_60.pth"
#RESUME_CHECKPOINT_PATH="${SAVE_DIR}/final_model_state.pth" # Set to path or leave empty
RESUME_CHECKPOINT_PATH=""

# --- Define Modules to Load ---
MODULES_TO_LOAD="Stages/2024 GCCcore/.12.3.0 Python/3.11.3" # List necessary modules


# Append divider and timestamp to log
echo -e "\n\n==================== RUN STARTED: $(date '+%Y-%m-%d %H:%M:%S') ====================\n" >> "${LOG_FILE}"

echo "--- Executing Python within Job ${JOB_ID} ---"
echo "Using Node-Local Data: ${NODE_LOCAL_DIR}"
echo "Using Script: ${SCRIPT_PATH}"
echo "CPUs for srun step: ${CPUS_FOR_PYTHON}"
echo "Loading Modules: ${MODULES_TO_LOAD}"
echo "Activating Venv: ${VENV_PATH}"
date

echo 'Loading modules...'
module load ${MODULES_TO_LOAD} || { echo 'ERROR: Failed to load modules inside srun'; exit 1; }
echo 'Modules loaded.'

# Execute srun, wrapping the python command in a subshell that loads modules and activates the venv
srun --jobid=${JOB_ID} --cpu-bind=none --ntasks=1 --cpus-per-task=${CPUS_FOR_PYTHON} \
bash -c "
echo '--- Inside srun subshell ---'

module --force purge
module load ${MODULES_TO_LOAD}
echo 'Activating venv...'
source \"${VENV_PATH}\" || { echo 'ERROR: Failed to activate venv inside srun'; exit 1; }
unset PYTHONPATH PYTHONHOME
hash -r
echo 'Venv activated.'
echo 'Running python: $(which python)' # Verify python path
python -V      # 3.11.3

# Before any Python import of matplotlib:
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/p/scratch/westai0043}/mplcache_${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

echo 'Job ID: ${JOB_ID}'
echo 'Continuing training with job ID: ${TRAINING_JOB_ID}'
echo 'Executing Python script: ${SCRIPT_PATH}'
python \"${SCRIPT_PATH}\" \\
    --data-dir \"${NODE_LOCAL_DIR}\" \\
    --wavelength-stride 1 \\
    --load-preprocessed False \\
    --use-local-nvme True \\
    --batch-size 20 \\
    --num-workers 8 \\
    --model-depth 18  \\
    --num-epochs 20 \\
    --model_params ${MODEL_PARAMS} \\
    --log-scale-params ${LOG_SCALE_PARAMS} \\
    --job_id ${JOB_ID} \\
    --use_attention_heads False \\
    --data-subset-fraction 0.8 \\
    --add-noise-level 0.01 \\
    --snr-threshold 5.0 \\
    ${TRAINING_JOB_ID:+--load_id "$TRAINING_JOB_ID"}
echo 'Python script finished.'
" 2>&1 | tee -a "${LOG_FILE}" # Capture output to log file

# Optional: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
# Capture the exit code of srun itself
SRUN_EXIT_CODE=$?
echo "--- srun command finished with exit code: ${SRUN_EXIT_CODE} ---"
date

exit $SRUN_EXIT_CODE
#!/bin/bash

# Script to run the Python CNN training within an existing SLURM job allocation
# Loading modules and activating venv inside srun's bash -c

# --- Configuration - Needs to be updated before running! ---
JOB_ID="13698517" # Replace with the actual JOB ID of the sleeping job
TRAINING_JOB_ID="" # Replace with the actual JOB ID of the trained job to continue
CPUS_FOR_PYTHON=8 # Match or be less than --cpus-per-task in the sleep job

# ---Parameters to train---
MODEL_PARAMS="Dens Lum radius prho" # Parameters to train on
LOG_SCALE_PARAMS="Dens Lum radius" # Parameters to log scale during training

# --- Define Paths ---
NODE_LOCAL_DIR="/local/nvme/${JOB_ID}_fits_data"
ORIGINAL_FITS_SCRATCH_DIR="/p/scratch/pasta/CNN/17.03.25/Data_100"
#ORIGINAL_FILE_LIST="${ORIGINAL_FITS_SCRATCH_DIR}/file_list.txt"
#SCALING_PARAMS_PATH="/p/scratch/pasta/production_run/CNN_Python/scaling_params.pt" # Correct path
SCRIPT_DIR="/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN_TS"
SCRIPT_NAME="ResNet3D.py"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
VENV_PATH="/p/scratch/pasta/CNN/.cnn_venv/bin/activate" # Path to your venv activation script
SAVE_DIR="${SCRIPT_DIR}/${TRAINING_JOB_ID}_model_checkpoints" # Where checkpoints are saved/loaded from

# --- Checkpoint Configuration ---
# Set this variable to the specific checkpoint file you want to resume from.
# Leave empty ("") to train from scratch.
# Example: RESUME_CHECKPOINT_PATH="${SAVE_DIR}/checkpoint_epoch_60.pth"
#RESUME_CHECKPOINT_PATH="${SAVE_DIR}/final_model_state.pth" # Set to path or leave empty
RESUME_CHECKPOINT_PATH=""

# --- Define Modules to Load ---
MODULES_TO_LOAD="Stages/2025 GCC PyTorch torchvision Python" # List necessary modules

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

echo 'Activating venv...'
source \"${VENV_PATH}\" || { echo 'ERROR: Failed to activate venv inside srun'; exit 1; }
echo 'Venv activated.'
echo 'Running python: $(which python)' # Verify python path
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python \"${SCRIPT_PATH}\" \\
    --data-dir \"${NODE_LOCAL_DIR}\" \\
    --wavelength-stride 1 \\
    --load-preprocessed False \\
    --use-local-nvme True \\
    --batch-size 64 \\
    --num-workers 8 \\
    --model-depth 18  \\
    --num-epochs 60 \\
    --model_params ${MODEL_PARAMS} \\
    --log-scale-params ${LOG_SCALE_PARAMS} \\
    --job_id ${JOB_ID} 
echo 'Python script finished.'
" | tee slurm_output/cnn_run_100.log
#" > "slurm_output/cnn_run.log" 2>&1 # End of bash -c command string

# Capture the exit code of srun itself
SRUN_EXIT_CODE=$?
echo "--- srun command finished with exit code: ${SRUN_EXIT_CODE} ---"
date

exit $SRUN_EXIT_CODE
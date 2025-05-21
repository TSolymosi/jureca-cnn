#!/bin/bash

# Script to run the Python CNN training within an existing SLURM job allocation
# Loading modules and activating venv inside srun's bash -c

# --- Configuration - Needs to be updated before running! ---
JOB_ID="13702381" # Replace with the actual JOB ID of the sleeping job
TRAINING_JOB_ID="13620964" # Replace with the actual JOB ID of the trained job to continue
CPUS_FOR_PYTHON=8 # Match or be less than --cpus-per-task in the sleep job

# --- Define Paths ---
NODE_LOCAL_DIR="/local/nvme/${JOB_ID}_fits_data"
ORIGINAL_FILE_LIST_PATH="/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN/FileList/file_list_100.txt"
#ORIGINAL_FILE_LIST="${ORIGINAL_FITS_SCRATCH_DIR}/file_list.txt"
#SCALING_PARAMS_PATH="/p/scratch/pasta/production_run/CNN_Python/scaling_params.pt" # Correct path
SCRIPT_DIR="/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN"
SCRIPT_NAME="ResNet3D_100.py"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
VENV_PATH="/p/scratch/pasta/CNN/.venv/bin/activate" # Path to your venv activation script
SAVE_DIR="${SCRIPT_DIR}/${TRAINING_JOB_ID}_model_checkpoints" # Where checkpoints are saved/loaded from

# --- Checkpoint Configuration ---
# Set this variable to the specific checkpoint file you want to resume from.
# Leave empty ("") to train from scratch.
# Example: RESUME_CHECKPOINT_PATH="${SAVE_DIR}/checkpoint_epoch_60.pth"
#RESUME_CHECKPOINT_PATH="${SAVE_DIR}/final_model_state.pth" # Set to path or leave empty
RESUME_CHECKPOINT_PATH=""

# --- Define Modules to Load ---
MODULES_TO_LOAD="GCC PyTorch torchvision Python" # List necessary modules

echo "--- Executing Python within Job ${JOB_ID} ---"
echo "Using Node-Local Data: ${NODE_LOCAL_DIR}"
echo "Using Script: ${SCRIPT_PATH}"
echo "CPUs for srun step: ${CPUS_FOR_PYTHON}"
echo "Loading Modules: ${MODULES_TO_LOAD}"
echo "Activating Venv: ${VENV_PATH}"
date

# Execute srun, wrapping the python command in a subshell that loads modules and activates the venv
srun --jobid=${JOB_ID} --cpu-bind=none --ntasks=1 --cpus-per-task=${CPUS_FOR_PYTHON} \
bash -c "
echo '--- Inside srun subshell ---'
echo 'Loading modules...'
module load ${MODULES_TO_LOAD} || { echo 'ERROR: Failed to load modules inside srun'; exit 1; }
echo 'Modules loaded.'
echo 'Activating venv...'
source \"${VENV_PATH}\" || { echo 'ERROR: Failed to activate venv inside srun'; exit 1; }
echo 'Venv activated.'
echo 'Running python: $(which python)' # Verify python path
python \"${SCRIPT_PATH}\" \\
    --data-dir \"${NODE_LOCAL_DIR}\" \\
    --original-file-list \"${ORIGINAL_FILE_LIST_PATH}\" \\
    --num-epochs 60 \\
    --wavelength-stride 1 \\
    --batch-size 32 \\
    --num-workers 16 \\
    --model-depth 34 \\
    --normalization-method 'minmax' \\
    --log-scale-params Dens Lum \\
    --job-id ${JOB_ID} 

echo 'Python script finished.'
" | tee slurm_output/100_files/${JOB_ID}.log
#" > "slurm_output/cnn_run.log" 2>&1 # End of bash -c command string

# Capture the exit code of srun itself
SRUN_EXIT_CODE=$?
echo "--- srun command finished with exit code: ${SRUN_EXIT_CODE} ---"
date

exit $SRUN_EXIT_CODE
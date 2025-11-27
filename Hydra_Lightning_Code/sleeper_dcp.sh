#!/usr/bin/env bash
#SBATCH --job-name=data_stage_sleep
#SBATCH --output=slurm_output/sleeper/sleeper_dcp_%j.out
#SBATCH --error=slurm_output/sleeper/sleeper_dcp_%j.err
#SBATCH --partition=dc-hwai
#SBATCH --account=westai0043
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8  # Allocate enough CPUs for the eventual Python run
#SBATCH --mem=128G         # Allocate enough memory
#SBATCH --gres=gpu:4       # Allocate GPU
#SBATCH --time=24:00:00    # Long enough for copy + debugging

echo "--- Loading Environment ---"
VENV_PATH="/p/scratch/pasta/CNN/.venv/bin/activate"

if [ -f "$VENV_PATH" ]; then source "$VENV_PATH"; echo "Activated venv"; else echo "Error: Venv not found"; exit 1; fi
echo "Loading modules..."
#module load GCC PyTorch torchvision Python Stages/2025  GCC/13.3.0  OpenMPI/5.0.5 mpifileutils/0.11.1 parallel/20240722 # Add other necessary modules
echo "Modules loaded."
echo "which dcp $(which dcp)"
echo "which mpirun $(which mpirun)"

echo "--- Job Info ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# --- Define Paths ---
SRC="/p/scratch/pasta/production_run/Data_14.10.25_mass_models/"
LIST="$(realpath arcsec_files.txt)"
#ORIGINAL_FITS_SCRATCH_DIR="/p/scratch/pasta/CNN/17.03.25/Data_100/"
NODE_LOCAL_DIR="/local/nvme/${SLURM_JOB_ID}_fits_data"
# Flag file to indicate copy completion
COPY_DONE_FLAG="${NODE_LOCAL_DIR}/.copy_done"



echo "--- Staging Data to Node-Local Storage (using dcp and pre-generated dfind list) ---"
echo "Source directory: ${SRC}"
echo "Node-local directory: ${NODE_LOCAL_DIR}"


# Conditional copy: Only copy if the flag file doesn't exist
if [ ! -f "${COPY_DONE_FLAG}" ]; then
    echo "Creating directory ${NODE_LOCAL_DIR}..."
    mkdir -p -v "${NODE_LOCAL_DIR}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create node-local directory ${NODE_LOCAL_DIR}"
        exit 1
    fi
    echo "Directory ${NODE_LOCAL_DIR} created or already exists."

    echo "Starting dcp..."
    TARGET_DIR_ARG="${NODE_LOCAL_DIR}"
    echo "Command to be used: dcp -v -p -i List \"${TARGET_DIR_ARG}\""
    start_copy_time=$(date +%s)


    mpirun -np 8 dcp -v -p "$SRC" "$TARGET_DIR_ARG"
    
    dcp_exit_code=$?
    end_copy_time=$(date +%s)
    copy_duration=$((end_copy_time - start_copy_time))

    if [ $dcp_exit_code -ne 0 ]; then
        echo "Error: dcp failed with code ${dcp_exit_code} using pre-generated dfind list."
        echo "Cleaning up potentially incomplete copy in ${NODE_LOCAL_DIR}..."
        rm -rf "${NODE_LOCAL_DIR}" # Clean up failed attempt
        exit 1
    fi

    echo "Data copy finished via dcp in ${copy_duration} seconds."

    # --- Verification ---
    echo "Checking total disk usage on worker node"
    # Check total size
    du -sh "${NODE_LOCAL_DIR}"


    # Create flag file to indicate successful completion
    echo "Creating copy completion flag: ${COPY_DONE_FLAG}"
    touch "${COPY_DONE_FLAG}"

else
    echo "Data already exists (found ${COPY_DONE_FLAG}). Skipping copy."
    du -sh "${NODE_LOCAL_DIR}"
fi



echo "--- Data Staging Complete. Job is now sleeping. ---"

# Notify the submitter terminal that the copy phase is done
if [ -n "$SLURM_JOB_USER" ]; then
    scontrol show job $SLURM_JOB_ID | grep -q "UserId=$SLURM_JOB_USER"
    if [ $? -eq 0 ]; then
        echo "Sending notification to terminal of user $SLURM_JOB_USER..."
        echo -e "\n>>> [Job ${SLURM_JOB_ID}] Data copy completed successfully on $(hostname) <<<\n" | wall -n
    fi
fi


echo "To run your script, use:"
echo "srun --jobid=$SLURM_JOB_ID --cpu-bind=none --ntasks=1 --cpus-per-task=<CPUs_for_Python> python /path/to/CNN_implementation.py --data-dir ${NODE_LOCAL_DIR} ..."
echo "Or alternatively"
echo "bash run_CNN_hydra.sh --job_id $SLURM_JOB_ID"
echo "(Remember to scancel $SLURM_JOB_ID when finished)"

# Sleep indefinitely (or for a very long time)
sleep infinity


# Alternatively: sleep 86400 # Sleep for 24 hours

echo "--- Sleep finished (or job terminated) ---"
# Cleanup happens automatically when job ends, but could be explicit if needed
# echo "Cleaning up ${NODE_LOCAL_DIR}"
# rm -rf "${NODE_LOCAL_DIR}"
exit 0
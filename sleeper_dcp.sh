#!/usr/bin/env bash
#SBATCH --job-name=data_stage_sleep
#SBATCH --output=slurm_output/sleeper_dcp.out
#SBATCH --error=slurm_output/sleeper_dcp.err
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
ORIGINAL_FITS_SCRATCH_DIR="/p/scratch/pasta/CNN/17.03.25/Data/"
#ORIGINAL_FITS_SCRATCH_DIR="/p/scratch/pasta/CNN/17.03.25/Data_100/"
NODE_LOCAL_DIR="/local/nvme/${SLURM_JOB_ID}_fits_data"
# Flag file to indicate copy completion
COPY_DONE_FLAG="${NODE_LOCAL_DIR}/.copy_done"

# --- Name of the pre-generated dfind list (MPIFileUtils format) ---
#PREGENERATED_DFIND_LIST_FILENAME="fits_file_list.mpi.bin" # Must match the name used in Step 1
#PREGENERATED_DFIND_LIST_PATH="${ORIGINAL_FITS_SCRATCH_DIR}${PREGENERATED_DFIND_LIST_FILENAME}"
# File pattern for filtering and verification
FILE_PATTERN="*arcsec.fits"

echo "--- Staging Data to Node-Local Storage (using dcp and pre-generated dfind list) ---"
echo "Source directory: ${ORIGINAL_FITS_SCRATCH_DIR}"
echo "Node-local directory: ${NODE_LOCAL_DIR}"
#echo "Using pre-generated dfind list: ${PREGENERATED_DFIND_LIST_PATH}"

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
    #echo "Command: dcp --progress 60 \"${PREGENERATED_DFIND_LIST_PATH}\" \"${TARGET_DIR_ARG}\"" #-p --progress 60
    echo "Command: dcp -v -p \"${ORIGINAL_FITS_SCRATCH_DIR}\" \"${TARGET_DIR_ARG}\""
    start_copy_time=$(date +%s)
    #mpirun dcp -v --progress 60 "${PREGENERATED_DFIND_LIST_PATH}" "${TARGET_DIR_ARG}" #
    mpirun -np 8 dcp -v -p ${ORIGINAL_FITS_SCRATCH_DIR} ${TARGET_DIR_ARG}
    
    dcp_exit_code=$?
    end_copy_time=$(date +%s)
    copy_duration=$((end_copy_time - start_copy_time))

    if [ $dcp_exit_code -ne 0 ]; then
        echo "Error: dcp failed with code ${dcp_exit_code} using pre-generated dfind list."
        echo "Cleaning up potentially incomplete copy in ${NODE_LOCAL_DIR}..."
        rm -rf "${NODE_LOCAL_DIR}" # Clean up failed attempt
        exit 1
    fi

    echo "Data copy finished via dcp (pre-gen list) in ${copy_duration} seconds."

    # --- Verification ---
    echo "Running verification..."
    # 1. Check total size
    du -sh "${NODE_LOCAL_DIR}"

    # 2. Basic structure check
    #first_data_subdir_name=$(find "${ORIGINAL_FITS_SCRATCH_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'Dens=*' -print -quit | xargs basename)
    #verification_passed=true
    #if [ -n "$first_data_subdir_name" ]; then
    #     if ! find "${NODE_LOCAL_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'Dens=*' -print -quit | grep -q .; then
    #          echo "Error: Verification failed - No subdirectories matching 'Dens=*' found in ${NODE_LOCAL_DIR}."
    #          verification_passed=false
    #     else
    #          echo "Verification: Found expected subdirectory pattern 'Dens=*'."
    #     fi
    #else
    #    echo "Verification: No 'Dens=*' subdirectory found in source to verify against."
    #fi

    # 3. Count copied files
    # We can't easily know the expected count without the text list,
    # but we check that *some* files matching the pattern were copied.
    echo "Verifying copied file count..."
    copied_files_count=$(find "${NODE_LOCAL_DIR}" -type f -name "${FILE_PATTERN}" | wc -l)
    if [ "$copied_files_count" -eq 0 ]; then
        # This implies dcp failed to copy anything despite a non-empty list
        echo "Error: Verification failed - Copied file count in destination is 0, but dfind list was non-empty."
        verification_passed=false
    else
        echo "Verification: Found ${copied_files_count} copied '${FILE_PATTERN}' files in destination."
        # If you generated the text list in Step 1, you could copy *that* list name here
        # and uncomment the comparison logic if needed.
        # PREGENERATED_TXT_LIST_PATH="${ORIGINAL_FITS_SCRATCH_DIR}/fits_file_list.txt"
        # if [ -f "${PREGENERATED_TXT_LIST_PATH}" ]; then
        #   expected_count=$(wc -l < "${PREGENERATED_TXT_LIST_PATH}")
        #   if [ "$copied_files_count" -ne "$expected_count" ]; then ... error ... fi
        # fi
    fi

    if [ "$verification_passed" = false ]; then
        echo "Cleaning up failed verification in ${NODE_LOCAL_DIR}..."
        rm -rf "${NODE_LOCAL_DIR}"
        exit 1
    fi
    # --- End Verification ---


    # Create flag file to indicate successful completion
    echo "Creating copy completion flag: ${COPY_DONE_FLAG}"
    touch "${COPY_DONE_FLAG}"

else
    echo "Data already exists (found ${COPY_DONE_FLAG}). Skipping copy."
    du -sh "${NODE_LOCAL_DIR}"
fi



echo "--- Data Staging Complete. Job is now sleeping. ---"
echo "To run your script, use:"
echo "srun --jobid=$SLURM_JOB_ID --cpu-bind=none --ntasks=1 --cpus-per-task=<CPUs_for_Python> python /path/to/CNN_implementation.py --data-dir ${NODE_LOCAL_DIR} ..."
echo "(Remember to scancel $SLURM_JOB_ID when finished)"

# Sleep indefinitely (or for a very long time)
sleep infinity
# Alternatively: sleep 86400 # Sleep for 24 hours

echo "--- Sleep finished (or job terminated) ---"
# Cleanup happens automatically when job ends, but could be explicit if needed
# echo "Cleaning up ${NODE_LOCAL_DIR}"
# rm -rf "${NODE_LOCAL_DIR}"
exit 0
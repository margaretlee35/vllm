#!/bin/bash
#SBATCH -J epd_compare  # Job name
#SBATCH -o ./epdtest/slurm_logs/epd_compare.o    # Name of stdout output file
#SBATCH -e ./epdtest/slurm_logs/epd_compare.e    # Name of stderr error file
#SBATCH -p gpu-a100-dev  # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A ASC23036        # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=sylee@utexas.edu

# Any other commands must follow all #SBATCH directives...
source "$HOME/.bashrc"
cd $WORK/git_repos/vllm
source .venv/bin/activate

mkdir -p ./epdtest/slurm_logs

RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
export RUN_STAMP

# --- Run ---
LMMS_RC=0
SWEEP_RC=0

bash ./epdtest/lmms_compare.sh || LMMS_RC=$?
bash ./epdtest/sweep_compare.sh || SWEEP_RC=$?

# Always summarize logs at the end.
bash ./epdtest/summarize_compare_logs.sh --run-stamp "$RUN_STAMP" || true

if [[ "$LMMS_RC" -ne 0 || "$SWEEP_RC" -ne 0 ]]; then
    echo "epd_compare completed with failures: lmms_rc=$LMMS_RC sweep_rc=$SWEEP_RC" >&2
    exit 1
fi

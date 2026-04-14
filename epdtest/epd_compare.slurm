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
vllm
unalias vllm

mkdir -p ./epdtest/slurm_logs

# --- Run ---
bash ./epdtest/lmms_compare.sh
bash ./epdtest/sweep_compare.sh

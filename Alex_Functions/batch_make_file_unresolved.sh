#!/bin/bash -l

#!/bin/bash
#SBATCH -p regular
#SBATCH -o logs/tracer_snap_unresolved
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=haswell
# load nbodykit
#SBATCH --mail-type=END    # notifications for job
#SBATCH --mail-user=cameron.grove@durham.ac.uk


conda activate halo_env

python3 tracer_snapshot_unresolved.py

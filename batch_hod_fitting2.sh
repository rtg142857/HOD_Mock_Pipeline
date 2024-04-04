#!/bin/bash -l

#!/bin/bash
#SBATCH -q regular
#SBATCH -o logs/tracer_snap2
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -C cpu

#conda activate halo_env
#source activate abacus-p
cosmodesienv main

module load gcc
module load gsl
module unload craype-hugepages2M

# Only do the halo paircounting in this step
python cencen.py
python censat.py
python satsat.py
python satsat_onehalo.py

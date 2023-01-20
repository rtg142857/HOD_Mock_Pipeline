#!/bin/bash -l

#!/bin/bash
#SBATCH -q regular
#SBATCH -o logs/tracer_snap
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -C cpu

conda activate halo_env

module load gcc
module load gsl
module unload craype-hugepages2M

python3 tracer_snapshot.py
python3 tracer_snapshot_unresolved.py

cd FastHodFitting/paircounting/

# Only do the halo paircounting in this step
python cencen.py
python censat.py
python satsat.py
python satsat_onehalo.py


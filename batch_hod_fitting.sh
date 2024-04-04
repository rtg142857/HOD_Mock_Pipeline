#!/bin/bash -l

#!/bin/bash
#SBATCH -q regular
#SBATCH -o logs/tracer_snap
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -C cpu
#SBATCH --mail-user=tlrt88@durham.ac.uk

#conda activate halo_env
#source activate abacus-p

#cosmodesienv main
conda activate abacus-env

module load gcc
module load gsl
module unload craype-hugepages2M

python3 tracer_snapshot.py
python3 tracer_snapshot_unresolved.py

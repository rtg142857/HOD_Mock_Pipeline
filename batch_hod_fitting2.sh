#!/bin/bash -l

#!/bin/bash
#SBATCH -q regular
#SBATCH -o logs/tracer_snap2
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -C cpu

conda activate halo_env

module load gcc
module load gsl
module unload craype-hugepages2M

cd FastHodFitting/fitting_smoothed_curves_nersc/


# Run the HOD fitting with several initial random seeds
python fasthod_fitting.py 1
python fasthod_fitting.py 2
python fasthod_fitting.py 3
python fasthod_fitting.py 4
python fasthod_fitting.py 5


#!/bin/bash -l

#!/bin/bash
#SBATCH -q regular
#SBATCH -o logs/make_mock_2
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -C cpu

conda activate halo_env

module load gcc
module load gsl
module unload craype-hugepages2M

python make_mocks_script_2.py


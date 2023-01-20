#!/bin/bash -l

#!/bin/bash
#SBATCH -q shared
#SBATCH -o logs/make_mock_hdf5
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G
#SBATCH -C cpu

conda activate halo_env

module load gcc
module load gsl
module unload craype-hugepages2M

python make_mocks_script_hdf5.py


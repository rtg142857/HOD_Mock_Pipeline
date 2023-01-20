#!/bin/bash -l

#!/bin/bash
#SBATCH -q shared
#SBATCH -o logs/make_mock
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G
#######SBATCH --constraint=haswell
#SBATCH -C cpu
# load nbodykit
#SBATCH --mail-type=ALL    # notifications for job
#SBATCH --mail-user=cameron.grove@durham.ac.uk


conda activate halo_env

module load gcc
module load gsl
module unload craype-hugepages2M

python3 make_mocks_script.py


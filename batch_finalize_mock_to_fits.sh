#!/bin/bash -l

#!/bin/bash
#SBATCH -q shared
#SBATCH -o logs/make_final_mock
#SBATCH --time=20
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=20G
#SBATCH -C cpu

conda activate halo_env
source /global/cfs/cdirs/desi/software/desi_environment.sh main

module load gcc
module load gsl
module unload craype-hugepages2M


rm galaxy_catalogue/final/galaxy_full_sky.fits
python finalize_mock_to_fits.py


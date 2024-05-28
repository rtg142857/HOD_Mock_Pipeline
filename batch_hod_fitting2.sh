#!/bin/bash -l

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J paircounting #Give it something meaningful.
#SBATCH -o logs/tracer_snap2
#SBATCH -e logs/tracer_snap2_error
#SBATCH -p cosma8 #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 720
#SBATCH --mail-type=END # notifications for job done & fail
#SBATCH --mail-user=tlrt88@durham.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#cosmodesienv main
#conda activate abacus-env
module use /cosma/home/dp004/dc-mene1/software/desi/cosmodesiconda/my-desiconda/modulefiles
module load cosmodesiconda/my-desiconda

#module load gcc
#module load gsl
#module unload craype-hugepages2M

module load python/3.10.12

# Only do the halo paircounting in this step
python cencen.py
python censat.py
python satsat.py
python satsat_onehalo.py

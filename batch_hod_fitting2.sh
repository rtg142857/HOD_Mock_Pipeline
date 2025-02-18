#!/bin/bash -l

# First argument is the path to the path_config.yml file

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J paircounting #Give it something meaningful.
#SBATCH -o logs/out_paircounting
#SBATCH -e logs/err_paircounting
#SBATCH -p cosma8 #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 1440
#SBATCH --mail-type=ALL # notifications for job done & fail
#SBATCH --mail-user=tlrt88@durham.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
module use /cosma/apps/dp004/dc-mene1/desi/cosmodesiconda/my-desiconda/modulefiles
module load cosmodesiconda/my-desiconda
#source /cosma/home/dp004/dc-mene1/use-cosmodesiconda

# Only do the halo paircounting in this step
#python cencen.py $1
#python censat.py $1
#python satsat.py $1
#python satsat_onehalo.py $1
python all_pairs.py $1
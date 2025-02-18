#!/bin/bash -l

# First argument is the path to the path_config.yml file

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J tracer_making #Give it something meaningful.
#SBATCH -o logs/out_tracers
#SBATCH -e logs/err_tracers
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

python tracer_snapshot.py $1
python tracer_snapshot_unresolved.py $1

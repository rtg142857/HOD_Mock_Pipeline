#!/bin/bash -l

# First argument is the path to the path_config.yml file

#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J HOD_pipeline_testing #Give it something meaningful.
#SBATCH -o logs/tracer_snap
#SBATCH -e logs/tracer_snap_error
#SBATCH -p cosma8 #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 720
#SBATCH --mail-type=ALL # notifications for job done & fail
#SBATCH --mail-user=tlrt88@durham.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
#cosmodesienv main
#conda activate abacus-env
module use /cosma/apps/dp004/dc-mene1/desi/cosmodesiconda/my-desiconda/modulefiles
module load cosmodesiconda/my-desiconda
#source /cosma/home/dp004/dc-mene1/activate-cosmodesiconda

#module load gcc
#module load gsl
#module unload craype-hugepages2M

#module load python/3.10.12

python tracer_snapshot.py $1
python tracer_snapshot_unresolved.py $1

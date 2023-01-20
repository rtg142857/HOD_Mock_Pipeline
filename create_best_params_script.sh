#!/bin/bash -l

cosmo_number=13
phase_number=0

cd "halo_fitting_${cosmo_number}_${phase_number}/shared_code/abacus_mocks/"


cp ../../../find_best_params.py ./

conda activate halo_env

python find_best_params.py ../../FastHodFitting/fitting_smoothed_curves_nersc/


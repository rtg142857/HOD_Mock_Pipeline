Code for HOD fitting and mock creation from the AbacusSummit catalogues

# HOD Fitting

create_new_HOD_fit.sh will make HOD fits for a particular cosmology/phase AbacusSummit box
The first couple of lines specify the cosmology and phase numbers to use

The script will queue two SLURM jobs, one to produce tabulated halo paircounts, another to use them to fit HOD parameters

All data relevant to a particular HOD fit will be stored under a folder with a name in the following format:
halo_fitting_{cosmo_number}_{phase_number}/

If the run is successful then you should see a bunch of files in locations: halo_fitting_{cosmo_number}_{phase_number}/FastHodFitting/fitting_smoothed_curves_nersc/diff_start_low_prior_*


If you wish to alter the target data or assumed errors in the fitting process, then it is probably best to fork https://github.com/cejgrove/FastHodFitting

Then alter lines 87-111 of config_fitting.py (e.g https://github.com/cejgrove/FastHodFitting/blob/master/fitting_smoothed_curves_nersc/config_fitting.py) to use new files or functions

Then point create_new_HOD_fit.sh to clone your fork instead of my original repository.




# Mock Creation

Mock creation is performed by running make_mock_bash.sh. As before, cosmology and phase numbers are specified at the top of the file

This script should be run after create_new_HOD_fit.sh has successfully finished and assumes the directory structure is the same as that produced by create_new_HOD_fit.sh. 

Mock galaxy catalogue files will be produced under:
halo_fitting_{cosmo_number}_{phase_number}/shared_code/abacus_mocks/galaxy_catalogue/final/

If creating the mock catalogue failed (due to slurm issues, hitting a time limit, etc.), then the directories can be cleaned and repeated attempts made by running make_mock_bash_remake.sh with the cosmology and phsae numbers specified at the beginning of the file





# Notes

The following repositories are cloned as part of the pipeline:
https://github.com/cejgrove/FastHodFitting
https://github.com/amjsmith/shared_code/

The pipeline currently uses a conda environment called halo_env, this contains all the relevant packages for running the code (abacusutils, corrfunc, nbodykit,...). If you have your own environment for such packages then you can find and replace "conda activate halo_env" with "conda activate {your environment name}" in all files.


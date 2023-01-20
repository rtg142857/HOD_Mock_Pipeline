#!/bin/bash -l
# Script to create a mock after HOD fits have been made
# Change these to use a different cosmology/phase number
cosmo_number=0
phase_number=1

# Only make a mock if the corresponding HOD fit directory is present
if ! [ -d "halo_fitting_${cosmo_number}_${phase_number}/shared_code/abacus_mocks/" ]
then
	echo "No Directory"
	exit 9999
fi

# Move into that directory and make some folders
cd "halo_fitting_${cosmo_number}_${phase_number}/shared_code/abacus_mocks/"

mkdir galaxy_catalogue
mkdir galaxy_catalogue/final
mkdir logs
mkdir lookup/mass_functions
mkdir lookup/particles


# Copy scripts from the original directory
# replace the cosmology and phase numbers with those specified at the top of the file
cp ../../../make_mocks_script_1.py ./

sed -i "s/cosmo=0/cosmo=${cosmo_number}/" make_mocks_script_1.py

sed -i "s/phase=0/phase=${phase_number}/" make_mocks_script_1.py



cp ../../../make_mocks_script_2.py ./

sed -i "s/cosmo=0/cosmo=${cosmo_number}/" make_mocks_script_2.py

sed -i "s/phase=0/phase=${phase_number}/" make_mocks_script_2.py




cp ../../../make_mocks_script_hdf5.py ./

sed -i "s/cosmo=0/cosmo=${cosmo_number}/" make_mocks_script_hdf5.py

sed -i "s/phase=0/phase=${phase_number}/" make_mocks_script_hdf5.py


cp ../../../finalize_mock.py ./

cp ../../../batch_make_mock.sh ./

cp ../../../batch_make_mock_1.sh ./

cp ../../../batch_make_mock_2.sh ./

cp ../../../batch_make_mock_hdf5.sh ./

cp ../../../batch_finalize_mock.sh ./

cp ../../../batch_finalize_mock_to_fits.sh ./

cp ../../../finalize_mock_to_fits.py ./

cp ../../../find_best_params.py ./

cp ../../../make_catalogue_snapshot.py ./

# Activate conda environment to get all the right packages
conda activate halo_env

# get the best HOD parameters from the multiple fitting runs and copy into the location for the mock creation code to use
python find_best_params.py ../../FastHodFitting/fitting_smoothed_curves_nersc/

cp best_params.txt lookup/hod_fits/best_params.txt

# run the jobs as dependencies, each step only starts after the previous one finishes
jid1=$(sbatch --parsable batch_make_mock_1.sh)
jid2=$(sbatch  --dependency=afterany:$jid1 --parsable batch_make_mock_2.sh)
jid3=$(sbatch  --dependency=afterany:$jid2 --parsable batch_make_mock_hdf5.sh)
jid4=$(sbatch  --dependency=afterany:$jid3 --parsable batch_finalize_mock.sh)
jid5=$(sbatch  --dependency=afterany:$jid4 --parsable batch_finalize_mock_to_fits.sh)


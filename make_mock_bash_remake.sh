#!/bin/bash -l
# Same as make_mock_bash but can be used if a run fails
cosmo_number=0
phase_number=0

if ! [ -d "halo_fitting_${cosmo_number}_${phase_number}/shared_code/abacus_mocks/" ]
then
	echo "No Directory"
	exit 9999
fi


cd "halo_fitting_${cosmo_number}_${phase_number}/shared_code/abacus_mocks/"

rm galaxy_catalogue/final/*
rm galaxy_catalogue/*


cp ../../../make_mocks_script.py ./

sed -i "s/cosmo=0/cosmo=${cosmo_number}/" make_mocks_script.py

sed -i "s/phase=0/phase=${phase_number}/" make_mocks_script.py


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

conda activate halo_env

jid1=$(sbatch --parsable batch_make_mock_1.sh)
jid2=$(sbatch  --dependency=afterany:$jid1 --parsable batch_make_mock_2.sh)
jid3=$(sbatch  --dependency=afterany:$jid2 --parsable batch_make_mock_hdf5.sh)
jid4=$(sbatch  --dependency=afterany:$jid3 --parsable batch_finalize_mock.sh)
jid5=$(sbatch  --dependency=afterany:$jid4 --parsable batch_finalize_mock_to_fits.sh)


#!/bin/bash -l

L=1000
N=1800
simulation="DMO_FIDUCIAL"

mkdir "halo_fitting_L${L}N${N}_${simulation}"

cd "halo_fitting_L${L}N${N}_${simulation}"

mkdir logs

cp ../Alex_Functions/*.py ./
cp ../Alex_Functions/*.csv ./
cp ../Alex_Functions/*.dat ./
cp ../Alex_Functions/*.sh ./
cp ../batch_hod_fitting.sh ./
cp ../batch_hod_fitting2.sh ./

cp ../rescaling_code/ -r ./

sed -i "s/cosmo_number = 0/cosmo_number = ${cosmo_number}/" rescaling_code/xi_rescaling_factor.py

sed -i "s/cosmo_number = 0/cosmo_number = ${cosmo_number}/" rescaling_code/luminosity_function.py

sed -i "s/simulation = \"DMO_FIDUCIAL\"/simulation = \"${simulation}\"/" tracer_snapshot.py

sed -i "s/L = 100/L = ${L}/" tracer_snapshot.py

sed -i "s/N = 180/N = ${N}/" tracer_snapshot.py

sed -i "s/simulation = \"DMO_FIDUCIAL\"/simulation = \"${simulation}\"/" tracer_snapshot_unresolved.py

sed -i "s/L = 100/L = ${L}/" tracer_snapshot_unresolved.py

sed -i "s/N = 180/N = ${N}/" tracer_snapshot_unresolved.py

#conda activate halo_env
#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module use /cosma/home/dp004/dc-mene1/software/desi/cosmodesiconda/my-desiconda/modulefiles
module load cosmodesiconda/my-desiconda

#python rescaling_code/xi_rescaling_factor.py

#python rescaling_code/luminosity_function.py

git clone --depth 1 "https://github.com/amjsmith/FastHodFitting"

git clone --depth 1 "https://github.com/amjsmith/shared_code/"

#cp cosmology_rescaling_factor_xi_lin_8.txt FastHodFitting/fitting_smoothed_curves_nersc/cosmology_rescaling_factor_xi_lin_8.txt
#cp target_num_den_rescaled.txt FastHodFitting/fitting_smoothed_curves_nersc/target_num_den_rescaled.txt
#cp -r ../FastHodFitting/ ./

jid1=$(sbatch --parsable batch_hod_fitting.sh)
jid2=$(sbatch  --dependency=afterany:$jid1 --parsable batch_hod_fitting2.sh)

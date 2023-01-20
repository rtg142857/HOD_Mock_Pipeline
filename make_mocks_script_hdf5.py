### Script for making cubic box and cut-sky mocks from the AbacusSummit simulations

import numpy as np
import sys
import os
import h5py
from hodpy.cosmology import CosmologyMXXL, CosmologyAbacus
from hodpy.hod_bgs_snapshot_abacus import HOD_BGS
from make_catalogue_snapshot import *
import fitsio

###################################################
# various variables to set, depending on the simulation we are using

cosmo=0   # AbacusSummit cosmology number
simulation="base"
phase=0
Lbox = 2000. # box size (Mpc/h)
snapshot_redshift = 0.2
Nfiles = 34  # number of files the simulation is split into

zmax_low = 0.15 # maximum redshift of low-z faint lightcone
zmax = 0.6      # maximum redshift of lightcone
mass_cut = 11   # mass cut between unresolved+resolved low z lightcone

# observer position in box (in Mpc/h)
# Note that in Abacus, position coordinates in box go from -Lbox/2 < x < Lbox/2
# so an observer at the origin is in the centre of the box
#observer=(0,0,0) 
observer=(-1000,-1000,-1000)

cosmology = CosmologyAbacus(cosmo)
cosmology_mxxl = CosmologyMXXL()
SODensity=304.64725494384766

mag_faint_snapshot  = -18 # faintest absolute mag when populating snapshot
mag_faint_lightcone = -10 # faintest absolute mag when populating low-z faint lightcone
app_mag_faint = 20.2 # faintest apparent magnitude for cut-sky mock

### locations of various files
lookup_path = "lookup/"
hod_param_file = lookup_path+"hod_fits/best_params.txt" # Results of HOD fitting

# these files will be created if they don't exists
# (lookup files for efficiently assigning cen/sat magnitudes, and fit to halo mass function)
# and files of number of field particles in shells around the observer
central_lookup_file   = lookup_path+"central_magnitudes_c%03d_test.npy"%cosmo
satellite_lookup_file = lookup_path+"satellite_magnitudes_c%03d_test.npy"%cosmo
mf_fit_file           = lookup_path+"mass_functions/mf_c%03d.txt"%cosmo

Nparticle             = lookup_path+"particles/N_c%03d_ph%03d.dat"%(cosmo,phase)
Nparticle_shell       = lookup_path+"particles/Nshells_c%03d_ph%03d.dat"%(cosmo, phase)


# input path of the simulation
mock = "AbacusSummit_%s_c%03d_ph%03d"%(simulation, cosmo, phase)
if simulation=="small":
    abacus_path = "/global/cfs/cdirs/desi/cosmosim/Abacus/small/"
else:
    abacus_path = "/global/cfs/cdirs/desi/cosmosim/Abacus/"

# output path to save temporary files
output_path = "galaxy_catalogue/"
# output path to save the final cubic box and cut-sky mocks
output_path_final = "galaxy_catalogue/final/"

# file names
galaxy_snapshot_file   = "galaxy_snapshot_%i.hdf5"
halo_lightcone_unres   = "halo_lightcone_unresolved_%i.hdf5"
galaxy_lightcone_unres = "galaxy_lightcone_unresolved_%i.hdf5"
galaxy_lightcone_res   = "galaxy_lightcone_resolved_%i.hdf5"
galaxy_cutsky_low      = "galaxy_cut_sky_low_%i.hdf5"
galaxy_cutsky          = "galaxy_cut_sky_%i.hdf5"

galaxy_cutsky_final   = "galaxy_full_sky.hdf5"
galaxy_snapshot_final = "galaxy_snapshot.hdf5"




###############################################
print("MERGE CUBIC BOX FILES INTO FINAL MOCK")

# join snapshot files together into single file
# use original (not rescaled) magnitudes

merge_box(output_path, galaxy_snapshot_file, output_path_final,
          galaxy_snapshot_final, fmt="hdf5", Nfiles=Nfiles, offset=Lbox/2.)



###############################################
print("MERGE CUT-SKY FILES INTO FINAL MOCK")


# join files together 

merge_lightcone(output_path, galaxy_cutsky, galaxy_cutsky_low, 
                output_path_final, galaxy_cutsky_final, fmt='hdf5',
                Nfiles=Nfiles, zmax_low=zmax_low, app_mag_faint=app_mag_faint)

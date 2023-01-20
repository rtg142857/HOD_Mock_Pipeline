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

galaxy_cutsky_final   = "galaxy_full_sky.fits"
galaxy_snapshot_final = "galaxy_snapshot.fits"


# how many periodic replications do we need for full cubic box to get to zmax?
# n_rep=0 is 1 replication (i.e. just the original box)
# n=1 is replicating at the 6 faces (total of 7 replications)
# n=2 is a 3x3x3 cube of replications, but omitting the corners (19 in total)
# n=3 is the full 3x3x3 cube of replications (27 in total)
rmax = cosmology.comoving_distance(zmax)
rmax_low = cosmology.comoving_distance(zmax_low)
n_rep = replications(Lbox, rmax)


# get fit to halo mass function
print("getting fit to halo mass function")
input_file = abacus_path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"
mass_function = get_mass_function(input_file, mf_fit_file,
                    redshift=snapshot_redshift, box_size=Lbox,
                                  cosmology=cosmology, Nfiles=Nfiles)


################################################
print("MAKING CUBIC BOX MOCK")

for file_number in range(Nfiles):
    print("FILE NUMBER", file_number)

    input_file = abacus_path+mock+"/halos/z%.3f/halo_info/halo_info_%03d.asdf"%(snapshot_redshift, file_number)
    output_file = output_path + galaxy_snapshot_file%file_number

    main(input_file, output_file, snapshot_redshift, mag_faint_snapshot,
         cosmology, hod_param_file, central_lookup_file, satellite_lookup_file,
         mass_function, cosmology_old=cosmology_mxxl)
    

###############################################
print("MAKING LOW Z UNRESOLVED HALO LIGHTCONE")

# this function will loop through the 34 particle files
# will find minimum halo mass needed to make mock to faint app mag limit
# (taking into account scaling of magnitudes by cosmology)
# this includes calculating a fit to the halo mass function, and counting the
# available number of field particles in shells. These will be read from files
# (mf_fit_file, Nparticle, Nparticle_shell) if they exist
# If the files don't exist yet, they will be automatically created
# (but this is fairly slow)
# app mag limit is also shifted slightly fainter than is needed

output_file = output_path+halo_lightcone_unres

halo_lightcone_unresolved(output_file, abacus_path, snapshot_redshift,
                cosmology, hod_param_file, central_lookup_file, satellite_lookup_file,
                mass_function, Nparticle, Nparticle_shell, box_size=Lbox,
                SODensity=SODensity, simulation=simulation, cosmo=cosmo, ph=phase,
                observer=observer, app_mag_faint=app_mag_faint+0.05,
                cosmology_orig=cosmology_mxxl, Nfiles=Nfiles)



###############################################
print("MAKING UNRESOLVED LOW Z GALAXY LIGHTCONE")

# this will populate the unresolved halo lightcone with galaxies

for file_number in range(Nfiles):
        print("FILE NUMBER", file_number)
        input_file = output_path+halo_lightcone_unres%file_number
        output_file = output_path+galaxy_lightcone_unres%file_number

        main_unresolved(input_file, output_file, snapshot_redshift,
                        mag_faint_lightcone, cosmology, hod_param_file, central_lookup_file,
                        satellite_lookup_file, mass_function, SODensity=SODensity,
                        zmax=zmax_low+0.01, log_mass_max=mass_cut,
                        cosmology_old=cosmology_mxxl, observer=observer, box_size=Lbox)





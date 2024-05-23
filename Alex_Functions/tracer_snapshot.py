import h5py
import numpy as np

from halo_catalogue import FlamingoSnapshot
from galaxy_catalogue import GalaxyCatalogueSnapshot
from cosmology import CosmologyFlamingo
from hod_tracer import HOD_Tracer



def make_snapshot_tracers(output_file,
                          L, N, simulation,
                          particles=False, redshift=0.2, ntracer=3, log_mass_min=11):
    """
    Make file of galaxy tracers for a Flamingo simulation snapshot
    Args:
        output_file: path of hdf5 file to save output
        L:           Box length of the simulation (the 350 in e.g. L350N1800_DMO)
        N:           Number of particles in the simulation (the 1800 in e.g. L350N1800_DMO)
        simulation:  Specific version of the simulation (e.g. "DMO_FIDUCIAL", "HYDRO_STRONG_AGN")
        particles:   use particles if True, NFW if False. Default is False; TRUE NOT YET SUPPORTED WITH FLAMINGO
        redshift:    snapshot redshift. Default z=0.2
        ntracer:     number of satellite tracers per halo. Default is 3
        log_mass_min: smallest halo mass to use. Default is logM = 11 Mpc/h

    Old args used in the Abacus mocks:
        file_number: snapshot file number (from 0 to 33)
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        A:           if particles=True, use A particles? Default is True
        B:           if particles=True, use B particles? Default is False
        abacus_cosmologies_file: file of Abacus cosmological parameters
        clean:       use cleaned Abacus halo catalogue? Default is True
    """
    #path = "/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_%s_c%03d_ph%03d/halos/"%(simulation, cosmo, ph)
    #file_name = path+"z%.3f/halo_info/halo_info_%03d.asdf"%(redshift, file_number)
    simulation_path = "/cosma8/data/dp004/flamingo/Runs/L%03dN%03d/"%(L, N) + simulation
    
    print(simulation_path)

    cosmology = CosmologyFlamingo(L, N, simulation)
    
    # read in the halo catalogue
    halo_cat = FlamingoSnapshot(file_name, snapshot_redshift=redshift, cosmology=cosmology, 
                              L=L, particles=particles)
    
    # cut to haloes above minimum mass
    halo_cat.cut(halo_cat.get("mass") >= 10**log_mass_min)
    
    # create empty galaxy catalogue
    gal_cat = GalaxyCatalogueSnapshot(halo_cat, cosmology, L)
    
    # use HOD to add galaxy tracers to the catalogue
    hod = HOD_Tracer(ntracer=ntracer)
    gal_cat.add_galaxies(hod)
    gal_cat.position_galaxies(conc="conc")
    
    # save the galaxy catalogue
    # mass units 1e10 Msun/h 
    # periodic boundary conditions already dealt with when positioning the satellites
    f = h5py.File(output_file, "a")
    f.create_dataset("mass",       data=gal_cat.get_halo("mass")/1e10, compression="gzip") 
    f.create_dataset("position",   data=gal_cat.get("pos"),            compression="gzip")
    f.create_dataset("velocity",   data=gal_cat.get("vel"),            compression="gzip")
    f.create_dataset("is_central", data=gal_cat.get("is_cen"),         compression="gzip")
    f.create_dataset("halo_id",    data=gal_cat.get("halo_ind"),       compression="gzip")
    f.close()
    
    
    
def add_missing_particles(output_file, box_size=2000):
    """
    Adds positions and velocities to particle satellite tracers, where this information is missing.
    For low mass haloes, there are not always enough particles to position the satellite tracers.
    This function uses other haloes, of the same mass, to add in these missing particles
    Args:
        output_file: path of hdf5 file to save output
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
    """
    
    f = h5py.File(output_file,"r")
    log_mass = np.log10(f["halo_mass"])
    pos = f["pos"][...]
    vel = f["vel"][...]
    is_sat = np.invert(f["is_cen"][...])
    cen_ind = f["cen_ind"][...]
    f.close()
    
    # find galaxies that need to be assigned positions/velocities
    v_av = np.sum(vel**2, axis=1)**0.5
    is_missing = v_av==0
    is_not_missing = np.invert(is_missing)
    
    # pos and vel of satellites relative to central
    pos_rel = pos - pos[cen_ind]
    vel_rel = vel - vel[cen_ind]

    # periodic boundary
    pos_rel[pos_rel>100] -= box_size
    pos_rel[pos_rel<-100] += box_size

    # loop through mass bins
    bins = np.arange(10.8,16,0.01)
    Mmax = np.max(log_mass[is_missing])

    for i in range(len(bins)):
        if bins[i] > Mmax: break
    
        idx_missing = np.logical_and.reduce((log_mass>=bins[i], log_mass<bins[i+1], is_missing))
        Nmissing = np.count_nonzero(idx_missing)
        if Nmissing ==0: continue

        print(bins[i])
            
        idx_not_missing = np.logical_and.reduce((log_mass>=bins[i], log_mass<bins[i+1], 
                                                 is_not_missing, is_sat))

        idx_not_missing = np.where(idx_not_missing)[0]
        np.random.shuffle(idx_not_missing)
        idx_not_missing = idx_not_missing[:Nmissing]
        idx_missing = np.where(idx_missing)[0]

        pos_rel[idx_missing] = pos_rel[idx_not_missing]
        vel_rel[idx_missing] = vel_rel[idx_not_missing]
    
    pos = pos[cen_ind] + pos_rel
    vel = vel[cen_ind] + vel_rel
    
    f = h5py.File(output_file,"a")
    f["pos"][...] = pos
    f["vel"][...] = vel
    f.close()
    
    
    
if __name__ == "__main__":

    path = "" #path to save the output files
    output_file = path+"galaxy_tracers_0.hdf5"
    
    # use cleaned halo catalogue
    clean=True
    
    # number of satellite tracers for each halo
    ntracer = 3
    
    # base L0100N0180 DMO_FIDUCIAL simultion snapshot
    simulation = "DMO_FIDUCIAL"
    L = 100
    N = 180

    redshift=

    log_mass_min = 11
    #simulation = "base"
    #cosmo = 0
    #ph = 0
    #box_size = 2000 #Mpc/h
    #redshift = 0.2
    #abacus_cosmologies_file = "abacus_cosmologies.csv"
    
    # for NFW profile
    particles=False
    A = False # doesn't matter what A and B are set to, since particles are not being used
    B = False
    
    # for using A particles
    #particles = True
    #A = True
    #B = False
    
    # loop through the 34 snapshot files, adding tracers, and saving the output to a file
    #for i in range(34):
    make_snapshot_tracers(output_file, L=L, N=N, simulation=simulation,
                              particles=False, redshift=redshift,
                              ntracer=ntracer, log_mass_min=log_mass_min)
        
        # add_missing_particles(output_file%i, box_size=box_size)

import h5py
import numpy as np

from halo_catalogue import AbacusSnapshot
from galaxy_catalogue import GalaxyCatalogueSnapshot
from cosmology import CosmologyAbacus
from hod_tracer import HOD_Tracer



def make_snapshot_tracers(file_number, output_file, clean=True, particles=False, redshift=0.2,
                          simulation="base", box_size=2000, cosmo=0, ph=0, A=True, B=False,
                          ntracer=3, log_mass_min=11, abacus_cosmologies_file="abacus_cosmologies.csv"):
    """
    Make file of galaxy tracers for an Abacus simulation snapshot
    Args:
        file_number: snapshot file number (from 0 to 33)
        output_file: path of hdf5 file to save output
        clean:       use cleaned Abacus halo catalogue? Default is True
        particles:   use particles if True, NFW if False. Default is False
        redshift:    snapshot redshift. Default z=0.2
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        A:           if particles=True, use A particles? Default is True
        B:           if particles=True, use B particles? Default is False
        ntracer:     number of satellite tracers per halo. Default is 3
        log_mass_min: smallest halo mass to use. Default is logM = 11 Mpc/h
        abacus_cosmologies_file: file of Abacus cosmological parameters
    """
    path = "/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_%s_c%03d_ph%03d/halos/"%(simulation, cosmo, ph)
    file_name = path+"z%.3f/halo_info/halo_info_%03d.asdf"%(redshift, file_number)
    
    print(file_name)
    
    cosmology = CosmologyAbacus(cosmo, abacus_cosmologies_file)
    
    # read in the halo catalogue
    halo_cat = AbacusSnapshot(file_name, snapshot_redshift=redshift, cosmology=cosmology, 
                              box_size=box_size, clean=clean, particles=particles, A=A, B=B)
    
    # cut to haloes above minimum mass
    halo_cat.cut(halo_cat.get("mass") >= 10**log_mass_min)
    
    # create empty galaxy catalogue
    gal_cat = GalaxyCatalogueSnapshot(halo_cat, cosmology, box_size)
    
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
    output_file = path+"galaxy_tracers_%i.hdf5"
    
    # use cleaned halo catalogue
    clean=True
    
    # number of satellite tracers for each halo
    ntracer = 3
    
    # base c000_ph000 simultion snapshot at z=0.2
    simulation = "base"
    cosmo = 0
    ph = 0
    box_size = 2000 #Mpc/h
    redshift = 0.2
    log_mass_min = 11 # set the minimim halo mass we want
    abacus_cosmologies_file = "abacus_cosmologies.csv"
    
    # for NFW profile
    particles=False
    A = False # doesn't matter what A and B are set to, since particles are not being used
    B = False
    
    # for using A particles
    #particles = True
    #A = True
    #B = False
    
    # loop through the 34 snapshot files, adding tracers, and saving the output to a file
    # if using particles, some of the low mass haloes do not have enough to position all satellite tracers
    # use the add_missing_particles function to add these missing particles, using other haloes of the same mass
    for i in range(34):
        make_snapshot_tracers(i, output_file%i, clean=clean, particles=particles, redshift=redshift,
                          simulation=simulation, box_size=box_size, cosmo=cosmo, ph=ph, A=A, B=B,
                          ntracer=ntracer, log_mass_min=log_mass_min,
                          abacus_cosmologies_file=abacus_cosmologies_file)
        
        # add_missing_particles(output_file%i, box_size=box_size)

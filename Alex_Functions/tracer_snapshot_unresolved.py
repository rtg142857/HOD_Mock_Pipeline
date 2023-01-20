from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.data.read_abacus import read_asdf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import gc
from mass_function import MassFunction
from cosmology import CosmologyAbacus


def get_mass_function(clean=True, redshift=0.2,
                      simulation="base", box_size=2000, cosmo=0, ph=0,
                      abacus_cosmologies_file="abacus_cosmologies.csv"):
    """
    Get smooth fit to the mass function of an Abacus snapshot
    Args:
        clean:       use cleaned Abacus halo catalogue? Default is True
        redshift:    snapshot redshift. Default z=0.2
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        abacus_cosmologies_file: file of Abacus cosmological parameters
    """
    
    path = "/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_%s_c%03d_ph%03d/halos/"%(simulation, cosmo, ph)
    file_name = path+"z%.3f/halo_info/halo_info_%03d.asdf"

    # loop through all 34 files, reading in halo masses
    log_mass = [None]*34
    for file_number in range(34):
        input_file = file_name%(redshift, file_number)

        halo_cat = CompaSOHaloCatalog(input_file, cleaned=True, fields=['N'])
        m_par = halo_cat.header["ParticleMassHMsun"]
        log_mass[file_number] = np.log10(np.array(halo_cat.halos["N"])*m_par)
        
        print(file_number, len(log_mass[file_number]))

    log_mass = np.concatenate(log_mass)

    # get number densities in mass bins  
    bin_size = 0.02
    mass_bins = np.arange(10,16,bin_size)
    mass_binc = mass_bins[:-1]+bin_size/2.
    hist, bins = np.histogram(log_mass, bins=mass_bins)
    n_halo = hist/bin_size/box_size**3
    
    # remove bins with zero haloes
    keep = n_halo > 0
    measured_mass_function = np.array([mass_binc[keep], n_halo[keep]])

    # create mass function object
    cosmology = CosmologyAbacus(cosmo, abacus_cosmologies_file)
    mf = MassFunction(cosmology=cosmology, redshift=redshift, 
                      measured_mass_function=measured_mass_function)
    
    # get fit to mass function
    mf.get_fit()
    
    return mf


def make_snapshot_tracers_unresolved(output_file, mass_function, logMmin, logMmax, 
                                    redshift=0.2, simulation="base", box_size=2000, cosmo=0, ph=0,
                                    abacus_cosmologies_file="abacus_cosmologies.csv"):
    """
    Make file of central galaxy tracers for unresolved haloes, using Abacus field particles
    (particles not in haloes)
    Args:
        output_file: path of hdf5 file to save output
        mass_function: halo mass function, of class MassFunction
        logMmin:     minimum log halo mass to add
        logMmax      maximum log halo mass to add
        redshift:    snapshot redshift. Default z=0.2
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        abacus_cosmologies_file: file of Abacus cosmological parameters
    """
    
    # number of random haloes we need to get correct mass function
    Nrand = mass_function.number_density_in_mass_bin(logMmin, logMmax) * (box_size**3)
    
    # get total number of field particles (using A particles)
    path = "/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_%s_c%03d_ph%03d/halos/"%(simulation, cosmo, ph)
    N = np.zeros(34, dtype="i")
    for file_number in range(34):
        # this loop is slow. Is there a faster way to get total number of field particles in each file?
        file_name = path+"z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(redshift, file_number)
        data = read_asdf(file_name, load_pos=True, load_vel=False)
        p = data["pos"]
        del data
        N[file_number] = p.shape[0]
        gc.collect() # need to run garbage collection to release memory
        
    # total number of field particles in full snapshot
    Npar = np.sum(N)
    
    # probability to keep a particle
    prob = Nrand*1.0 / Npar
    
    for file_number in range(34):
        file_name = path+"z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(redshift, file_number)
        print(file_name)
        
        # choose random particles to keep, based on probability
        keep = np.random.rand(int(N[file_number])) <= prob
        
        # generate random masses for particles
        log_mass = mass_function.get_random_masses(np.count_nonzero(keep), logMmin, logMmax)
        
        # get pos and vel of random particles
        data = read_asdf(file_name, load_pos=True, load_vel=True)
        pos = data["pos"][keep] + box_size/2.
        vel = data["vel"][keep]
        
        del data
        gc.collect() # need to run garbage collection to release memory
    
        # save to file, converting masses to units 1e10 Msun/h
        f = h5py.File(output_file%file_number, "a")
        f.create_dataset("mass",     data=10**(log_mass-10), compression="gzip")
        f.create_dataset("position", data=pos, compression="gzip")
        f.create_dataset("velocity", data=vel, compression="gzip")
        f.close()
        
        
        
if __name__ == "__main__":
    
    path = "" #path to save the output files
    output_file = path+"galaxy_tracers_unresolved_%i.hdf5"
    
    # use cleaned halo catalogue mass function
    clean=True
    
    # base c000_ph000 simultion snapshot at z=0.2
    simulation = "base"
    cosmo = 0
    ph = 0
    box_size = 2000 #Mpc/h
    redshift = 0.2
    abacus_cosmologies_file = "abacus_cosmologies.csv"
    
    # masses of unressolved haloes to add
    logMmin = 10.3
    logMmax = 11
    
    # get halo mass function
    mass_function = get_mass_function(clean=clean, redshift=redshift, simulation=simulation, 
                                      box_size=box_size, cosmo=cosmo, ph=ph,
                                      abacus_cosmologies_file=abacus_cosmologies_file)
           
    # make file of central tracers, using particles, assigning random masses from mass function
    # this function automatically loops through all 34 files
    make_snapshot_tracers_unresolved(output_file, mass_function, logMmin, logMmax, 
                                    redshift=redshift, simulation=simulation, box_size=box_size, 
                                    cosmo=cosmo, ph=ph, abacus_cosmologies_file=abacus_cosmologies_file)
  

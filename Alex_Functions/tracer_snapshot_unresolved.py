import numpy as np
import h5py
import matplotlib.pyplot as plt
import gc
import swiftsimio as sw
from mass_function import MassFunction
from cosmology import CosmologyFlamingo
import yaml


def get_mass_function(L, N, simulation, redshift):
    """
    Get smooth fit to the mass function of a Flamingo snapshot
    Args:
        L:           Box length of the simulation (the 350 in e.g. L350N1800_DMO)
        N:           Number of particles in the simulation (the 1800 in e.g. L350N1800_DMO)
        simulation:  Specific version of the simulation (e.g. "DMO_FIDUCIAL", "HYDRO_STRONG_AGN")
        redshift:    snapshot redshift.
    Old args:
        clean:       use cleaned Abacus halo catalogue? Default is True
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        abacus_cosmologies_file: file of Abacus cosmological parameters
    """
    
    #path = "/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_%s_c%03d_ph%03d/halos/"%(simulation, cosmo, ph)
    #file_name = path+"z%.3f/halo_info/halo_info_%03d.asdf"

    simulation_path = "/cosma8/data/dp004/flamingo/Runs/L%03dN%03d/"%(L, N) + simulation
    input_file = "/cosma7/data/dp004/dc-mene1/flamingo_copies/L1000N1800_soap.hdf5"
    print("WARNING: Using incorrect path for making unresolved snapshot tracer halo mass function")

    # loop through all 34 files, reading in halo masses
    log_mass = [None]*34
    for file_number in range(34):
        #input_file = file_name%(redshift, file_number)

        halo_cat = h5py.File(input_file, "r")
        m_par = halo_cat["SO"]["DarkMatterMass"][0] / halo_cat["SO"]["NumberOfDarkMatterParticles"][0]
        log_mass[file_number] = np.log10(np.array(halo_cat["SO"]["DarkMatterMass"]))

        #halo_cat = CompaSOHaloCatalog(input_file, cleaned=True, fields=['N'])
        #m_par = halo_cat.header["ParticleMassHMsun"]
        #log_mass[file_number] = np.log10(np.array(halo_cat.halos["N"])*m_par)
        
        print(file_number, len(log_mass[file_number]))

    log_mass = np.concatenate(log_mass)

    # get number densities in mass bins  
    bin_size = 0.02
    mass_bins = np.arange(10,16,bin_size)
    mass_binc = mass_bins[:-1]+bin_size/2.
    hist, bins = np.histogram(log_mass, bins=mass_bins)
    n_halo = hist/bin_size/L**3
    
    # remove bins with zero haloes
    keep = n_halo > 0
    measured_mass_function = np.array([mass_binc[keep], n_halo[keep]])

    # create mass function object
    cosmology = CosmologyFlamingo(L, N, simulation)
    mf = MassFunction(cosmology=cosmology, redshift=redshift, 
                      measured_mass_function=measured_mass_function)
    
    # get fit to mass function
    mf.get_fit()
    
    return mf


def make_snapshot_tracers_unresolved(output_file, mass_function, logMmin, logMmax, 
                                    redshift, L, N, simulation, group_id_default):
    """
    Make file of central galaxy tracers for unresolved haloes, using Abacus field particles
    (particles not in haloes)
    Args:
        output_file: path of hdf5 file to save output
        mass_function: halo mass function, of class MassFunction
        logMmin:     minimum log halo mass to add
        logMmax      maximum log halo mass to add
        redshift:    snapshot redshift.
        L:           Box length of the simulation (the 350 in e.g. L350N1800_DMO)
        N:           Number of particles in the simulation (the 1800 in e.g. L350N1800_DMO)
        simulation:  Specific version of the simulation (e.g. "DMO_FIDUCIAL", "HYDRO_STRONG_AGN")
        group_id_default: Default halo ID for particles that aren't in any FOF halo
    Old args:
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        abacus_cosmologies_file: file of Abacus cosmological parameters
    """
    
    # number of random haloes we need to get correct mass function
    Nrand = mass_function.number_density_in_mass_bin(logMmin, logMmax) * (L**3)
    
    # get total number of field particles (formerly using A particles)
    #path = "/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_%s_c%03d_ph%03d/halos/"%(simulation, cosmo, ph)
    simulation_path = "/cosma8/data/dp004/flamingo/Runs/L%03dN%03d/"%(L, N) + simulation
    
    N = np.zeros(34, dtype="i")
    field_boolean = np.empty(34, dtype=np.ndarray)
    for file_number in range(1):
        # this loop is slow. Is there a faster way to get total number of field particles in each file?
        #file_name = path+"z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(redshift, file_number)
        file_name = "/cosma7/data/dp004/dc-mene1/flamingo_copies/L1000N1800_snapshot_77.hdf5"
        print("WARNING: Using incorrect path for making unresolved snapshot tracers")

        #file = h5py.File(file_name, "r")
        #data = read_asdf(file_name, load_pos=True, load_vel=False)
        #p = data["pos"]

        data = sw.load(file_name)
        p = (data.dark_matter.fofgroup_ids == group_id_default)
        del data
        field_boolean[file_number] = p
        N[file_number] = p.sum()
        gc.collect() # need to run garbage collection to release memory
        
    ##################the above should be looped over all files with the right snapshot

    # total number of field particles in full snapshot
    Npar = np.sum(N)
    
    # probability to keep a particle
    prob = Nrand*1.0 / Npar
    
    for file_number in range(34):
        #file_name = path+"z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(redshift, file_number)
        file_name = "/cosma7/data/dp004/dc-mene1/flamingo_copies/L1000N1800_snapshot_77.hdf5"
        print("WARNING: Using incorrect path for making unresolved snapshot tracers")
        print(file_name)
        
        # choose random particles to keep, based on probability
        keep_unfiltered = np.random.rand(int(N[file_number])) <= prob
        keep = np.logical_and(keep_unfiltered, field_boolean[file_number])
        
        # generate random masses for particles
        log_mass = mass_function.get_random_masses(np.count_nonzero(keep), logMmin, logMmax)
        
        # get pos and vel of random particles
        #data = read_asdf(file_name, load_pos=True, load_vel=True)
        data = sw.load(file_name)
        pos = np.array(data.dark_matter.coordinates[keep])# + L/2
        vel = np.array(data.dark_matter.velocities[keep])
        
        del data
        gc.collect() # need to run garbage collection to release memory
    
        # save to file, converting masses to units 1e10 Msun/h
        f = h5py.File(output_file%file_number, "a")
        f.create_dataset("mass",     data=10**(log_mass-10), compression="gzip")
        f.create_dataset("position", data=pos, compression="gzip")
        f.create_dataset("velocity", data=vel, compression="gzip")
        f.close()
        
        
        
if __name__ == "__main__":
    
    path = "tracer_output" #path to save the output files
    output_file = path+"galaxy_tracers_unresolved_%i.hdf5"
    
    # use cleaned halo catalogue mass function
    #clean=True
    
    # base L0100N0180 DMO_FIDUCIAL simultion snapshot
    simulation = "DMO_FIDUCIAL"
    L = 100
    N = 180
    #simulation = "base"
    #cosmo = 0
    #ph = 0
    #box_size = 2000 #Mpc/h
    #redshift = 0.2
    #abacus_cosmologies_file = "abacus_cosmologies.csv"
    sw_data = sw.load("/cosma7/data/dp004/dc-mene1/flamingo_copies/L1000N1800_snapshot_77.hdf5")
    print("WARNING: Using incorrect path for loading redshift")
    redshift=sw_data.metadata.redshift
    param_file_path = "/cosma7/data/dp004/dc-mene1/flamingo_copies/L1000N1800_params.yml"
    print("WARNING: Using incorrect path in loading cosmology")

    with open(param_file_path, "r") as file:
        run_params = yaml.safe_load(file)

    group_id_default = run_params["FOF"]["group_id_default"]
    
    # masses of unressolved haloes to add
    # In AbacusSummit, halos are resolved down to 10^11; for making mocks, they need halos down to 10^10.3 for making BGS stuff
    logMmin = 10.3
    logMmax = 11
    print("WARNING: Using wrong upper mass bound for unresolved halos")
    
    # get halo mass function
    mass_function = get_mass_function(redshift=redshift, L=L, N=N, simulation=simulation)
    
    # make file of central tracers, using particles, assigning random masses from mass function
    # this function automatically loops through all files
    make_snapshot_tracers_unresolved(output_file, mass_function, logMmin, logMmax, redshift=redshift, L=L, N=N, simulation=simulation, group_id_default=group_id_default)
    print("WARNING: Not creating any unresolved tracers")
  

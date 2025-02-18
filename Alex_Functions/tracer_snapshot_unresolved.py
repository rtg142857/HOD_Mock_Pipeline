import numpy as np
import h5py
import gc
import swiftsimio as sw
from mass_function import MassFunction
from cosmology import CosmologyFlamingo
from read_hdf5 import read_hbt_log_mass, read_soap_log_mass, find_field_particles_snapshot_file
import yaml
import sys
import os
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

def get_mass_function(path_config_filename):
    """
    Get smooth fit to the mass function of a Flamingo snapshot
    Args:
        path_config_filename: Path to the config file containing paths to other useful things
    Old args:
        clean:       use cleaned Abacus halo catalogue? Default is True
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        abacus_cosmologies_file: file of Abacus cosmological parameters
    """
    with open(path_config_filename, "r") as file:
        path_config = yaml.safe_load(file)
    with open(path_config["Paths"]["params_path"], "r") as file:
        used_params = yaml.safe_load(file)

    cosmology = CosmologyFlamingo(path_config_filename)

    soap_path = path_config["Paths"]["soap_path"]
    redshift = path_config["Params"]["redshift"]
    h = used_params["Cosmology"]["h"]
    L = path_config["Params"]["L"] * h
    log_mass_min = path_config["Params"]["log_mass_min"]
    try:
        halo_type = path_config["Misc"]["halo_type"]
    except:
        halo_type = "soap"
    UnitMass_in_cgs = float(used_params["InternalUnitSystem"]["UnitMass_in_cgs"])

    print("Reading log mass from file...", flush=True)
    if soap_path[-5:] == ".hdf5": # if the soap path is a single file

        input_file = soap_path
        if halo_type == "peregrinus":
            log_mass = read_hbt_log_mass(input_file, UnitMass_in_cgs, h)
        else:
            log_mass = read_soap_log_mass(input_file, UnitMass_in_cgs, h, redshift, cosmology)
        
        print("Read log mass from file", flush=True)

    else: # if it's a directory
    # location of the snapshots
        soap_files_list = os.listdir(soap_path)
        if halo_type == "peregrinus":
            soap_files_list = [file for file in soap_files_list if "Catalogue" in file]

        # loop through all files, reading in halo masses
        log_mass = [None]*len(soap_files_list)
        for file_name in soap_files_list:
            file_number = int(file_name.split(".")[1])
            input_file = soap_path + file_name

            if halo_type == "peregrinus":
                log_mass[file_number] = read_hbt_log_mass(input_file, UnitMass_in_cgs, h)
            else:
                log_mass[file_number] = read_soap_log_mass(input_file, UnitMass_in_cgs, h, redshift, cosmology)

            
            print(file_number, len(log_mass[file_number]))

        log_mass = np.concatenate(log_mass)
        print("Read log mass from files", flush=True)

    # get number densities in mass bins  
    bin_size = 0.02
    mass_bins = np.arange(log_mass_min,16,bin_size)
    mass_binc = mass_bins[:-1]+bin_size/2.
    hist, bins = np.histogram(log_mass, bins=mass_bins)
    n_halo = hist/bin_size/L**3
    
    # remove bins with zero haloes
    keep = n_halo > 0
    measured_mass_function = np.array([mass_binc[keep], n_halo[keep]])
    print("Mass function from data:")
    print(np.array2string(measured_mass_function, separator=","), flush=True)

    # create mass function object
    mf = MassFunction(cosmology=cosmology, redshift=redshift, 
                      measured_mass_function=measured_mass_function)
    
    # get fit to mass function
    mf.get_fit()
    
    func = mf.number_density(mass_bins, redshift)
    print("Mass bins: ", np.array2string(mass_bins, separator=","))
    print("Calculated mass function for these bins: ", np.array2string(func, separator=","), flush=True)

    return mf


def make_snapshot_tracers_unresolved(output_file, mass_function, path_config_filename):
    """
    Make file of central galaxy tracers for unresolved haloes, using Flamingo field particles
    (particles not in haloes)
    Args:
        output_file: path of hdf5 file to save output
        mass_function: halo mass function, of class MassFunction
        path_config_filename: Path to the config file containing paths to other useful things
    Old args:
        simulation:  Abacus simulation. Default is "base"
        box_size:    Simulation box size, in Mpc/h. Default is 2000 Mpc/h
        cosmo:       Abacus cosmology number. Default is 0
        ph:          Abacus simulation phase. Default is 0
        abacus_cosmologies_file: file of Abacus cosmological parameters
        logMmin:     minimum log halo mass to add
        logMmax      maximum log halo mass to add
        redshift:    snapshot redshift.
    """
    with open(path_config_filename, "r") as file:
        path_config = yaml.safe_load(file)
    with open(path_config["Paths"]["params_path"], "r") as file:
        run_params = yaml.safe_load(file)
    #redshift = path_config["Params"]["redshift"]
    h = run_params["Cosmology"]["h"]
    L = path_config["Params"]["L"] * h
    logMmin = path_config["Params"]["logMmin"]
    logMmax = path_config["Params"]["logMmax"]
    particle_rate = path_config["Misc"]["particle_rate"]

    group_id_default = run_params["FOF"]["group_id_default"]

    # number of random haloes we need to get correct mass function
    Nrand = mass_function.number_density_in_mass_bin(logMmin, logMmax) * (L**3)
    
    # get total number of field particles (formerly using A particles)
    snapshot_path = path_config["Paths"]["snapshot_path"]

    if not os.path.isfile("tracer_output/field_boolean.pickle"):
        print("Counting field particles", flush=True)

        if ".hdf5" in snapshot_path: # it's a file
            field_boolean = find_field_particles_snapshot_file(snapshot_path, group_id_default, particle_rate)
            Npar = field_boolean.sum()
            gc.collect() # need to run garbage collection to release memory

            #save field boolean to pickle file
            with open("tracer_output/field_boolean.pickle", "wb") as handle:
                pickle.dump(field_boolean, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else: # it's a directory of files
            snapshot_files_list = os.listdir(snapshot_path)
            snapshot_files_list = [file for file in snapshot_files_list if file.count(".") == 2]
            N_particles_in_file = np.zeros(len(snapshot_files_list), dtype="i")
            field_boolean = [None]*len(snapshot_files_list)

            for file_name in snapshot_files_list:
                file_number = int(file_name.split(".")[1])
                input_file = snapshot_path + file_name
                print("Finding field particles (file "+str(file_number)+" out of "+str(len(snapshot_files_list))+")...", flush=True)
                field_boolean[file_number] = find_field_particles_snapshot_file(input_file, group_id_default, particle_rate)
                N_particles_in_file[file_number] = field_boolean[file_number].sum()

                gc.collect() # release memory

            #save field boolean to pickle file
            with open("tracer_output/field_boolean.pickle", "wb") as handle:
                pickle.dump(field_boolean, handle, protocol=pickle.HIGHEST_PROTOCOL)

            Npar = N_particles_in_file.sum()

    else: # if there's a field boolean pickle file
        with open("tracer_output/field_boolean.pickle", "rb") as handle:
            field_boolean = pickle.load(handle)
            if isinstance(field_boolean, list):
                N_particles_in_file = [array.sum() for array in field_boolean]
                Npar = sum(N_particles_in_file)
            else:
                Npar = field_boolean.sum()

        
    # probability to keep a particle
    prob = Nrand*1.0 / Npar
    
    print("Choosing random particles to keep with probability "+str(prob), flush=True)
    
    if ".hdf5" in snapshot_path:
        # choose random particles to keep, based on probability
        keep_unfiltered = np.random.rand(int(field_boolean.shape[0])) <= prob
        keep = np.logical_and(keep_unfiltered, field_boolean)
        
        # generate random masses for particles
        log_mass = mass_function.get_random_masses(np.count_nonzero(keep), logMmin, logMmax)
        
        # get pos and vel of random particles
        #data = read_asdf(file_name, load_pos=True, load_vel=True)
        data = sw.load(snapshot_path)
        pos = np.array(data.dark_matter.coordinates)[::particle_rate] * h
        pos = pos[keep]
        vel = np.array(data.dark_matter.velocities)[::particle_rate]
        vel = vel[keep]
        
        del data
        gc.collect() # need to run garbage collection to release memory

        # save to file, converting masses to units 1e10 Msun/h
        print("Saving field particles to file as unresolved tracers", flush=True)
        f = h5py.File(output_file%0, "a")
        f.create_dataset("mass",     data=10**(log_mass-10), compression="gzip")
        f.create_dataset("position", data=pos, compression="gzip")
        f.create_dataset("velocity", data=vel, compression="gzip")
        f.close()
    else:
        snapshot_files_list = os.listdir(snapshot_path)
        snapshot_files_list = [file for file in snapshot_files_list if file.count(".") == 2]
        for file_name in snapshot_files_list:
            file_number = int(file_name.split(".")[1])
            input_file = snapshot_path + file_name

            if not os.path.isfile(output_file%file_number): # only write to file if it doesn't already exist
                keep_unfiltered = np.random.rand(int(field_boolean[file_number].shape[0])) <= prob
                keep = np.logical_and(keep_unfiltered, field_boolean[file_number])

                # generate random masses for particles
                log_mass = mass_function.get_random_masses(np.count_nonzero(keep), logMmin, logMmax)
                
                # get pos and vel of random particles
                data = sw.load(input_file)
                pos = np.array(data.dark_matter.coordinates)[::particle_rate] * h
                pos = pos[keep]
                vel = np.array(data.dark_matter.velocities)[::particle_rate]
                vel = vel[keep]
                
                del data
                gc.collect() # need to run garbage collection to release memory

                # save to file, converting masses to units 1e10 Msun/h
                print("Saving field particles to file "+str(file_number)+" as unresolved tracers", flush=True)
                f = h5py.File(output_file%file_number, "a")
                f.create_dataset("mass",     data=10**(log_mass-10), compression="gzip")
                f.create_dataset("position", data=pos, compression="gzip")
                f.create_dataset("velocity", data=vel, compression="gzip")
                f.close()

        
if __name__ == "__main__":
    path_config_filename = sys.argv[1] # Config file path

    with open(path_config_filename, "r") as file:
        path_config = yaml.safe_load(file)

    soap_path = path_config["Paths"]["soap_path"]
    
    output_path = "tracer_output/" #path to save the output files
    output_file = output_path+"galaxy_tracers_unresolved_%i.hdf5"
    

    print("Getting mass function for unresolved tracers")
    mass_function = get_mass_function(path_config_filename)

    print("Unresolved tracer mass function obtained, making snapshot tracers now")
    # make file of central tracers, using particles, assigning random masses from mass function
    # this function automatically loops through all files
    make_snapshot_tracers_unresolved(output_file, mass_function, path_config_filename=path_config_filename)
  

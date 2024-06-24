import h5py
import numpy as np

from halo_catalogue import FlamingoSnapshot
from galaxy_catalogue import GalaxyCatalogueSnapshot
from cosmology import CosmologyFlamingo
from hod_tracer import HOD_Tracer
import yaml
import sys
import os

def make_snapshot_tracers(input_file, output_file,
                          path_config_filename):
    """
    Make file of galaxy tracers for a Flamingo simulation snapshot
    Args:
        input_file: Path of SOAP-style file in which we find halos
        output_file: path of hdf5 file to save output
        path_config_filename: The path to the path_config file saying the path to everything

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
        particles: use particles if True, NFW if False
    """
    #path = "/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_%s_c%03d_ph%03d/halos/"%(simulation, cosmo, ph)
    #file_name = path+"z%.3f/halo_info/halo_info_%03d.asdf"%(redshift, file_number)
    #simulation_path = "/cosma8/data/dp004/flamingo/Runs/L%03dN%03d/"%(L, N) + simulation
    #file_name = "/cosma7/data/dp004/dc-mene1/flamingo_copies/L1000N1800_soap.hdf5"
    print(input_file)

    with open(path_config_filename, "r") as file:
        path_config = yaml.safe_load(file)
    log_mass_min = path_config["Params"]["log_mass_min"]
    L = path_config["Params"]["L"]
    ntracer = path_config["Params"]["ntracer"]

    cosmology = CosmologyFlamingo(path_config_filename=path_config_filename)
    
    # read in the halo catalogue
    halo_cat = FlamingoSnapshot(input_file, path_config_filename=path_config_filename)
    
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
    path_config_filename = sys.argv[1] # Config file path

    with open(path_config_filename, "r") as file:
        path_config = yaml.safe_load(file)

    soap_path = path_config["Paths"]["soap_path"]
    #photsys = path_config["photsys"]
    #mag_faint = path_config["mag_faint"]
    #redshift = path_config["redshift"]
    #L = path_config["L"]
    #N = path_config["N"]
    #log_mass_min = path_config["log_mass_min"]
    
    # number of satellite tracers for each halo
    #ntracer = 3

    #sw_data = sw.load("/cosma8/data/dp004/flamingo/Runs/L%03dN%03d/"%(L, N) + simulation)
    #sw_data = sw.load("/cosma7/data/dp004/dc-mene1/flamingo_copies/L1000N1800_snapshot_77.hdf5")
    #print("WARNING: Using incorrect path for loading redshift")
    #redshift=sw_data.metadata.redshift

    #simulation = "base"
    #cosmo = 0
    #ph = 0
    #box_size = 2000 #Mpc/h
    #redshift = 0.2
    #abacus_cosmologies_file = "abacus_cosmologies.csv"
    
    # for NFW profile
    #particles=False

    path = "tracer_output/" #path to save the output files
    
    if soap_path[-5:] == ".hdf5": # if the soap path is a single file
        output_file = path+"galaxy_tracers_0.hdf5"
        make_snapshot_tracers(soap_path, output_file,
                              path_config_filename=path_config_filename)
    else:

        # location of the snapshots
        soap_files_list = os.listdir(soap_path)
        # loop through the SOAP files, adding tracers, and saving the output to a file
        for file_number, file_name in enumerate(soap_files_list):
            output_file = path+"galaxy_tracers_"+str(file_number)+".hdf5"

            make_snapshot_tracers(soap_path+file_name, output_file,
                                path_config_filename=path_config_filename)
            
            # add_missing_particles(output_file%i, box_size=box_size)

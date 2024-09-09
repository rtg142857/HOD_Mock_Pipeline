import numpy as np
import h5py
import swiftsimio as sw

def read_soap_log_mass(input_file, UnitMass_in_cgs):
    """
    Get an array representing the base-10 logarithm of the 200-mean dark matter masses from a SOAP file, in units of log solar mass.
    Args:
        input_file: Path to the SOAP file.
        UnitMass_in_cgs: The mass unit used in the SOAP calc (same as the snapshot units) in grams.
    """
    halo_cat = h5py.File(input_file, "r")
    UnitMass_in_Msol = UnitMass_in_cgs / 1.98841e33
    log_mass = np.log10(np.array(halo_cat["SO"]["200_mean"]["DarkMatterMass"]) * UnitMass_in_Msol)
    return log_mass

def find_field_particles_snapshot_file(input_file, group_id_default):
    """
    Identify the field particles (DM particles not in FOF halos) in a Flamingo snapshot file.
    Return a boolean array representing whether or not each particle in the file is a field particle.
    Args:
        input_file: Path to the Flamingo snapshot file.
        group_id_default: Halo ID given to field particles (particles not in halos).
    """
    data = sw.load(input_file)
    field_boolean = (data.dark_matter.fofgroup_ids == group_id_default)
    del data
    return field_boolean

#def count_field_particles_snapshot_directory(input_directory, group_id_default):
    """
    Count the number of field particles (particles not in halos) across all Flamingo snapshot files in a directory,
    not including the virtual file containing every particle.
    Use this if you encounter out-of-memory errors.
    Args:
        input_file: Path to the Flamingo snapshot file.
        group_id_default: Halo ID given to field particles (particles not in halos).
    """
import numpy as np
import h5py
import swiftsimio as sw

def read_soap_log_mass(input_file, UnitMass_in_cgs):
    """
    Get an array representing the base-10 logarithm of the 200-crit dark matter field halo masses from a SOAP file, in units of log solar mass.
    Args:
        input_file: Path to the SOAP file.
        UnitMass_in_cgs: The mass unit used in the SOAP calc (same as the snapshot units) in grams.
    """
    halo_cat = h5py.File(input_file, "r")
    is_not_subhalo = np.array(halo_cat["InputHalos"]["HBTplus"]["Depth"]) == 0

    UnitMass_in_Msol = UnitMass_in_cgs / 1.98841e33
    log_mass = np.log10(np.array(halo_cat["SO"]["200_crit"]["DarkMatterMass"])[is_not_subhalo] * UnitMass_in_Msol)
    return log_mass

def read_hbt_log_mass(input_file, UnitMass_in_cgs):
    """
    Get an array representing the base-10 logarithm of the 200-crit dark matter field halo masses from an HBT file, in units of log solar mass.
    Args:
        input_file: Path to the HBT file.
        UnitMass_in_cgs: The mass unit used in the HBT calc (same as the snapshot units) in grams.
    """
    halo_cat = h5py.File(input_file, "r")
    is_not_subhalo = np.array(halo_cat["Subhalos"]["Rank"]) == 0
    # Some (field) halos have 0 mass and 0 radius; these are the orphan halos that were ejected from their host halo; we discard these
    is_not_0mass = np.array(halo_cat["Subhalos"]["BoundM200Crit"]) != 0
    relevant_field_halos = np.logical_and(is_not_0mass, is_not_subhalo)

    UnitMass_in_Msol = UnitMass_in_cgs / 1.98841e33
    log_mass = np.log10(np.array(halo_cat["Subhalos"]["BoundM200Crit"])[relevant_field_halos] * UnitMass_in_Msol)
    return log_mass

def find_field_particles_snapshot_file(input_file, group_id_default, particle_rate):
    """
    Identify the field particles (DM particles not in FOF halos) in a Flamingo snapshot file.
    Return a boolean array representing whether or not each particle in the file is a field particle.
    Args:
        input_file: Path to the Flamingo snapshot file.
        group_id_default: Halo ID given to field particles (particles not in halos).
        particle_rate: If particle_rate is N, only consider every Nth particle. Done to avoid out of memory errors.
    """
    data = sw.load(input_file)
    DM_IDs = data.dark_matter.fofgroup_ids[::particle_rate]
    field_boolean = (DM_IDs == group_id_default)
    del data
    del DM_IDs
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
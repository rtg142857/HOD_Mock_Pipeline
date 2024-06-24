import numpy as np
import h5py

def read_soap_log_mass(input_file):
    """
    Get an array representing the base-10 logarithm of the 200-mean dark matter masses from a SOAP file.
    Args:
        input_file: Path to the SOAP file.
    """
    halo_cat = h5py.File(input_file, "r")
    log_mass = np.log10(np.array(halo_cat["SO"]["200_mean"]["DarkMatterMass"]))
    return log_mass
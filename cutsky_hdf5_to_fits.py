import numpy as np
import h5py
import pandas as pd
from astropy.table import Table
from desimodel.io import load_tiles
from desimodel.footprint import is_point_in_desi
import fitsio
import sys

file_in = sys.argv[1]
file_out = sys.argv[2]


def hdf5_to_fits_cutsky(input_file, output_file):
    """
    Convert the final cut sky mock from hdf5 to fits format
    """
    f = h5py.File(input_file, "r")

    gtype = f["Data/galaxy_type"][...]
    cen = np.array(gtype%2 == 0, dtype="i")
    res = np.array(gtype < 2, dtype="i")
    N = len(gtype)

    hdict = {'SV3_AREA': 207.5, 'Y5_AREA':14850.4}
    data_fits = np.zeros(N, dtype=[('R_MAG_APP', 'f4'), ('R_MAG_ABS', 'f4'),
                               ('G_R_REST', 'f4'), ('G_R_OBS', 'f4'),
                               ('DEC', 'f4'), ('HALO_MASS', 'f4'),
                               ('CEN', 'i4'), ('RES', 'i4'), ('RA', 'f4'),
                               ('Z_COSMO', 'f4'), ('Z', 'f4'),
                               ('STATUS', 'i4')])

    data_fits['R_MAG_APP']   = f["Data/app_mag"][...]
    data_fits['R_MAG_ABS']   = f["Data/abs_mag"][...]
    data_fits['G_R_REST']    = f["Data/g_r"][...]
    data_fits['G_R_OBS']     = f["Data/g_r_obs"][...]
    data_fits['DEC']         = f["Data/dec"][...]
    data_fits['HALO_MASS']   = f["Data/halo_mass"][...]
    data_fits['CEN']         = cen
    data_fits['RES']         = res
    data_fits['RA']          = f["Data/ra"][...]
    data_fits['Z_COSMO']     = f["Data/z_cos"][...]
    data_fits['Z']           = f["Data/z_obs"][...]
    data_fits['STATUS']      = f["Data/STATUS"][...]

    f.close()

    fits = fitsio.FITS(output_file, "rw")
    fits.write(data_fits, header=hdict)
    fits.close()


hdf5_to_fits_cutsky(file_in,file_out)

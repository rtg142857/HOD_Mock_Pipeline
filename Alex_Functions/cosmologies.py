import numpy as np
import nbodykit.cosmology


def cosmology_mxxl():
    # MXXL Simulation cosmology
    cosmo = nbodykit.cosmology.WMAP5
    cosmo = cosmo.clone(Omega0_b=0.045, Omega0_cdm=0.25-0.045, h=0.73, n_s=1)
    cosmo = cosmo.match(sigma8=0.9)
    return cosmo
        
        
def cosmology_unit():
    # UNIT Simulation cosmology
    cosmo = nbodykit.cosmology.Planck15
    cosmo = cosmo.clone(Omega0_b=0.04860, Omega0_cdm=0.3089-0.04860, h=0.6774, n_s=0.9667)
    cosmo = cosmo.match(sigma8=0.8147)
    return cosmo
        
        
def cosmology_uchuu():
    # Uchuu Simulation cosmology
    cosmo = nbodykit.cosmology.Planck15
    cosmo = cosmo.clone(Omega0_b=0.04860, Omega0_cdm=0.3089-0.04860, h=0.6774, n_s=0.9667)
    cosmo = cosmo.match(sigma8=0.8159)
    return cosmo
    

def cosmology_abacus(cosmo_number=0, cosmo_file="abacus_cosmologies.dat"):
    # Abacus Silumation cosmologies
    
    omega_b, omega_cdm, h, A_s, n_s, alpha_s, N_ur, N_ncdm, omega_ncdm, w0_fld, \
                 wa_fld, sigma8_m, sigma8_cb = read_abacus_parameters(cosmo_number, cosmo_file)
    Omega_b = omega_b/h**2
    Omega_cdm = omega_cdm/h**2

    cosmo = nbodykit.cosmology.Cosmology(h=h, T0_cmb=2.7255, Omega0_b=Omega_b, 
                                         Omega0_cdm=Omega_cdm, N_ur=N_ur, 
                                         m_ncdm=[0.06], n_s=n_s, A_s=A_s)

    return cosmo


def read_abacus_parameters(cosmo_number=0, cosmo_file="abacus_cosmologies.dat"):
    params = np.loadtxt(cosmo_file)
    idx = np.where(params[:,0]==cosmo_number)[0]
    return params[idx,1:][0]
    

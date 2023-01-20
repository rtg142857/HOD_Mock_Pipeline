import numpy as np
from nbodykit.lab import cosmology 
from cosmologies import cosmology_mxxl, cosmology_abacus


def get_xi(cosmo, r_bins, scale=8, power_spectrum="zel"):
    """
    Returns the correlation function xi(r)
    
    Args:
        cosmo: nbodykit cosmology class
        r_bins: array of bins in r to calculate xi(r)
        scale: scale for normalisation (default 8 Mpc/h)
        power_spectrum: can be "lin", "nl" or "zel" (default is "zel")

    Returns:
        xi at the normalisation scale
        array of xi evaluated in r_bins
    """
    if power_spectrum=="lin":
        # linear power spectrum
        Pk = cosmology.LinearPower(cosmo, redshift=0.2, transfer='CLASS')
    elif power_spectrum=="nl":
        # non-linear power spectrum
        Pk = cosmology.HalofitPower(cosmo, redshift=0.2)
    elif power_spectrum=="zel":
        # Zel'dovich power spectrum 
        Pk = cosmology.ZeldovichPower(cosmo, redshift=0.2)
    else:
        raise ValueError("Invalid power spectrum", power_spectrum)
    
    xi = cosmology.CorrelationFunction(Pk)
    
    return xi(scale), xi(r_bins)


def get_wp(cosmo, rp_bins, pimax=120, scale=8, power_spectrum="zel"):
    """
    Returns the projected correlation function wp(rp)
    
    Args:
        cosmo: nbodykit cosmology class
        rp_bins: array of bins in rp to calculate wp(rp)
        pimax: maximum value of pi in integral
        scale: scale for normalisation (default 8 Mpc/h)
        power_spectrum: can be "lin", "nl" or "zel" (default is "zel")

    Returns:
        wp at the normalisation scale
        array of wp evaluated in rp_bins
    """
    pi_bins = np.arange(0,pimax+0.01,0.1)
    rp_grid, pi_grid = np.meshgrid(np.append(rp_bins,scale), pi_bins)
    r_bins = (rp_grid**2 + pi_grid**2)**0.5
    
    xi0, xi = get_xi(cosmo, r_bins=r_bins, scale=scale, power_spectrum=power_spectrum)
    wp = np.sum(xi,axis=0)
    
    return wp[-1], wp[:-1]



def get_scaling_factor(cosmo1, cosmo2, r_bins, pimax=120, correlation_function="xi", scale=8, 
                       power_spectrum="zel"):
    """
    Returns the cosmology rescaling factor
    """
        
    if correlation_function=="xi":
        xi_c1_8, xi_c1 = get_xi(cosmo1, r_bins, scale=scale, power_spectrum=power_spectrum)
        xi_c2_8, xi_c2 = get_xi(cosmo2, r_bins, scale=scale, power_spectrum=power_spectrum)
        scaling_factor = xi_c2/xi_c1 * (xi_c1_8/xi_c2_8)
    elif correlation_function=="wp":
        wp_c1_8, wp_c1 = get_wp(cosmo1, r_bins, pimax=pimax, scale=scale, power_spectrum=power_spectrum)
        wp_c2_8, wp_c2 = get_wp(cosmo2, r_bins, pimax=pimax, scale=scale, power_spectrum=power_spectrum)
        scaling_factor = wp_c2/wp_c1 * (wp_c1_8/wp_c2_8)
    else: 
        raise ValueError("Invalid correlation function", correlation_function)
        
    # keep scaling_factor fixed to 1 below scale
    scaling_factor[r_bins<scale] = 1.0
    
    return scaling_factor


if __name__ == "__main__":
    
    cosmo1 = cosmology_mxxl()
    cosmo2 = cosmology_abacus(cosmo_number=1, cosmo_file="abacus_cosmologies.dat")
    
    dr = 0.05
    r_bins = 10**np.arange(-2+dr/2., 2.21-dr/2., dr)
    
    correlation_function="xi" #"wp"
    pimax = 120 #Mpc/h, only needed if correlation_function="wp"
    scale = 8 #Mpc/h, scale where scaling factor normalised to be 1 (and set to 1 below this)
    
    power_spectrum = "zel" #use Zel'dovich power spectrum
    
    
    scaling_factor = get_scaling_factor(cosmo1, cosmo2, r_bins, pimax=pimax, 
                                        correlation_function=correlation_function, 
                                        scale=scale, power_spectrum=power_spectrum)
    
    
    np.savetxt("cosmology_rescaling_factor_%s_%s_%i.txt"%(correlation_function, power_spectrum, scale), scaling_factor)
    

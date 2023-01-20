import numpy as np
from nbodykit.lab import cosmology 
from cosmologies import cosmology_mxxl, cosmology_abacus
from scipy.special import gamma, gammaincc
from scipy.interpolate import RegularGridInterpolator, splrep, splev


class LuminosityFunction(object):
    """
    Luminsity function base class
    """
    def __init__(self):
        pass

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self.__Phi_z01(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)

        return 10**log_lf

    def __Phi_z01(self, magnitude):
        # returns a spline fit to the LF at z=0.1 (using the cumulative LF)
        mags = np.arange(0, -25, -0.001)
        phi_cums = self.Phi_cumulative(mags, 0.1)
        phi = (phi_cums[:-1] - phi_cums[1:]) / 0.001
        tck = splrep((mags[1:]+0.0005)[::-1], np.log10(phi[::-1]))
        return splev(magnitude, tck)

    def Phi_cumulative(self, magnitude, redshift):
        raise NotImplementedError




class LuminosityFunctionSchechter(LuminosityFunction):
    """
    Schecter luminosity function with evolution
    Args:
        Phi_star: LF normalization [h^3/Mpc^3]
        M_star: characteristic absolute magnitude [M-5logh]
        alpha: faint end slope
        P: number density evolution parameter
        Q: magnitude evolution parameter
    """
    def __init__(self, Phi_star, M_star, alpha, P, Q):

        # Evolving Shechter luminosity function parameters
        self.Phi_star = Phi_star
        self.M_star = M_star
        self.alpha = alpha
        self.P = P
        self.Q = Q

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # evolve M_star and Phi_star to redshift
        M_star = self.M_star - self.Q * (redshift - 0.1)
        Phi_star = self.Phi_star * 10**(0.4*self.P*redshift)

        # calculate luminosity function
        lf = 0.4 * np.log(10) * Phi_star
        lf *= (10**(0.4*(M_star-magnitude)))**(self.alpha+1)
        lf *= np.exp(-10**(0.4*(M_star-magnitude)))

        return lf


    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # evolve M_star and Phi_star to redshift
        M_star = self.M_star - self.Q * (redshift - 0.1)
        Phi_star = self.Phi_star * 10**(0.4*self.P*redshift)

        # calculate cumulative luminosity function
        t = 10**(0.4 * (M_star-magnitude))
        lf = Phi_star*(gammaincc(self.alpha+2, t)*gamma(self.alpha+2) - \
                           t**(self.alpha+1)*np.exp(-t)) / (self.alpha+1)

        return lf


class LuminosityFunctionTabulated(LuminosityFunction):
    """
    Luminosity function from tabulated file, with evolution
    Args:
        filename: path to ascii file containing tabulated values of cumulative
                  luminsity function
        P: number density evolution parameter
        Q: magnitude evolution parameter
    """
    def __init__(self, filename, P, Q):

        self.magnitude, self.log_number_density = \
                              np.loadtxt(filename, unpack=True)
        self.P = P
        self.Q = Q

        self.__lf_interpolator = \
            RegularGridInterpolator((self.magnitude,), self.log_number_density,
                                    bounds_error=False, fill_value=None)

    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self.__Phi_z01(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)

        return 10**log_lf

    def __Phi_z01(self, magnitude):
        # returns a spline fit to the LF at z=0.1 (using the cumulative LF)
        mags = np.arange(0, -25, -0.001)
        phi_cums = self.Phi_cumulative(mags, 0.1)
        phi = (phi_cums[:-1] - phi_cums[1:]) / 0.001
        tck = splrep((mags[1:]+0.0005)[::-1], np.log10(phi[::-1]))
        return splev(magnitude, tck)

    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # shift magnitudes to redshift z=0.1
        magnitude01 = magnitude + self.Q * (redshift - 0.1)

        # find interpolated number density at z=0.1
        log_lf01 = self.__lf_interpolator(magnitude01)

        # shift back to redshift
        log_lf = log_lf01 + 0.4 * self.P * (redshift - 0.1)

        return 10**log_lf


class LuminosityFunctionTarget(LuminosityFunction):
    """
    Target LF which transitions from SDSS at low redshifts to GAMA at
    high redshifts

    Args:
        lf_file: tabulated file of LF at z=0.1
        lf_param_file: file containing Schechter LF paramters at high z
    """

    def __init__(self, Phi_star=9.4e-3, M_star=-20.70, alpha=-1.23,
                 target_lf_file="target_lf.dat", P=1.8, Q=0.7):

        self.lf_sdss = LuminosityFunctionTabulated(target_lf_file, P, Q)

        self.lf_gama = LuminosityFunctionSchechter(Phi_star, M_star, alpha, P, Q)


    def transition(self, redshift):
        """
        Function which describes the transition between the SDSS LF
        at low z and the GAMA LF at high z
        """
        return 1. / (1. + np.exp(120*(redshift-0.15)))


    def Phi(self, magnitude, redshift):
        """
        Luminosity function as a function of absoulte magnitude and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        w = self.transition(redshift)

        lf_sdss = self.lf_sdss.Phi(magnitude, redshift)
        lf_gama = self.lf_gama.Phi(magnitude, redshift)

        return w*lf_sdss + (1-w)*lf_gama


    def Phi_cumulative(self, magnitude, redshift):
        """
        Cumulative luminosity function as a function of absoulte magnitude
        and redshift
        Args:
            magnitude: array of absolute magnitudes [M-5logh]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        w = self.transition(redshift)

        lf_sdss = self.lf_sdss.Phi_cumulative(magnitude, redshift)
        lf_gama = self.lf_gama.Phi_cumulative(magnitude, redshift)

        return w*lf_sdss + (1-w)*lf_gama


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

    #r_ratio = cosmo1.comoving_distance(0.2) / cosmo2.comoving_distance(0.2)

    mags = np.linspace(-22,-17,11)

    lf = LuminosityFunctionTarget()

    #lf_original = lf.Phi_cumulative(mags, 0.2)

    #shift_mag = 5 * np.log10(r_ratio)

    #lf_new = lf.Phi_cumulative(mags+shift_mag,0.2)

    #Vol_rescale = r_ratio**3


    #data_to_save = np.zeros((11,2))
    #data_to_save[:,0] = mags
    #data_to_save[:,1] = np.log10(Vol_rescale*lf_new)

    #np.savetxt("target_num_den_rescaled.txt",data_to_save)
    

    # adjust magnitudes to take into account different luminosity distance
    redshift = 0.2
    rcom_orig = cosmo1.comoving_distance(redshift)
    rcom_new = cosmo2.comoving_distance(redshift)
    magnitude_old = mags + 5*np.log10(rcom_new/rcom_orig)

    # get LF at new magnitudes
    lf = lf.Phi_cumulative(magnitude_old, redshift)

    # adjust number densities to take into accound volume differences
    
    H1_redshift = 100 * cosmo1.h * cosmo1.efunc(redshift)
    H1_0 = 100 * cosmo1.h * cosmo1.efunc(0)
    H2_redshift = 100 * cosmo2.h * cosmo2.efunc(redshift)
    H2_0 = 100 * cosmo2.h * cosmo2.efunc(0)
    print(H1_redshift,H1_0,H2_redshift,H2_0)
    vol_orig = cosmo1.comoving_distance(redshift)**2 / \
                  (H1_redshift/H1_0)
    vol_new = cosmo2.comoving_distance(redshift)**2 / \
                  (H2_redshift/H2_0)


    data_to_save = np.zeros((11,2))
    data_to_save[:,0] = mags
    data_to_save[:,1] = np.log10(lf * (vol_orig / vol_new))

    np.savetxt("target_num_den_rescaled.txt",data_to_save)

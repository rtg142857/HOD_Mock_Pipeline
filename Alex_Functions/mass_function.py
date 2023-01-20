#! /usr/bin/env python
import numpy as np
from scipy.interpolate import RegularGridInterpolator, splrep, splev
from scipy.optimize import curve_fit
from scipy.integrate import quad

from power_spectrum import PowerSpectrum


class MassFunction(object):
    """
    Class for fitting a Sheth-Tormen-like mass function to measurements from a simulation snapshot

    Args:
        cosmology: hodpy.Cosmology object, in the cosmology of the simulation
        redshift: redshift of the simulation snapshot
        [fit_params]: if provided, sets the best fit parameters to these values. 
                      fit_params is an array of [dc, A, a, p]
        [measured_mass_function]: mass function measurements to fit to, if provided.
                      measured_mass_function is an array of [log mass bins, mass function]
    """
    def __init__(self, cosmology, redshift, fit_params=None, measured_mass_function=None):
        
        self.cosmology = cosmology
        self.power_spectrum = PowerSpectrum(self.cosmology)
        self.redshift = redshift
        
        if not fit_params is None:
            self.dc, self.A, self.a, self.p = fit_params
            
        if not measured_mass_function is None:
            self.__mass_bins = measured_mass_function[0]
            self.__mass_func = measured_mass_function[1]
            
     
    def __func(self, sigma, dc, A, a, p):
        # Sheth-Tormen mass function
        mf = A * np.sqrt(2*a/np.pi)
        mf *= 1 + (sigma**2 / (a * dc**2))**p
        mf *= dc / sigma
        mf *= np.exp(-a * dc**2 / (2*sigma**2))

        return np.log10(mf)
        
    
    def get_fit(self):
        """
        Fits the Sheth-Tormen mass function to the measured mass function, returning the
        best fit parameters
        
        Returns:
            an array of [dc, A, a, p]
        """
        sigma = self.power_spectrum.sigma(10**self.__mass_bins, self.redshift)
        mf = self.__mass_func / self.power_spectrum.cosmo.mean_density(0) * 10**self.__mass_bins
        
        popt, pcov = curve_fit(self.__func, sigma, np.log10(mf), p0=[1,0.1,1.5,-0.5])
        
        self.update_params(popt)
        print("Fit parameters", popt)
        
        return popt

    
    def update_params(self, fit_params):
        '''
        Update the values of the best fit params
        
        Args:
            fit_params: an array of [dc, A, a, p]
        '''     
        self.dc, self.A, self.a, self.p = fit_params
        
    
    def mass_function(self, log_mass, redshift=None):
        '''
        Returns the halo mass function as a function of mass and redshift
        (where f is defined as Eq. 4 of Jenkins 2000)

        Args:
            log_mass: array of log_10 halo mass, where halo mass is in units Msun/h
        Returns:
            array of halo mass function
        '''        
        
        sigma = self.power_spectrum.sigma(10**log_mass, self.redshift)

        return 10**self.__func(sigma, self.dc, self.A, self.a, self.p)

    
    def number_density(self, log_mass, redshift=None):
        '''
        Returns the number density of haloes as a function of mass and redshift

        Args:
            log_mass: array of log_10 halo mass, where halo mass is in units Msun/h
        Returns:
            array of halo number density in units (Mpc/h)^-3
        '''  
        mf = self.mass_function(log_mass)

        return mf * self.power_spectrum.cosmo.mean_density(0) / 10**log_mass
    
    
    def number_density_in_mass_bin(self, log_mass_min, log_mass_max, redshift=None):
        '''
        Returns the number density of haloes in a mass bin

        Args:
            log_mass: array of log_10 halo mass, where halo mass is in units Msun/h
        Returns:
            array of halo number density in units (Mpc/h)^-3
        '''  
        
        return quad(self.number_density, log_mass_min, log_mass_max)[0]
    
    
    def get_random_masses(self, N, log_mass_min, log_mass_max, redshift=None):
        """
        Returns random masses, following the fit to the mass function
        Args:
            N: total number of masses to generate
            log_mass_min: log of minimum halo mass
            log_mass_max: log of maximum halo mass
        Returns:
            array of log halo mass
        """
        # get cumulative probability distribution
        log_mass_bins = np.linspace(log_mass_min, log_mass_max, 10000)
        prob_cum = self.number_density(log_mass_bins)
        prob_cum = np.cumsum(prob_cum)
        prob_cum-=prob_cum[0]
        prob_cum/=prob_cum[-1]
        
        tck = splrep(prob_cum, log_mass_bins)
    
        r = np.random.rand(N)
        
        return splev(r, tck)
    
        

#! /usr/bin/env python
from __future__ import print_function
import numpy as np


class HOD_Tracer(object):
    """
    HOD for adding satellite tracers to the halo catalogue
    """

    def __init__(self, ntracer=3):
        
        self.ntracer = ntracer
    
    def number_centrals_mean(self, log_mass, magnitude=None, redshift=None, f=None):
        """
        Average number of central galaxies in each halo brighter than
        some absolute magnitude threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of mean number of central galaxies
        """

        # use pseudo gaussian spline kernel
        return np.ones(len(log_mass))


    def number_satellites_mean(self, log_mass, magnitude=None, redshift=None, f=None):
        """
        Average number of satellite galaxies in each halo brighter than
        some absolute magnitude threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of mean number of satellite galaxies
        """
        
        return np.ones(len(log_mass)) * self.ntracer


    def number_galaxies_mean(self, log_mass, magnitude=None, redshift=None, f=None):
        """
        Average total number of galaxies in each halo brighter than
        some absolute magnitude threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of mean number of galaxies
        """
        return self.number_centrals_mean(log_mass, magnitude, redshift, f) + \
            self.number_satellites_mean(log_mass, magnitude, redshift, f)


    def get_number_satellites(self, log_mass, redshift=None):
        """
        Randomly draw the number of satellite galaxies in each halo,
        brighter than mag_faint, from a Poisson distribution
        Args:
            log_mass: array of the log10 of halo mass (Msun/h)
            redshift: array of halo redshifts
        Returns:
            array of number of satellite galaxies
        """
        # faint magnitude threshold at each redshift
        
        return np.ones(len(log_mass), dtype="i") * self.ntracer


    def get_magnitude_centrals(self, log_mass, redshift=None):
        """
        Use the HODs to draw a random magnitude for each central galaxy
        Args:
            log_mass: array of the log10 of halo mass (Msun/h)
            redshift: array of halo redshifts
        Returns:
            array of central galaxy magnitudes
        """
        # random number from spline kernel distribution
        return np.ones(len(log_mass))
    

    def get_magnitude_satellites(self, log_mass, number_satellites, redshift=None):
        """
        Use the HODs to draw a random magnitude for each satellite galaxy
        Args:
            log_mass:          array of the log10 of halo mass (Msun/h)
            redshift:          array of halo redshifts
            number_satellites: array of number of sateillites in each halo
        Returns:
            array of the index of each galaxy's halo in the input arrays
            array of satellite galaxy magnitudes
        """
        # create arrays of log_mass, redshift and halo_index for galaxies
        halo_index = np.arange(len(log_mass))
        halo_index = np.repeat(halo_index, number_satellites)
        
        return halo_index, np.ones(len(log_mass)*self.ntracer)


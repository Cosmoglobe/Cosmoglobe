from .skycomponent import SkyComponent
from numba import njit

import numpy as np
import astropy.units as u
import astropy.constants as const

# Initializing common astrophy units
h = const.h.value
c = const.c.value
k_B = const.k_B.value


class Dust(SkyComponent):
    """
    Parent class for all Dust models.
    """
    comp_label = 'dust'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs



class ModifiedBlackbody(Dust):
    """
    Model for modifed blackbody emission from thermal dust.

    """    
    model_label = 'MBB'

    def __init__(self, data, nside=None, fwhm=None):
        super().__init__(data, nside=nside, fwhm=fwhm)


    @u.quantity_input(nu=u.Hz)
    def get_emission(self, nu):
        """
        Returns the model emission of at a given frequency in units of K_RJ.
        Makes use of numba to speed up computations. Numba does not support
        astropy, so astropy objects are converted to pure values or 
        numpy.ndarrays during computation.

        Parameters
        ----------
        nu : astropy.units.quantity.Quantity
            Frequency at which to evaluate the model. 

        Returns
        -------
        emission : astropy.units.quantity.Quantity
            Model emission at given frequency in units of K_RJ.

        """
        emission = self._compute_emission(nu.si.value, 
                                         self.params['nu_ref'].si.value, 
                                         self.amp,
                                         self.beta, 
                                         self.T.value)

        return u.Quantity(emission, unit=self.amp.unit)


    @staticmethod
    @njit
    def _compute_emission(nu, nu_ref, amp, beta, T):
        """
        Computes the simulated modified blackbody emission of dust at a given
        frequency .

        Parameters
        ----------
        nu : int, float,
            Frequency at which to evaluate the modified blackbody radiation. 
        nu_ref : numpy.ndarray
            Reference frequency at which the Commander alm outputs are 
            produced.        
        beta : numpy.ndarray
            Estimated dust beta map from Commander.
        T : numpy.ndarray
            Temperature the modified blackbody. 
            
        Returns
        -------
        emission : numpy.ndarray
            Modeled modified blackbody emission at a given frequency.

        """
        nu_ref = nu_ref.reshape(len(nu_ref), 1) #For broadcasting compatibility
        
        emission = np.copy(amp)
        emission *= (nu/nu_ref)**(beta - 2) #(beta - 2) converts to K_RJ
        emission *= blackbody_ratio(nu, nu_ref, T)

        return emission



@njit
def blackbody_ratio(nu, nu_ref, T):
    """
    Returns the ratio between the emission emitted by a blackbody at two
    different frequencies.

    Parameters
    ----------
    nu : int, float,
        New frequency at which to evaluate the blackbody radiation.
    nu_ref : numpy.ndarray
        Reference frequency at which the Commander alm outputs are 
        produced.        
    T : numpy.ndarray
        Temperature of the blackbody.

    Returns
    -------
    numpy.ndarray
        Array containing the ratio between the blackbody emission at two 
        different frequencies.

    """
    return blackbody_emission(nu, T) / blackbody_emission(nu_ref, T)


@njit
def blackbody_emission(nu, T):
    """
    Returns blackbody radiation at a given frequency and temperature
    in units of uK_RJ.

    Parameters
    ----------
    nu : int, float,
        Frequency at which to evaluate the blackbody radiation.
    T : numpy.ndarray
        Temperature of the blackbody. 

    Returns
    -------
    numpy.ndarray
        Blackbody emission in units of Jy/sr

    """
    return ((2*h*nu**3)/c**2)/np.expm1(h*nu/(k_B*T))

from .skycomponent import SkyComponent

from numba import njit
import numpy as np
import astropy.units as u
import astropy.constants as const

# Initializing common astrophy units
h = const.h
k_B = const.k_B


class FreeFree(SkyComponent):
    """
    Parent class for all Free-Free models.
    """
    comp_label = 'ff'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs



class LinearOpticallyThin(FreeFree):
    """
    Linearized model strictly only valid in the optically thin case (tau << 1).

    """    
    model_label= 'freefree'
    other_quantities = ['Te_map']

    def __init__(self, data, nside=None, fwhm=None):
        super().__init__(data, nside=nside, fwhm=fwhm)


    @u.quantity_input(nu=u.GHz)
    def get_emission(self, nu):
        """
        Returns the model emission of at a given frequency in units of K_RJ.

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
                                         self.Te_map.value)

        return u.Quantity(emission, unit=self.amp.unit)


    @staticmethod
    @njit
    def _compute_emission(nu, nu_ref, amp, T_e):
        """
        Computes the power law emission at a given frequency for the 
        given Commander data.

        Parameters
        ----------
        nu : int, float,
            Frequency at which to evaluate the modified blackbody radiation. 
        nu_ref : numpy.ndarray
            Reference frequency at which the Commander alm outputs are 
            estimated.        
        T_e : numpy.ndarray
            Electron temperature map

        Returns
        -------
        emission : numpy.ndarray
            Modeled emission at a given frequency.

        """
        g_eff = gaunt_factor(nu, T_e)
        g_eff_ref = gaunt_factor(nu_ref, T_e)
        emission = np.copy(amp)
        emission *= (g_eff/g_eff_ref)*(nu/nu_ref)**-2
        emission *= np.exp(-h * ((nu-nu_ref)/(k_B*T_e)))

        return emission


@njit
def gaunt_factor(nu, T_e):
    """
    Computes the caunt factor for a given frequency and electron temperature.

    Parameters
    ----------
    nu : int, float
        Frequency at which to evaluate the Gaunt factor.   
    T_e : numpy.ndarray
        Electron temperature at which to evaluate the Gaunt factor.

    Returns
    -------
    numpy.ndarray
        Gaunt Factor.

    """
    return np.log(np.exp(5.96-(np.sqrt(3)/np.pi) * np.log(nu*
                  (T_e*1e-4)**-1.5)) + np.e )

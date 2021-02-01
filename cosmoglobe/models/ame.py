from .skycomponent import SkyComponent
from .. import data as data_dir

from numba import njit
import astropy.units as u
import numpy as np
import os

data_path = os.path.dirname(data_dir.__file__) + '/'


class AME(SkyComponent):
    """
    Parent class for all AME models.
    
    """
    comp_label = 'ame'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs


class SpinningDust2(AME):
    """
    Model for spinning dust emission.

    """    
    model_label = 'spindust2'
    other_quantities = ['nu_p_map']

    def __init__(self, data, nside=None, fwhm=None):
        super().__init__(data, nside=nside, fwhm=fwhm)

        # Reading in spdust2 template to interpolate for arbitrary frequency
        self.spdust2_nu, self.spdust2_amp = np.loadtxt(data_path + 'spdust2_cnm.dat', 
                                                       unpack=True)


    @u.quantity_input(nu=u.Hz)
    def get_emission(self, nu):
        """
        Returns the model emission of at a given frequency in units of K_RJ.

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency at which to evaluate the model. 

        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Model emission at given frequency in units of K_RJ.

        """
        if nu.value >= self.spdust2_nu[-1]:
            return u.Quantity(np.zeros_like(self.amp), unit=self.amp.unit)
        
        if nu.value <= self.spdust2_nu[0]:
            raise ValueError(
                             'Invalid frequency range for AME. '
                             'AME is not defined below 0.05 GHz.'
            )

        emission = self._compute_emission(nu.si.value, 
                                         self.params['nu_ref'].si.value, 
                                         self.nu_p_map.si.value, 
                                         self.amp,
                                         self.spdust2_amp, 
                                         self.spdust2_nu*1e9)

        return u.Quantity(emission, unit=self.amp.unit)


    @staticmethod
    @njit
    def _compute_emission(nu, nu_ref, nu_p, amp, spdust2_amp, spdust2_nu):
        """
        Computes the simulated spinning dust emission at a given frequency.

        Parameters
        ----------
        nu : int, float
            Frequency (GHz) at which to evaluate the modified blackbody radiation. 
        nu_ref : int, float, numpy.ndarray
            Reference frequency (GHz) at which the Commander alm outputs are 
            produced.
        nu_p : numpy.ndarray
            Peak frequency (GHz) parameter that shifts the specrum in log nu -
            log s space.
        spdust2_amp : numpy.ndarray
            SpDust2 code template amplitude. See Ali-Haimoud et al. 2009 for more 
            information.        
        spdust2_nu : numpy.ndarray
            SpDust2 code template frequency. See Ali-Haimoud et al. 2009 for more 
            information.

        Returns
        -------
        emission : numpy.ndarray
            Modeled emission at a given frequency.

        """
        scaled_nu = (nu * (30*1e9 / nu_p))
        scaled_nu_ref = (nu_ref * (30*1e9 / nu_p))

        emission = np.copy(amp)
        emission *= np.interp(scaled_nu, spdust2_nu, spdust2_amp)   
        emission /= np.interp(scaled_nu_ref, spdust2_nu, spdust2_amp)   
        emission *= (nu_ref/nu)**2

        return emission


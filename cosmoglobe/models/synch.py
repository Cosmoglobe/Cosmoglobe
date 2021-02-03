import astropy.units as u
from numba import njit
import numpy as np

from .skycomponent import SkyComponent


class Synchrotron(SkyComponent):
    """Parent class for all Synchrotron models."""

    comp_label = 'synch'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs



class PowerLaw(Synchrotron):
    """Model for power law emission for synchrotron."""

    model_label = 'power_law'

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
            Frequencies at which to evaluate the model. 

        Returns
        -------
        emission : astropy.units.quantity.Quantity
            Model emission at given frequency in units of K_RJ.

        """

        emission = self._compute_emission(nu.si.value, 
                                         self.params['nu_ref'].si.value, 
                                         self.amp,
                                         self.beta)

        return u.Quantity(emission, unit=self.amp.unit)


    @staticmethod
    @njit
    def _compute_emission(nu, nu_ref, amp, beta):
        """
        Computes the simulated power law emission of synchrotron at a given
        frequency. 

        Parameters
        ----------
        nu : int, float,
            Frequency at which to evaluate the emission. 
        nu_ref : numpy.ndarray
            Reference frequency at which the Commander alm outputs are 
            produced.        
        beta : numpy.ndarray
            Estimated synch beta map from Commander.
            
        Returns
        -------
        emission : numpy.ndarray
            Modeled synchrotron emission at a given frequency.

        """
        nu_ref = nu_ref.reshape(len(nu_ref), 1)
        emission = np.copy(amp)
        emission *= (nu/nu_ref)**beta

        return emission


import astropy.constants as const
import astropy.units as u
import numpy as np

from .skycomponent import SkyComponent

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


    @u.quantity_input(nu=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, nu, bandpass=None, output_unit=u.K):
        """
        Returns the model emission at an arbitrary frequency nu in units 
        of K_RJ.

        Parameters
        ----------
        nu : astropy.units.quantity.Quantity
            A frequency, or a frequency array at which to evaluate the model.
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile in units of (k_RJ, Jy/sr) corresponding to 
            frequency array nu. If None, a delta peak in frequency is assumed.
            Default : None
        output_unit : astropy.units.quantity.Quantity or str
            Desired unit for the output map. Must be a valid astropy.unit or 
            one of the two following strings ('K_CMB', 'K_RJ').
            Default : None

        Returns
        -------
        emission : astropy.units.quantity.Quantity
            Model emission at given frequency in units of K_RJ.

        """
        if bandpass is None:
            scaling = self._get_freq_scaling(nu.si.value, self.beta, self.T.value)
            emission = self.amp*scaling

        else:
            bandpass = self._get_normalized_bandpass(nu, bandpass)
            U = self._get_unit_conversion(nu, bandpass, output_unit)
            M = self._get_mixing(bandpass=bandpass.value,
                                 nus=nu.si.value, 
                                 spectral_params=(self.beta, self.T.value))

            emission = self.amp*M*U

        return emission


    def _get_freq_scaling(self, nu, beta, T):
        """
        Computes the frequency scaling from the reference frequency nu_ref to 
        an arbitrary frequency nu, which depends on the spectral parameters
        beta and T.

        Parameters
        ----------
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. 
        nu_ref : int, float, numpy.ndarray
            Reference frequency.        
        beta : numpy.ndarray
            Dust beta map.
        T : numpy.ndarray
            Dust temperature map.

        Returns
        -------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        nu_ref = np.expand_dims(self.params['nu_ref'].si.value, axis=1)

        scaling = (nu/nu_ref)**(beta-2)
        scaling *= blackbody_emission(nu, T) / blackbody_emission(nu_ref, T)

        return scaling



def blackbody_emission(nu, T):
    """
    Returns the emission emitted by a blackbody with with temperature T at 
    a frequency nu.

    Parameters
    ----------
    nu : int, float, numpy.ndarray
        Frequency at which to evaluate the blackbody radiation.
    T : numpy.ndarray
        Temperature of the blackbody. 

    Returns
    -------
    numpy.ndarray
        Blackbody emission in units of Jy/sr

    """
    return ((2*h*nu**3)/c**2) / np.expm1((h*nu) / (k_B*T))

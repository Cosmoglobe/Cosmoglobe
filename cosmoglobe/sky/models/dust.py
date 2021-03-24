import astropy.constants as const
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
        self._spectral_params = (self.beta, self.T.value)


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
        nu_ref = np.expand_dims(self.params.nu_ref.si.value, axis=1)

        scaling = (nu/nu_ref)**(beta-2)
        scaling *= blackbody_emission(nu, T) / blackbody_emission(nu_ref, T)

        return scaling

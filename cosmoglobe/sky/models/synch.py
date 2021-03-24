import numpy as np

from .skycomponent import SkyComponent

class Synchrotron(SkyComponent):
    """
    Parent class for all Synchrotron models.

    """
    comp_label = 'synch'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs



class PowerLaw(Synchrotron):
    """
    Model for power law emission for synchrotron.

    """
    model_label = 'power_law'

    def __init__(self, data, nside=None, fwhm=None):
        super().__init__(data, nside=nside, fwhm=fwhm)
        self._spectral_params = (self.beta,)


    def _get_freq_scaling(self, nu, beta):
        """
        Computes the frequency scaling from the reference frequency nu_ref to 
        an arbitrary frequency nu, which depends on the spectral parameter
        beta.

        Parameters
        ----------
        nu : int, float, numpy.ndarray
            Frequencies at which to evaluate the model. 
        nu_ref : int, float, numpy.ndarray
            Reference frequency.        
        beta : numpy.ndarray
            Synch beta map.
            
        Returns
        -------
        scaling : numpy.ndarray
            Frequency scaling factor.

        """
        nu_ref = np.expand_dims(self.params.nu_ref.si.value, axis=1)
        scaling = (nu/nu_ref)**beta

        return scaling
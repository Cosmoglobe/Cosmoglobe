import astropy.units as u

from .skycomponent import SkyComponent
from ..tools.utils import KRJ_to_KCMB, KCMB_to_KRJ


class CMB(SkyComponent):
    """Parent class for all CMB models."""

    comp_label = 'cmb'
    multipoles = (0, 1)
    
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.kwargs = kwargs

        if kwargs.get('remove_monopole', False):
            self.amp -= self.monopole

        if kwargs.get('remove_dipole', False):
            self.amp -= self.dipole


class BlackBody(CMB):
    """Model for BlackBody CMB emission."""

    model_label = 'cmb'

    def __init__(self, data, nside=None, fwhm=None,
                 remove_monopole=False, remove_dipole=False):

        super().__init__(data, nside=nside, fwhm=fwhm, 
                         remove_monopole=remove_monopole, 
                         remove_dipole=remove_dipole)


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
        emission = self._compute_emission(nu)

        return emission


    @u.quantity_input(nu=u.Hz)
    def _compute_emission(self, nu):
        """
        Computes the simulated emission CMB of at a given frequency .

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency at which to evaluate the CMB radiation.    
            
        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            CMB emission at a given frequency.

        """
        emission = self.amp.copy()
    
        return KCMB_to_KRJ(emission, nu).to(u.uK)
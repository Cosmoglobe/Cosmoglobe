from .skycomponent import SkyComponent
import astropy.units as u


class CMB(SkyComponent):
    """
    Parent class for all CMB models.

    """
    comp_label = 'cmb'

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

        self.monopole = data.get_alms('amp',
                                      self.comp_label, 
                                      self.params['nside'], 
                                      self.params['polarization'], 
                                      self.params['fwhm'],
                                      multipole=0)*u.uK
        self.dipole = data.get_alms('amp',
                                      self.comp_label, 
                                      self.params['nside'], 
                                      self.params['polarization'], 
                                      self.params['fwhm'],
                                      multipole=1)*u.uK

        if remove_monopole:
            self.amp -= self.monopole

        if remove_dipole:
            self.amp -= self.dipole



class BlackBody(CMB):
    """
    Model for BlackBody CMB emission.

    """    
    model_label = 'cmb'

    def __init__(self, data, remove_monopole=False, remove_dipole=False):

        super().__init__(data, remove_monopole=remove_monopole, 
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
        emission = self.compute_emission(nu)

        return emission


    @u.quantity_input(nu=u.Hz)
    def compute_emission(self, nu):
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

        return self.KCMB_to_KRJ(emission, nu)

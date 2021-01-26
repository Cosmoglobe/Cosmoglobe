from .skycomponent import SkyComponent
import astropy.units as u


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

    def __init__(self, data, fwhm=None):
        super().__init__(data, fwhm=fwhm)


    @u.quantity_input(nu=u.Hz)
    def get_emission(self, nu):
        """
        Returns the model emission of at a given frequency in units of K_RJ.

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity' (array)
            Frequencies at which to evaluate the model. 

        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Model emission at given frequency in units of K_RJ.

        """
        emission = self.compute_emission(nu, self.params['nu_ref'], self.beta)
        return emission

    @u.quantity_input(nu=u.Hz, nu_ref=u.Hz)
    def compute_emission(self, nu, nu_ref, beta):
        """
        Computes the simulated power law emission of synchrotron at a given
        frequency .

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency at which to evaluate the power law synchrotron radiation.
        nu_ref : 'astropy.units.quantity.Quantity'
            Reference frequency at which the Commander alm outputs are 
            produced.        
        beta : ndarray
            Estimated sycnh beta map from Commander.
            
        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Modeled modified blackbody emission at a given frequency.

        """
        emission = self.amp.copy()
        for i in range(len(emission)):
            emission[i] *= (nu/nu_ref[i])**beta[i]

        return emission



class CurvedPowerLaw(Synchrotron):
    """
    Model for curved power law emission for synchrotron.

    """
    model_label = 'curved_power_law' # should be replaced with commander param

    def __init__(self, data, model_params):
        super().__init__(data, model_params)


    @u.quantity_input(nu=u.Hz)
    def get_emission(self, nu):
        """
        Returns the model emission of at a given frequency in units of K_RJ.

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity' (array)
            Frequencies at which to evaluate the model. 

        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Model emission at given frequency in units of K_RJ.

        """
        emission = self.compute_emission(nu, self.params['nu_ref'], self.beta)
        return emission


    @u.quantity_input(nu=u.Hz, nu_ref=u.Hz)
    def compute_emission(self, nu, nu_ref, beta):
        """
        Computes the simulated power law emission of synchrotron at a given
        frequency .

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency at which to evaluate the power law synchrotron radiation.
        nu_ref : 'astropy.units.quantity.Quantity'
            Reference frequency at which the Commander alm outputs are 
            produced.        
        beta : ndarray
            Estimated sycnh beta map from Commander.
            
        Returns
        -------
        emission : 'astropy.units.quantity.Quantity'
            Modeled modified blackbody emission at a given frequency.

        """
        curvature = 0 # extract from data output once present
        emission = self.amp.copy()
        for i in range(len(emission)):
            emission[i] *= (nu/nu_ref[i])**(beta[i] + curvature)

        return emission
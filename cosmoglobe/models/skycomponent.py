import sys
import numpy as np
import healpy as hp
import astropy.units as u
import astropy.constants as const

# Initializing common astrophy units
h = const.h
c = const.c
k_B = const.k_B
T_0 = 2.7255*u.K


class SkyComponent:
    """
    Generalized template for all sky components.

    """

    def __init__(self, chain, **kwargs):
        """
        Initializes base model attributes and methods for a sky component.

        Parameters
        ----------
        chain : 'cosmoglobe.tools.chain.Chain'
            Commander3 chainfile object.
        **kwargs : dict
            Additional keyword value pairs. Valid keywords can vary from 
            component to component.

        """
        self.chain = chain
        self.params = chain.model_params[self.comp_label]

        if 'nside' in kwargs:
            self.params['nside'] = kwargs['nside']

        maps = self.initialize_maps()
        for key, value in maps.items():
            setattr(self, key, value)

        # print(self.__init__.__code__.co_varnames)

    def initialize_maps(self):
        """
        Initializes model maps from alms.

        Returns
        -------
        maps : dict
            Dictionary of loaded model maps.

        """
        maps = {}
        alm_list = self.chain.get_alm_list(self.comp_label)
        for alm in alm_list:
            alm_map = self.chain.get_alms(alm,
                                          self.comp_label, 
                                          self.params['nside'], 
                                          self.params['polarization'], 
                                          self.params['fwhm'])
            maps[alm] = alm_map
        
        maps = self.set_units(maps)
        return maps


    def to_nside(self, nside):
        """
        Reinitializes the model instance with at new nside.

        Parameters
        ----------
        nside: int
            Healpix map resolution.

        Returns
        -------
        Model instance with a new nside.
    
        """
        if hp.isnsideok(nside, nest=True):
            return super(self.__class__, self).__init__(self.chain, nside=nside)
        else:
            print(f'nside: {nside} is not valid.')
            sys.exit()


    def set_units(self, maps):
        """
        Sets units for alm maps.

        Parameters
        ----------
        maps : dict
            Dictionary of loaded model maps.

        Returns
        -------
        maps : dict
            Loaded maps with astropy units.
        
        """
        # Hardcoding units for model quantities
        units = {
            'amp':u.uK,
            'beta':None,
            'T':u.K,
        }
        if self.params['unit'] not in ['uK_RJ', 'uK_rj', 'uK_cmb', 'uK_CMB']:
            units['amp'] = self.params['unit']

        for alm in maps:
            if alm in units:
                if units[alm]:
                    maps[alm] *= units[alm]
        return maps


    @staticmethod
    @u.quantity_input(input_map=u.K, nu=u.Hz)
    def KRJ_to_KCMB(input_map, nu):
        """
        Converts input map from units of K_RJ to K_CMB.

        Parameters
        ----------
        input_map : 'astropy.units.quantity.Quantity'
            Healpix map in units of K_RJ.
        Returns
        -------
        'astropy.units.quantity.Quantity'
            Output map in units of K_CMB.

        """
        x = (h*nu)/(k_B*T_0)
        scaling_factor = (np.expm1(x)**2)/(x**2 * np.exp(x))
    
        return input_map*scaling_factor

    @staticmethod
    @u.quantity_input(input_map=u.K, nu=u.Hz)
    def KCMB_to_KRJ(input_map, nu):
        """
        Converts input map from units of K_CMB to K_RJ.

        Parameters
        ----------
        input_map : 'astropy.units.quantity.Quantity'
            Healpix map in units of K_RJ.
        Returns
        -------
        'astropy.units.quantity.Quantity'
            Output map in units of K_CMB.

        """
        x = (h*nu)/(k_B*T_0)
        scaling_factor = (np.expm1(x)**2)/(x**2 * np.exp(x))
        
        return input_map/scaling_factor


    @u.quantity_input(nu=u.Hz)
    def __getitem__(self, nu):
        """
        Returns simulated model emission at a given frequency.

        Parameters
        ----------
        nu : 'astropy.units.quantity.Quantity'
            Frequency at which to evaluate the model emission.

        Returns
        -------
        'astropy.units.quantity.Quantity'
            Model emission at the given frequency.

        """
        return self.get_emission(nu.to(u.GHz))


    @property
    def model_name(self):
        """
        Returns the name of the sky model.

        """
        return self.__class__.__name__


    def __repr__(self):
        """
        Unambigious representation of the sky model object.

        """
        return f"model('{self.model_label}')"    
        

    def __str__(self):
        """
        Readable representation of the sky model object.

        """
        return f"Cosmoglobe {self.comp_label} - {self.model_label} model."





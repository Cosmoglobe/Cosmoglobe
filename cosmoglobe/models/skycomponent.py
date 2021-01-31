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
    Generalized template for all sky models.

    """

    def __init__(self, chain, **kwargs):
        """
        Initializes model attributes and methods for a sky component.

        Parameters
        ----------
        chain : 'cosmoglobe.tools.chain.Chain'
            Commander3 chainfile object.
        **kwargs : dict
            Additional keywords. Valid keywords can vary from 
            component to component.

        """
        self.chain = chain
        self.params = self.chain.model_params[self.comp_label]

        attributes = self._get_model_attributes()
        self._set_model_attributes(attributes=attributes, 
                                   nside=kwargs.get('nside', None), 
                                   fwhm=kwargs.get('fwhm', None))


    def _get_model_attributes(self):
        """
        Returns a dictionary of maps and other model attributes.

        """
        attributes = {}
        attributes_list = self.chain.get_alm_list(self.comp_label)

        if hasattr(self, 'other_quantities'):
            attributes_list += self.other_quantities

        for attr in attributes_list:
            if 'alm' in attr:
                unpack_alms = True
                attr_name = attr.split('_')[0]
            else:
                unpack_alms = False  
                attr_name = attr

            item = self.chain.get_item(item_name=attr, 
                                       component=self.comp_label, 
                                       unpack_alms=unpack_alms, 
                                       multipoles=getattr(self, 'multipoles', None))

            if isinstance(item, dict):
                for key, value in item.items():
                    attributes[key] = value
            else:
                attributes[attr_name] = item

        return attributes


    def _set_model_attributes(self, attributes, nside, fwhm):
        """
        Sets model maps and quantities to instance attributes.

        Parameters
        ----------
        attributes : dict
            Dictionary of maps and other model attributes.
        nside : int
            Healpix map resolution. If not None, then maps are downgraded to
            nside with hp.ud_grade.
        fwhm : float
            The fwhm of the Gaussian used to smooth the map with hp.smoothing.

        """
        if nside is not None:
            if hp.isnsideok(nside, nest=True):
                ud_grade = True
            else:
                raise ValueError(f'nside: {nside} is not valid.')
        else: 
            ud_grade = False

        if fwhm is not None:
            if not isinstance(fwhm, u.Quantity):
                raise u.UnitsError(
                    'Fwhm must be an astropy object of units deg, rad, or arcmin'
                )
            smooth = True
        else:
            smooth = False
             
        for atr in attributes:
            if isinstance(attributes[atr], (list, np.ndarray)):
                if ud_grade:
                    attributes[atr] = hp.ud_grade(attributes[atr],
                                                  nside_out=int(nside), 
                                                  dtype=np.float64)
                if smooth:
                    attributes[atr] = hp.smoothing(attributes[atr], 
                                                   fwhm=fwhm.to('rad').value)
        
        for key, value in attributes.items():
            if any(attr in key for attr in ['amp', 'monopole', 'dipole']):
                if self.params['unit'].lower() in ['uk_rj', 'uk_cmb']:
                    value *= u.uK
                else: 
                    raise ValueError(
                        f'Unit {self.params["unit"]!r} is unrecognized'
                    )
            elif 'T' in key:
                value *= u.uK
            elif 'nu' in key:
                value *= u.GHz
            setattr(self, key, value)


    def _to_nside(self, nside):
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
            self.kwargs['nside'] = nside
            return super(self.__class__, self).__init__(self.chain, **self.kwargs)
        else:
            raise ValueError(f'nside: {nside} is not valid.')


    @staticmethod
    @u.quantity_input(input_map=u.K, nu=u.Hz)
    def KRJ_to_KCMB(input_map, nu):
        """
        Converts input map from units of K_RJ to K_CMB.

        Parameters
        ----------
        input_map : astropy.units.quantity.Quantity
            Healpix map in units of K_RJ.
        Returns
        -------
        astropy.units.quantity.Quantity
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
        input_map : astropy.units.quantity.Quantity
            Healpix map in units of K_RJ.
        Returns
        -------
        astropy.units.quantity.Quantity
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
        nu : astropy.units.quantity.Quantity
            Frequency at which to evaluate the model emission.

        Returns
        -------
        astropy.units.quantity.Quantity
            Model emission at the given frequency.

        """
        return self.get_emission(nu.to(u.GHz))


    @property
    def name(self):
        return self.__class__.__name__


    def __repr__(self):
        kwargs = {key:value for key, value in self.kwargs.items() if value is not None}
        kwargs_str = ", ".join(f"'{key}={value}'" for key, value in kwargs.items())
        return f'Cosmoglobe.model({self.comp_label!r}, {kwargs_str})'


    def __str__(self):
        return f"Cosmoglobe {self.comp_label} {self.__class__.__name__} model."





import astropy.units as u
import healpy as hp
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

from ..tools import utils

class SkyComponent:
    """
    Generalized template for all sky models.
    
    """
    
    def __init__(self, chain, **kwargs):
        """
        Initializes model attributes and methods for a sky component.

        Parameters
        ----------
        chain : cosmoglobe.tools.chain.Chain
            Commander3 chainfile object.
        kwargs : dict
            Additional keyword value pairs. Valid keywords can vary from 
            component to component.

        """
        self.chain = chain
        self.params = self.chain.model_params[self.comp_label]

        attributes = self._get_model_attributes(nside=kwargs['nside'], 
                                                fwhm=kwargs['fwhm'])
        self._set_model_attributes(attributes)


    @utils.nside_isvalid
    @u.quantity_input(fwhm=(u.arcmin, u.deg, u.rad, None))
    def _get_model_attributes(self, nside, fwhm):
        """
        Returns a dictionary of maps and other model attributes.
        
        Parameters
        ---------
        nside : int
            Healpix map resolution.
        fwhm : float
            The fwhm of the Gaussian used to smooth the map.

        """
        if nside is not None:
            self.params['nside'] = nside      

        if fwhm is not None:
            self.params['fwhm'] = fwhm

        attributes = {}
        attributes_list = self.chain.get_alm_list(self.comp_label)

        if hasattr(self, '_other_quantities'):
            attributes_list += self._other_quantities

        for attr in attributes_list:
            if 'alm' in attr:
                unpack_alms = True
                attr_name = attr.split('_')[0]
            else:
                unpack_alms = False  
                attr_name = attr

            item = self.chain.get_item(
                item_name=attr, 
                component=self.comp_label, 
                unpack_alms=unpack_alms, 
                multipoles=getattr(self, '_multipoles', None)
            )

            if not unpack_alms and nside is not None:
                item_nside = hp.npix2nside(len(item))
                if item_nside != nside:
                    item = hp.ud_grade(item, nside)

            if isinstance(item, dict):
                for key, value in item.items():
                    attributes[key] = value
            else:
                attributes[attr_name] = item

        return attributes


    def _set_model_attributes(self, attributes):
        """
        Sets model maps and quantities to instance attributes.

        Parameters
        ----------
        attributes : dict
            Dictionary of maps and other model attributes.

        """
        for key, value in attributes.items():
            if any(attr in key for attr in ('amp', 'monopole', 'dipole')):
                if self.params['unit'].lower() in ('uk_rj', 'uk_cmb'):
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


    @utils.timer
    def _get_mixing(self, bandpass, nus, spectral_params):
        """
        Returns the frequency scaling factor given a bandpass profile and a 
        set of spectral parameters. Uses the mixing matrix implementation 
        from commander3.

        Parameters
        ----------
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile array normalized to sum to one. Units must be 
            either K_RJ or K_CMB.
        nus : astropy.units.quantity.Quantity
            Frequency array corresponding to the bandpass array.
        spectral_parameters : tuple
            Tuple of all spectral parameters required to compute the frequency 
            scaling for a component i.e (self.beta, self.T).

        """

        n_interpols = 10

        polarized = self.params['polarization']

        interp_ranges, consts, interp_dim, interp_ind = (
            utils.get_interp_ranges(spectral_params, n_interpols)
        )


        if interp_dim == 0:
            freq_scaling = self._get_freq_scaling(nus, *consts)
            M = np.trapz(freq_scaling*bandpass, nus)

            return M

        elif interp_dim == 1:
            M = []
            for val in interp_ranges[0]:
                freq_scaling = self._get_freq_scaling(nus, val, *consts)
                M.append(np.trapz(freq_scaling*bandpass, nus))

            if polarized:
                M = np.transpose(M)
                f = [interp1d(interp_ranges[0], M[IQU]) for IQU in range(3)]
                M_interp = [f[IQU](spectral_params[interp_ind][IQU]) for IQU in range(3)]
            else:
                f = interp1d(interp_ranges[0], M) 
                M_interp = f(spectral_params[interp_ind]) 

            return M_interp

        elif interp_dim == 2:
            interp_meshgrid = np.meshgrid(*interp_ranges)
 
            if polarized:
                M = np.zeros((n_interpols, n_interpols, 3))
            else:
                M = np.zeros((n_interpols, n_interpols))

            for i in range(n_interpols):
                for j in range(n_interpols):
                    spec_inds = (interp_meshgrid[0][i,j], 
                                 interp_meshgrid[1][i,j])
                    freq_scaling = self._get_freq_scaling(nus, *spec_inds)
                    M[i,j] = np.trapz(freq_scaling*bandpass, nus)
            
            if polarized:
                M = np.transpose(M)
                f = [RectBivariateSpline(*interp_ranges, M[IQU]) 
                     for IQU in range(3)]
                M_interp = [f[IQU](spectral_params[0][IQU], 
                            spectral_params[1][IQU], grid=False) 
                            for IQU in range(3)]  
            else: 
                f = RectBivariateSpline(*interp_ranges, M) 
                M_interp = f(*spectral_params, grid=False) 

            return M_interp


    @u.quantity_input(nus=u.Hz, bandpass=(u.Jy/u.sr, u.K))
    def _get_unit_conversion(self, nus, bandpass):
        """
        Returns normalized integration weights from a bandpass profile. First
        converts to same units as the model amplitude, then normalizes.

        Parameters
        ----------
        nus : astropy.units.quantity.Quantity
            Array or list of frequencies for the bandpass profile.
        bandpass : astropy.units.quantity.Quantity
            Bandpass profile array or list. Must be in units of Jy/sr or K_RJ.

        Returns
        -------
        weights : numpy.ndarray
            Normalized bandpass profile to unit integral.
            
        """
        if np.shape(nus) != np.shape(bandpass):
            raise ValueError(
                'Frequency and bandpass arrays must have the same shape'
            )
        amp_unit = self.params['unit'].lower()

        if bandpass.unit == u.Jy/u.sr:
            if amp_unit.endswith('rj'):
                bandpass = bandpass.to(u.K, 
                    equivalencies=u.brightness_temperature(nus)
                )
            elif amp_unit.endswith('cmb'):
                bandpass = bandpass.to(u.K, 
                    equivalencies=u.thermodynamic_temperature(nus)
                )

        elif bandpass.unit == u.K:
            if amp_unit.endswith('cmb'):
                bandpass = utils.KRJ_to_KCMB(bandpass, nus)
        
        weights = bandpass/np.trapz(bandpass, nus.si.value)

        return weights.value


    @utils.nside_isvalid
    def to_nside(self, nside):
        """
        Reinitializes the model instance at new nside.

        Parameters
        ----------
        nside: int
            Healpix map resolution.

        Returns
        -------
        Model instance with a new nside.
    
        """
        self.kwargs['nside'] = nside
        return super(self.__class__, self).__init__(self.chain, **self.kwargs)


    @u.quantity_input(fwhm=(u.arcmin, u.deg, u.rad))
    def smooth(self, fwhm):
        """
        Reinitializes the model instance at new fwhm.

        Parameters
        ----------
        fwhm : float
            The fwhm of the Gaussian used to smooth the map.

        Returns
        -------
        Model instance with a new nside.
    
        """
        self.kwargs['fwhm'] = fwhm
        return super(self.__class__, self).__init__(self.chain, **self.kwargs)


    def to_KCMB(self):
        """
        Converts input map from units of K_RJ to K_CMB.
        
        """
        if self.params['unit'].lower() != 'uk_rj':
            raise ValueError(f'Unit is already {self.params["unit"]!r}')
        
        nu_ref = np.expand_dims(self.params['nu_ref'], axis=1)

        return utils.KRJ_to_KCMB(self.amp, nu_ref)


    def to_KRJ(self):
        """
        Converts input map from units of K_CMB to K_RJ.

        """
        if self.params['unit'].lower() != 'uk_cmb':
            raise ValueError(f'Unit is already {self.params["unit"]!r}')

        nu_ref = np.expand_dims(self.params['nu_ref'], axis=1)

        return utils.KCMB_to_KRJ(self.amp, nu_ref)


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
        kwargs = {key: value for key, value in self.kwargs.items() if value is not None}
        if kwargs:
            kwargs_str = ", ".join(f"'{key}={value}'" for key, value in kwargs.items())
            return f'model({self.comp_label!r}, {kwargs_str})'
        else:
            return f'model({self.comp_label!r})'


    def __str__(self):
        return f"Cosmoglobe {self.comp_label} {self.__class__.__name__} model."





import astropy.units as u
import healpy as hp
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from cosmoglobe.tools import chain
from cosmoglobe.tools.data import get_params_from_config
import pathlib
import sys

from ..tools import utils

class SkyComponent:
    """
    Generalized template for all sky components.
    
    """
    def __init__(self, data, **kwargs):
        """
        Initializes model attributes and methods for a sky component.

        Parameters
        ----------
        data : cosmoglobe.tools.chain.Chain
            Commander3 chainfile object or a config file.
        kwargs : dict
            Additional keyword value pairs.

        """
        self._from_chain = False
        self._from_config = False

        if isinstance(data, chain.Chain):
            self._from_chain = True
            self.chain = data
            self.params = self.chain.params[self.comp_label]
        
        elif isinstance(data, dict):
            self._from_config = True
            self.data = data
            self.params = get_params_from_config(self.data)[self.comp_label]

        attributes = self._get_model_attrs(nside=kwargs['nside'], 
                                            fwhm=kwargs['fwhm'])
        self._set_model_attributes(attributes)


    @utils.nside_isvalid
    @u.quantity_input(fwhm=(u.arcmin, u.deg, u.rad, None))
    def _get_model_attrs(self, nside, fwhm):
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
            self.params.nside = nside      

        if fwhm is not None:
            self.params.fwhm = fwhm

        attributes = {}

        if self._from_chain:
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
        
        elif self._from_config:
            attributes = {}
            for key, value in self.data[self.comp_label].items():
                if isinstance(value, str):
                    path = pathlib.Path(value)
                    if path.is_file():
                        item = hp.read_map(
                            path, field=(0,1,2), dtype=np.float32
                        )
                        if nside is not None:
                            item_nside = hp.npix2nside(len(item[0]))
                            if item_nside != nside:
                                item = hp.ud_grade(item, nside)  
                        if fwhm is not None:
                            item = hp.smoothing(item, fwhm=fwhm.to(u.rad).value)       
                        attributes[key] = item

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
                if self.params.unit.lower() in ('uk_rj', 'uk_cmb'):
                    value *= u.uK
                else: 
                    raise ValueError(
                        f'Unit {self.params.unit!r} is unrecognized'
                    )
            elif 'T' in key:
                value *= u.uK
            elif 'nu' in key:
                value *= u.GHz

            setattr(self, key, value)


    @u.quantity_input(nu=u.Hz, bandpass=(u.Jy/u.sr, u.K, None))
    def get_emission(self, nu, bandpass=None, output_unit=u.K):
        """
        Returns the full sky model emission at an arbitrary frequency nu in 
        units of K_RJ.

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
            scaling = self._get_freq_scaling(nu.si.value, *self._spectral_params)
            emission = self.amp*scaling

        else:
            bandpass = utils.get_normalized_bandpass(nu, bandpass)
            U = utils.get_unit_conversion(nu, bandpass, output_unit)
            M = self._get_mixing(bandpass=bandpass.value,
                                 nus=nu.si.value, 
                                 spectral_params=self._spectral_params)
            emission = self.amp*M*U

        return emission


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
        polarized = self.params.polarization

        n_interpols = 25
        interp_ranges, consts, interp_dim, interp_ind = (
            utils.get_interp_ranges(spectral_params, n_interpols)
        )

        if interp_dim == 0:
            freq_scaling = self._get_freq_scaling(nus, *consts)
            M_interp = np.trapz(freq_scaling*bandpass, nus)

            return M_interp

        elif interp_dim == 1:
            M = []
            for val in interp_ranges[0]:
                freq_scaling = self._get_freq_scaling(nus, val, *consts)
                M.append(np.trapz(freq_scaling*bandpass, nus))

            if polarized:
                M = np.transpose(M)
                f = [interp1d(interp_ranges[0], M[IQU]) for IQU in range(3)]
                M_interp = [f[IQU](spectral_params[interp_ind][IQU]) 
                            for IQU in range(3)]
            else:
                f = interp1d(interp_ranges[0], M) 
                M_interp = f(spectral_params[interp_ind]) 


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

        else:
            raise NotImplementedError(
                'Models with more than three spectral parameters are not '
                'currently supported'
            )

        return M_interp


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
        return self.__init__(self.chain, **self.kwargs)


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
        return self.__init__(self.chain, **self.kwargs)


    def to_KCMB(self):
        """
        Converts input map from units of K_RJ to K_CMB.
        
        """
        if self.params.unit.lower() != 'uk_rj':
            raise ValueError(f'Unit is already {self.params.unit!r}')
        
        nu_ref = np.expand_dims(self.params.nu_ref, axis=1)

        return utils.KRJ_to_KCMB(self.amp, nu_ref)


    def to_KRJ(self):
        """
        Converts input map from units of K_CMB to K_RJ.

        """
        if self.params.unit.lower() != 'uk_cmb':
            raise ValueError(f'Unit is already {self.params.unit!r}')

        nu_ref = np.expand_dims(self.params.nu_ref, axis=1)

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
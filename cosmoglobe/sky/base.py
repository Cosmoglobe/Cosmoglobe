from abc import ABC, abstractmethod
from typing import Dict
from sys import exit
import sys
import warnings

import astropy.units as u
import healpy as hp
import numpy as np
from tqdm import tqdm

from cosmoglobe.utils import utils
import cosmoglobe.utils.bandpass as bp

warnings.simplefilter('once', UserWarning)


class _Component(ABC):
    """Abstract base class for a sky component"""

    amp: u.Quantity
    freq_ref: u.Quantity
    spectral_parameters: Dict[str, u.Quantity]

    def __init__(self, amp, freq_ref, **spectral_parameters):
        """Initializes a sky component."""
        
        self.amp = self._reshape_amp(amp)
        self.freq_ref = self._reshape_freq_ref(freq_ref)
        self.spectral_parameters = self._reshape_spectral_parameters(
            spectral_parameters
        )

    @u.quantity_input(
        freqs=u.Hz, 
        bandpass=(u.Jy/u.sr, u.K, None), 
        fwhm=(u.rad, u.deg, u.arcmin),
    )
    def __call__(
        self, freqs, bandpass=None, fwhm=0.0 * u.rad, output_unit=u.uK
    ):
        r"""Simulates the component sky emission. 

        This method computes the full component sky emission for a single 
        frequency :math:`\nu` or integrated over a  bandpass :math:`\tau`.

        Parameters
        ----------
        freqs : `astropy.units.Quantity`
            A frequency, or a list of frequencies for which to evaluate 
            the sky emission.
        bandpass : `astropy.units.Quantity`, optional
            Bandpass profile corresponding to the frequencies. Default is 
            None. If `bandpass` is None and `freqs` is a single frequency,
            a delta peak is assumed (unless `freqs` is a list of 
            frequencies, for which a top-hat bandpass is used to perform  
            bandpass integration instead).
        fwhm : `astropy.units.Quantity`, optional
            The full width half max parameter of the Gaussian (Default is 
            0.0, which indicates no smoothing of output maps).
        output_unit : str, `astropy.units.UnitBase`, optional
            The desired output units of the emission. The supported units are
            :math:`\mathrm{\mu K_{RJ}}` and :math:`\mathrm{MJ/sr}` (By 
            default the output unit of the model is always in 
            :math:`\mathrm{\mu K_{RJ}}`. 

        Returns
        -------
        `astropy.units.Quantity`
            The component emission.

        Notes
        -----
        This function computes the following expression for a given 
        component:
        
        .. math::

            \boldsymbol{s}_i(\nu) = \boldsymbol{a}
            _i(\nu_{0,i}) \; f_{i}
            (\nu),

        where :math:`\boldsymbol{a}_i` is the amplitude map of 
        component :math:`i` at some reference frequency 
        :math:`\nu_{0,i}`, and :math:`f_i(\nu)` is the 
        scaling factor given by the SED of the component.

        See Also
        --------
        Model.__call__
            See for explicit expressions for :math:`\boldsymbol{s}_i(\nu)`.

        Examples
        --------
        Simulated dust emission at :math:`350\; \mathrm{GHz}`:

        >>> from cosmoglobe import skymodel
        >>> import astropy.units as u
        >>> model = skymodel(nside=256) 
        >>> model.dust(350*u.GHz)[0]
        [21.12408523 -0.30044442  5.8907252  ...  4.6465226   6.94981578
          8.23490773] uK

        Simulated synchrotron emission at :math:`500\; \mathrm{GHz}` 
        smoothed with a :math:`50\; '` Gaussian beam, outputed in units of 
        :math:`\mathrm{MJy} / \mathrm{sr}`:

        >>> model.synch(40*u.GHz, fwhm=50*u.arcmin, output_unit='MJy/sr')[0]
        [0.00054551 0.00053428 0.00051471 ... 0.00046559 0.00047185 0.00047065]
         MJy / sr
        """

        if freqs.size == 1:
            emission = self._get_delta_emission(freqs)
            emission = self._smooth_emission(emission, fwhm)
            if output_unit is not None:
                emission = utils.emission_to_unit(emission, freqs, output_unit)
        else:
            emission = self._get_bandpass_emission(
                freqs, bandpass, output_unit=output_unit
            )
            emission = self._smooth_emission(emission, fwhm)

        return emission

    def _get_delta_emission(self, freq):
        """Simulates the component emission at a delta frequency.
        
        Parameters
        ----------
        freq : `astropy.units.Quantity`
            A delta frequency.

        Returns
        -------
        `astropy.units.Quantity`
            Simulated emission.
        """

        scaling = self._get_freq_scaling(
            freq, self.freq_ref, **self.spectral_parameters
        )

        return self.amp * scaling

    def _get_bandpass_emission(self, freqs, bandpass=None, output_unit=u.uK):
        """Computes the simulated component emission over a bandpass.

        If no bandpass is passed, a top-hat bandpass is assumed.

        Parameters
        ----------
        freqs : `astropy.units.Quantity`
            Bandpass frequencies.
        bandpass : `astropy.units.Quantity`
            Bandpass profile. Default: None
        output_unit : `astropy.units.UnitBase`
            The desired output unit of the emission. Default: None

        Returns
        -------
        emission : `astropy.units.Quantity`
            Simulated emission.
        """

        if bandpass is None:
            warnings.warn('Bandpass is None. Defaulting to top-hat bandpass')
            bandpass = np.ones(freqs_len := len(freqs))/freqs_len * u.K

        bandpass = bp.get_normalized_bandpass(bandpass, freqs)
        bandpass_coefficient = bp.get_bandpass_coefficient(
            bandpass, freqs, output_unit
        )

        bandpass_scaling = bp.get_bandpass_scaling(freqs, bandpass, self)
        emission = self.amp * bandpass_scaling * bandpass_coefficient
        
        return emission

    @property
    def _is_polarized(self):
        """Returns True if component is polarized and False if not."""

        if self.amp.shape[0] == 3:
            return True

        return False

    @staticmethod
    def _reshape_amp(amp):
        """Reshapes the reference frequency to a broadcastable shape."""

        return amp if amp.ndim != 1 else np.expand_dims(amp, axis=0)

    @staticmethod
    def _reshape_freq_ref(freq_ref):
        """Reshapes the reference frequency to a broadcastable shape."""

        if freq_ref is None:
            return
        elif freq_ref.size == 1:
            return freq_ref
        elif freq_ref.size == 2:
            return np.expand_dims(
                u.Quantity([freq_ref[0], freq_ref[1], freq_ref[1]]), axis=1
            )
        elif freq_ref.size == 3:
            return freq_ref.reshape((3,1))
        else:
            raise ValueError('Unrecognized shape.')

    @staticmethod
    def _reshape_spectral_parameters(spectral_parameters):
        """Reshapes the spectral parameters to a broadcastable shape."""

        return {
            key: (np.expand_dims(value, axis=0) if value.ndim == 1 else value)
            for key, value in spectral_parameters.items()
        }

    @abstractmethod
    def _smooth_emission(self, emission, fwhm):
        """Smooths simulated emission given a beam FWHM.

        Parameters
        ----------
        emission : `astropy.units.Quantity`
            Simulated emission.
        fwhm : `astropy.units.Quantity`
            The full width half max parameter of the Gaussian. Default: 0.0

        Returns
        -------
        emission : `astropy.units.Quantity`
            Smoothed emission.
        """

    def __repr__(self):
        """Representation of the component."""

        main_repr = f'{self.__class__.__name__}'
        main_repr += '('
        extra_repr = ''
        for key in self.spectral_parameters.keys():
            extra_repr += f'{key}, '
        if extra_repr:
            extra_repr = extra_repr[:-2]
        main_repr += extra_repr
        main_repr += ')'

        return main_repr


class _DiffuseComponent(_Component):
    """Abstract base class for a diffuse sky component."""

    def __init__(self, amp, freq_ref, **spectral_parameters):
        super().__init__(amp, freq_ref, **spectral_parameters)

    @abstractmethod
    def _get_freq_scaling(freq, freq_ref, **spectral_parameters):
        r"""Computes the frequency scaling for a component.
        
        Parameters
        ----------
        freq : `astropy.units.Quantity`
            Frequency at which to evaluate the model.
        freq_ref : `astropy.units.Quantity`
            Reference frequencies for the components `amp` map.
        spectral_parameters : `np.ndarray`, `astropy.units.Quantity`
            The spectral parameters of the component. 
            
        Returns
        -------
        scaling : `astropy.units.Quantity`
            Frequency scaling factor with dimensionless units.
        """

    def _smooth_emission(self, emission, fwhm):
        """See base class."""
        
        if fwhm == 0.0:
            return emission
        
        fwhm = (fwhm.to(u.rad)).value
        if self._is_polarized:
            emission = u.Quantity(
                hp.smoothing(emission, fwhm=fwhm), unit=emission.unit
            )
        else:
            emission[0] = u.Quantity(
                hp.smoothing(emission[0], fwhm=fwhm), unit=emission.unit
            )

        return emission


class _PointSourceComponent(_Component):
    """Abstract base class for a point source sky component."""

    def __init__(self, amp, freq_ref, nside, **spectral_parameters):
        super().__init__(amp, freq_ref, **spectral_parameters)

        # Point source components explicitly require a nside attribute 
        # since the amplitudes are not stored in healpix maps.
        self.nside = nside

    @abstractmethod
    def _get_freq_scaling(freq, freq_ref, **spectral_parameters):
        r"""Computes the frequency scaling for a component.
        
        Parameters
        ----------
        freq : `astropy.units.Quantity`
            Frequency at which to evaluate the model.
        freq_ref : `astropy.units.Quantity`
            Reference frequencies for the components `amp` map.
        spectral_parameters : `np.ndarray`, `astropy.units.Quantity`
            The spectral parameters of the component. 
            
        Returns
        -------
        scaling : `astropy.units.Quantity`
            Frequency scaling factor with dimensionless units.
        """

    def _smooth_emission(self, emission, fwhm):
        """See base class. 
        
        In addition to smoothing the point sources, this method also maps 
        the cataloged points to a healpix map in the process.
        """

        return self._points_to_map(emission, fwhm=fwhm)

    def _points_to_map(self, amp, fwhm, n_fwhm=2, verbose=False):
        """Maps the cataloged point sources onto a healpix map with a truncated 
        gaussian beam. For more information, see 
         `Mitra et al. (2010) <https://arxiv.org/pdf/1005.1929.pdf>`_.

        Parameters
        ----------
        amp : `astropy.units.Quantity`
            Amplitudes of the point sources.
        fwhm : `astropy.units.Quantity`
            The full width half max parameter of the Gaussian. Default: 0.0
        n_fwhm : int, float
            The fwhm multiplier used in computing radial cut off r_max 
            calculated as r_max = n_fwhm * fwhm of the Gaussian.
            Default: 2.
        nside : int
            The healpix map resolution of the output map. If component is part 
            of a sky model, we automatically select the model nside. 
            Default: None.
        """

        nside = self.nside
        angular_coords = self.angular_coords
        
        if amp.ndim > 1:
            amp = np.squeeze(amp)
        healpix_map = u.Quantity(
            np.zeros(hp.nside2npix(nside)), unit=amp.unit
        )
        pix_lon, pix_lat = hp.pix2ang(
            nside, np.arange(hp.nside2npix(nside)), lonlat=True
        )

        fwhm = fwhm.to(u.rad)
        sigma = fwhm / (2*np.sqrt(2*np.log(2)))

        if sigma == 0.0:
            # No smoothing nesecarry. We directly map the sources to pixels
            warnings.warn('mapping point sources to pixels without beam smoothing.')
            pixels = hp.ang2pix(nside, *angular_coords.T, lonlat=True)
            beam_area = hp.nside2pixarea(nside) * u.sr
            healpix_map[pixels] = amp
            healpix_map /= beam_area.value

        else:
            # Apply a truncated gaussian beam to each point source
            pix_res = hp.nside2resol(nside)
            if fwhm.value < pix_res:
                raise ValueError(
                    'fwhm must be >= pixel resolution to resolve the '
                    'point sources.'
                )
            beam_area = 2 * np.pi * sigma**2
            r_max = n_fwhm * fwhm.value

            with tqdm(
                total=len(angular_coords), file=sys.stdout, disable=not verbose
            ) as pbar:
                sigma = sigma.value
                beam = utils.gaussian_beam_2D
                print('Smoothing point sources...')

                for idx, (lon, lat) in enumerate(angular_coords):
                    vec = hp.ang2vec(lon, lat, lonlat=True)
                    inds = hp.query_disc(nside, vec, r_max)
                    r = hp.rotator.angdist(
                        np.array(
                            [pix_lon[inds], pix_lat[inds]]), np.array([lon, lat]
                        ),
                        lonlat=True
                    )
                    healpix_map[inds] += amp[idx] * beam(r, sigma)
                    pbar.update()

        healpix_map /= beam_area.value
        return np.expand_dims(healpix_map, axis=0)

    def _read_coords_from_catalog(self, catalog):
        """Reads in the angular coordinates of the point sources from a 
        given catalog.

        Parameters
        ----------
        catalog : str
            Path to the point source catalog. Default is the COM_GB6 
            catalog.
        
        Returns
        -------
        coords : `numpy.ndarray`
            Longitude and latitude values of each point source
        """
        
        try:
            coords = np.loadtxt(catalog, usecols=(0,1))
        except OSError:
            raise OSError('Could not find point source catalog')

        if len(coords) != len(self.amp[0]):
            raise ValueError('Cataloge does not match chain catalog')

        return coords
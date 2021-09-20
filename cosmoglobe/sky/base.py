from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import warnings

import astropy.units as u
import healpy as hp
import numpy as np

from cosmoglobe.utils.utils import gaussian_beam_2D, to_unit
from cosmoglobe.utils.bandpass import (
    get_bandpass_coefficient,
    get_bandpass_scaling,
    get_normalized_bandpass,
)

warnings.simplefilter("once", UserWarning)


class SkyComponent(ABC):
    """Abstract base class for a sky component.

    This class implements the `__call__` function which is responsible for
    simulating and returning the sky emission. Additionally, it enforces
    the implementation of a `_smooth_emission` function for any sub class.
    """

    def __init__(
        self,
        amp: u.Quantity,
        freq_ref: u.Quantity,
        **spectral_parameters: Dict[str, u.Quantity],
    ) -> None:
        """Initializes a sky component and reshape inputs."""

        self._amp = self._reshape_amp(amp)
        self._freq_ref = self._reshape_freq_ref(freq_ref)
        self._spectral_parameters = self._reshape_spectral_parameters(
            spectral_parameters
        )
        self._is_polarized = True if self._amp.shape[0] == 3 else False

    @property
    def amp(self) -> u.Quantity:
        """Amplitude map the sky component."""

        return self._amp

    @property
    def freq_ref(self) -> u.Quantity:
        """Reference frequencies of the amplitude map."""

        return self._freq_ref

    @property
    def spectral_parameters(self) -> Dict[str, u.Quantity]:
        """Spectral parameters."""

        return self._spectral_parameters

    @abstractmethod
    def _smooth_emission(self, emission: u.Quantity, fwhm: u.Quantity) -> u.Quantity:
        """Smooths simulated emission given a beam FWHM.

        Parameters
        ----------
        emission
            Simulated emission.
        fwhm
            The full width half max parameter of the Gaussian. Default: 0.0

        Returns
        -------
        emission
            Smoothed emission.
        """

    @u.quantity_input(
        freqs=u.Hz,
        bandpass=(u.Jy / u.sr, u.K, None),
        fwhm=(u.rad, u.deg, u.arcmin),
    )
    def __call__(
        self,
        freqs: u.Quantity,
        bandpass: Optional[u.Quantity] = None,
        fwhm: u.Quantity = 0.0 * u.rad,
        output_unit: Union[str, u.UnitBase] = u.uK,
    ) -> u.Quantity:
        r"""Simulates and returns the sky emission of a component.

        This method computes the sky emission of a component for a single
        frequency :math:`\nu` or integrated over a  bandpass :math:`\tau`.

        Parameters
        ----------
        freqs
            A frequency, or a list of frequencies for which to evaluate
            the sky emission.
        bandpass
            Bandpass profile corresponding to the frequencies. Default is
            None. If `bandpass` is None and `freqs` is a single frequency,
            a delta peak is assumed (unless `freqs` is a list of
            frequencies, for which a top-hat bandpass is used to perform
            bandpass integration instead).
        fwhm
            The full width half max parameter of the Gaussian (Default is
            0.0, which indicates no smoothing of output maps).
        output_unit
            The desired output units of the emission. The supported units are
            :math:`\mathrm{\mu K_{RJ}}` and :math:`\mathrm{MJ/sr}` (By
            default the output unit of the model is always in
            :math:`\mathrm{\mu K_{RJ}}`.

        Returns
        -------
        Emission
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

        if np.size(freqs) == 1:
            emission = self._get_delta_emission(freqs)
            emission = self._smooth_emission(emission, fwhm)
            if output_unit is not None:
                emission = to_unit(emission, freqs, output_unit)
        else:
            emission = self._get_bandpass_emission(
                freqs, bandpass, output_unit=output_unit
            )
            emission = self._smooth_emission(emission, fwhm)

        return emission

    def _get_delta_emission(self, freq: u.Quantity) -> u.Quantity:
        """Simulates the component emission at a delta frequency."""

        scaling = self._get_freq_scaling(
            freq, self.freq_ref, **self.spectral_parameters
        )

        return self.amp * scaling

    def _get_bandpass_emission(
        self,
        freqs: u.Quantity,
        bandpass: u.Quantity = None,
        output_unit: u.UnitBase = u.uK,
    ) -> u.Quantity:
        """Simulates the component emission at over a bandpass."""

        if bandpass is None:
            warnings.warn("Bandpass is None. Defaulting to top-hat bandpass")
            bandpass = np.ones(freqs_len := len(freqs)) / freqs_len * u.K

        bandpass = get_normalized_bandpass(bandpass, freqs)
        bandpass_coefficient = get_bandpass_coefficient(bandpass, freqs, output_unit)

        bandpass_scaling = get_bandpass_scaling(freqs, bandpass, self)
        emission = self.amp * bandpass_scaling * bandpass_coefficient

        return emission

    @staticmethod
    def _reshape_amp(amp: u.Quantity) -> u.Quantity:
        """Reshapes the reference frequency to a broadcastable shape."""

        return amp if amp.ndim != 1 else np.expand_dims(amp, axis=0)

    @staticmethod
    def _reshape_freq_ref(freq_ref: u.Quantity) -> u.Quantity:
        """Reshapes the reference frequency to a broadcastable shape."""

        if freq_ref is None:
            return
        elif freq_ref.size == 1:
            if np.ndim(freq_ref) == 0:
                return np.expand_dims(freq_ref, axis=0)
            return freq_ref
        elif freq_ref.size == 2:
            return np.expand_dims(
                u.Quantity([freq_ref[0], freq_ref[1], freq_ref[1]]), axis=1
            )
        elif freq_ref.size == 3:
            return freq_ref.reshape((3, 1))
        else:
            raise ValueError("Unrecognized shape")

    @staticmethod
    def _reshape_spectral_parameters(
        spectral_parameters: Dict[str, u.Quantity]
    ) -> Dict[str, u.Quantity]:
        """Reshapes the spectral parameters to a broadcastable shape."""

        return {
            key: (np.expand_dims(value, axis=0) if value.ndim == 1 else value)
            for key, value in spectral_parameters.items()
        }

    def __repr__(self) -> str:
        """Representation of the component."""

        main_repr = f"{self.__class__.__name__}"
        main_repr += "("
        extra_repr = ""
        for key in self.spectral_parameters.keys():
            extra_repr += f"{key}, "
        if extra_repr:
            extra_repr = extra_repr[:-2]
        main_repr += extra_repr
        main_repr += ")"

        return main_repr


class Diffuse(SkyComponent):
    """Abstract base class for diffuse sky components."""

    @abstractmethod
    def _get_freq_scaling(
        freq: u.Quantity,
        freq_ref: u.Quantity,
        **spectral_parameters: Dict[str, u.Quantity],
    ) -> u.Quantity:
        r"""Computes the frequency scaling for a component.

        Parameters
        ----------
        freq
            Frequency at which to evaluate the model.
        freq_ref
            Reference frequencies for the components `amp` map.
        spectral_parameters
            The spectral parameters of the component.

        Returns
        -------
        scaling
            Frequency scaling factor with dimensionless units.
        """

    def _smooth_emission(self, emission: u.Quantity, fwhm: u.Quantity) -> u.Quantity:
        """See base class."""

        if fwhm == 0.0:
            return emission

        fwhm = (fwhm.to(u.rad)).value
        if self._is_polarized:
            emission = u.Quantity(hp.smoothing(emission, fwhm=fwhm), unit=emission.unit)
        else:
            emission[0] = u.Quantity(
                hp.smoothing(emission[0], fwhm=fwhm), unit=emission.unit
            )

        return emission


class PointSource(SkyComponent):
    """Abstract base class for point source sky components."""

    _nside: int = None

    @abstractmethod
    def _get_freq_scaling(
        freq: u.Quantity,
        freq_ref: u.Quantity,
        **spectral_parameters: Dict[str, u.Quantity],
    ) -> u.Quantity:
        r"""Computes the frequency scaling for a component.

        Parameters
        ----------
        freq
            Frequency at which to evaluate the model.
        freq_ref
            Reference frequencies for the components `amp` map.
        spectral_parameters
            The spectral parameters of the component.

        Returns
        -------
        scaling
            Frequency scaling factor with dimensionless units.
        """

    def _smooth_emission(self, emission: u.Quantity, fwhm: u.Quantity) -> u.Quantity:
        """See base class.

        In addition to smoothing the point sources, this method also maps
        the cataloged points to a healpix map in the process.
        """

        return self._points_to_map(emission, fwhm=fwhm)

    def _points_to_map(
        self, amp: u.Quantity, fwhm: u.Quantity, n_fwhm: int = 2, verbose: bool = False
    ) -> u.Quantity:
        """Maps the cataloged point sources onto a healpix map with a truncated
        gaussian beam. For more information, see
         `Mitra et al. (2010) <https://arxiv.org/pdf/1005.1929.pdf>`_.

        Parameters
        ----------
        amp
            Amplitudes of the point sources.
        fwhm
            The full width half max parameter of the Gaussian. Default: 0.0
        n_fwhm
            The fwhm multiplier used in computing radial cut off r_max
            calculated as r_max = n_fwhm * fwhm of the Gaussian.
            Default: 2.
        """

        nside = self._nside
        if nside is None:
            raise AttributeError(
                "The _nside attribute of a point source have not been set"
            )

        angular_coords = self.angular_coords

        if amp.ndim > 1:
            amp = np.squeeze(amp)

        healpix_map = u.Quantity(np.zeros(hp.nside2npix(nside)), unit=amp.unit)
        pix_lon, pix_lat = hp.pix2ang(
            nside, np.arange(hp.nside2npix(nside)), lonlat=True
        )

        fwhm = fwhm.to(u.rad)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        if sigma == 0.0:
            # No smoothing nesecarry. We directly map the sources to pixels
            warnings.warn("Mapping point sources to pixels without beam smoothing.")
            pixels = hp.ang2pix(nside, *angular_coords.T, lonlat=True)
            beam_area = hp.nside2pixarea(nside) * u.sr
            healpix_map[pixels] = amp
            healpix_map /= beam_area.value

        else:
            # Apply a truncated gaussian beam to each point source
            pix_res = hp.nside2resol(nside)
            if fwhm.value < pix_res:
                raise ValueError(
                    "fwhm must be >= pixel resolution to resolve the " "point sources."
                )
            beam_area = 2 * np.pi * sigma ** 2
            r_max = n_fwhm * fwhm.value

            sigma = sigma.value
            beam = gaussian_beam_2D
            print("Smoothing point sources...")

            for idx, (lon, lat) in enumerate(angular_coords):
                vec = hp.ang2vec(lon, lat, lonlat=True)
                inds = hp.query_disc(nside, vec, r_max)
                r = hp.rotator.angdist(
                    np.array([pix_lon[inds], pix_lat[inds]]),
                    np.array([lon, lat]),
                    lonlat=True,
                )
                healpix_map[inds] += amp[idx] * beam(r, sigma)

        healpix_map /= beam_area.value

        return np.expand_dims(healpix_map, axis=0)

    def _read_coords_from_catalog(self, catalog: str) -> np.ndarray:
        """Reads in the angular coordinates of the point sources from a
        given catalog.

        Parameters
        ----------
        catalog
            Path to the point source catalog. Default is the COM_GB6
            catalog.

        Returns
        -------
        coords
            Longitude and latitude values of each point source
        """

        try:
            coords = np.loadtxt(catalog, usecols=(0, 1))
        except OSError:
            raise OSError("Could not find point source catalog")

        if len(coords) != len(self.amp[0]):
            raise ValueError("Cataloge does not match chain catalog")

        return coords

from typing import Dict, Protocol, Union

import astropy.units as u
from astropy.units import Quantity, Unit

from cosmoglobe.refactored_sky.enums import SkyComponentLabel, SkyComponentType


class SpectralEnergyDistribution(Protocol):
    """Interface for SEDs."""

    def get_freq_scaling(
        self,
        freqs: u.GHz,
        freq_ref: u.GHz,
        **spectral_parameters: Quantity,
    ) -> Quantity:
        """Returns the scale factor for a given SED

        Parameters
        ----------
        freqs
            Frequencies for which to evaluate the SED.
        spectral_parameters
            Parameters describing the SED.

        Returns
        -------
            Frequency scaling factor.
        """


class SkyComponent(Protocol):
    """Interface for a sky component."""

    label: SkyComponentLabel
    component_type: SkyComponentType
    amp: u.uK
    freq_ref: u.GHz
    SED: SpectralEnergyDistribution
    spectral_parameters: Dict[str, Quantity]


class SkySimulation(Protocol):
    """Protocol defining how a component behaves when used in simulations."""

    def delta(
        self,
        component: SkyComponent,
        freq: u.GHz,
        output_unit: Union[str, Unit],
        *,
        nside: int,
        fwhm: u.rad,
        catalog: Quantity,
    ) -> Quantity:
        """Computes and returns the emission for a delta frequency.

        Parameters
        ----------
        component
            A SkyComponent.
        freq
            Frequency for which to compute the delta emission.
        nside
            nside of the HEALPIX map.
        fwhm
            The full width half max parameter of the Gaussian.
        output_unit
            The requested output units of the simulated emission.

        Returns
        -------
            Simulated delta frequency emission.
        """

    def bandpass(
        self,
        component: SkyComponent,
        freqs: u.GHz,
        bandpass: Quantity,
        output_unit: Union[str, Unit],
        *,
        nside: int,
        fwhm: u.rad,
        catalog: Quantity,
    ) -> Quantity:
        """Computes and returns the emission for a bandpass profile.

        Parameters
        ----------
        freqs
            An array of frequencies for which to compute the bandpass emission
            over.
        bandpass
            An array of bandpass weights corresponding the frequencies in the
            frequencies array.
        nside
            nside of the HEALPIX map.
        fwhm
            The full width half max parameter of the Gaussian.
        output_unit
            The requested output units of the simulated emission.

        Returns
        -------
            Integrated bandpass emission.
        """

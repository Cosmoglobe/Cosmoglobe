from typing import Dict, Protocol, Union

import astropy.units as u
from astropy.units import Quantity, Unit

from cosmoglobe.refactored_sky.enums import SkyComponentLabel, SkyComponentType



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

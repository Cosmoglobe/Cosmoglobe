from typing import Dict, Union
import warnings

import astropy.units as u
from astropy.units import Quantity, Unit
import numpy as np

from cosmoglobe.utils.bandpass import get_bandpass_scaling
from cosmoglobe.utils.utils import to_unit

from cosmoglobe.refactored_sky.protocols import SkyComponent, SkySimulation
from cosmoglobe.refactored_sky.enums import SkyComponentType
from cosmoglobe.refactored_sky.constants import DEFAULT_OUTPUT_UNIT
from cosmoglobe.refactored_sky._bandpass import (
    get_normalized_bandpass,
    get_bandpass_coefficient,
)
from cosmoglobe.refactored_sky.beam import pointsources_to_healpix


class DiffuseSimulation:
    """Simulation protocol for diffuse components."""

    def delta(
        self,
        component: SkyComponent,
        freq: u.GHz,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""

        emission = component.amp * component.SED.get_freq_scaling(
            freq, component.freq_ref, **component.spectral_parameters
        )

        if output_unit != DEFAULT_OUTPUT_UNIT:
            emission = to_unit(emission, freq, output_unit)

        return emission

    def bandpass(
        self,
        component: SkyComponent,
        freqs: u.GHz,
        bandpass: Quantity,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""

        if bandpass is None:
            warnings.warn("Bandpass is None. Defaulting to top-hat bandpass")
            bandpass = (
                np.ones(freqs_len := len(freqs)) / freqs_len * DEFAULT_OUTPUT_UNIT
            )

        bandpass = get_normalized_bandpass(freqs, bandpass)

        bandpass_scaling = get_bandpass_scaling(freqs, bandpass, component)
        emission = component.amp * bandpass_scaling
        input_unit = emission.unit
        unit_coefficient = get_bandpass_coefficient(
            freqs, bandpass, input_unit, output_unit
        )

        return emission * unit_coefficient


class PointSourceSimulation:
    """Simulation protocol for Point Source components."""

    def delta(
        self,
        component: SkyComponent,
        freq: u.GHz,
        output_unit: Union[str, Unit],
        nside: int,
        fwhm: u.rad,
        catalog: Quantity,
    ) -> Quantity:
        """See base class."""

        point_sources = component.amp * component.SED.get_freq_scaling(
            freq, **component.spectral_parameters
        )

        emission = pointsources_to_healpix(point_sources, catalog, nside, fwhm)
        emission = to_unit(emission, freq, output_unit)

        return emission

    def bandpass(
        self,
        component: SkyComponent,
        freqs: u.GHz,
        bandpass: Quantity,
        output_unit: Union[str, Unit],
        nside: int,
        fwhm: u.rad,
        catalog: Quantity,
    ) -> Quantity:
        """See base class."""

        if bandpass is None:
            warnings.warn("Bandpass is None. Defaulting to top-hat bandpass")
            bandpass = (
                np.ones(freqs_len := len(freqs)) / freqs_len * DEFAULT_OUTPUT_UNIT
            )

        bandpass = get_normalized_bandpass(freqs, bandpass)
        bandpass_scaling = get_bandpass_scaling(freqs, bandpass, component)
        point_sources = component.amp * bandpass_scaling
        emission = pointsources_to_healpix(point_sources, catalog, nside, fwhm)
        input_unit = emission.unit
        unit_coefficient = get_bandpass_coefficient(
            freqs, bandpass, input_unit, output_unit
        )

        return emission * unit_coefficient


class LineSimulation:
    """Simulation protocol for Line components."""

    def delta(
        self,
        component: SkyComponent,
        freq: u.GHz,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""

    def bandpass(
        self,
        component: SkyComponent,
        freqs: u.GHz,
        bandpass: Quantity,
        output_unit: Union[str, Unit],
        **_,
    ) -> Quantity:
        """See base class."""


SIMULATION_PROTOCOLS: Dict[SkyComponentType, SkySimulation] = {
    SkyComponentType.DIFFUSE: DiffuseSimulation(),
    SkyComponentType.POINTSOURCE: PointSourceSimulation(),
    SkyComponentType.LINE: LineSimulation(),
}


def get_simulation_protocol(component: SkyComponent) -> SkySimulation:
    """Returns a simulation protocol given a component."""

    try:
        return SIMULATION_PROTOCOLS[component.component_type]
    except KeyError:
        raise NotImplementedError(
            "simulation protocol not implemented for comp of type "
            f"{component.__class__.__name__}"
        )

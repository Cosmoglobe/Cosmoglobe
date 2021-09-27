from typing import List, Optional, Union

from astropy.units import Quantity, Unit, quantity_input
import numpy as np
import healpy as hp

from cosmoglobe.sky.simulation_strategies import get_simulation_protocol
from cosmoglobe.sky.base_components import (
    SkyComponent,
    PointSourceComponent,
    DiffuseComponent,
)
from cosmoglobe.sky._constants import DEFAULT_OUTPUT_UNIT, NO_SMOOTHING
from cosmoglobe.utils.utils import str_to_astropy_unit


class SkySimulator:
    """Sky simulation interface."""

    @quantity_input(
        freqs=Unit("Hz"),
        bandpass=[Unit("K"), Unit("Jy/sr"), Unit("1/Hz")],
        fwhm=[Unit("rad"), Unit("arcmin"), Unit("deg")],
    )
    def __call__(
        self,
        nside: int,
        components: List[SkyComponent],
        freqs: Quantity,
        bandpass: Optional[Quantity] = None,
        fwhm: Quantity = NO_SMOOTHING,
        output_unit: Union[str, Unit] = DEFAULT_OUTPUT_UNIT,
    ) -> Quantity:
        """Simulates the diffuse sky emission."""

        diffuse_comps = [
            comp for comp in components if isinstance(comp, DiffuseComponent)
        ]
        pointsource_comps = [
            comp for comp in components if isinstance(comp, PointSourceComponent)
        ]

        shape = (
            (3, hp.nside2npix(nside))
            if any(comp.amp.shape[0] == 3 for comp in components)
            else (1, hp.nside2npix(nside))
        )
        emission = Quantity(
            np.zeros(shape),
            unit=output_unit
            if isinstance(output_unit, Unit)
            else str_to_astropy_unit(output_unit),
        )

        for diffuse_comp in diffuse_comps:
            comp_emission = self.simulate_component_emission(
                nside,
                diffuse_comp,
                freqs,
                bandpass=bandpass,
                output_unit=output_unit,
                fwhm=fwhm,
            )
            for idx, row in enumerate(comp_emission):
                emission[idx] += row

        if fwhm.value != NO_SMOOTHING:
            fwhm_rad = fwhm.to("rad").value
            if shape[0] == 3:
                emission = Quantity(
                    hp.smoothing(emission, fwhm=fwhm_rad), unit=emission.unit
                )
            else:
                emission = Quantity(
                    np.expand_dims(
                        hp.smoothing(np.squeeze(emission), fwhm=fwhm_rad), axis=0
                    ),
                    unit=emission.unit,
                )

        for pointsource_comp in pointsource_comps:
            comp_emission = self.simulate_component_emission(
                nside,
                pointsource_comp,
                freqs,
                bandpass=bandpass,
                output_unit=output_unit,
                fwhm=fwhm,
            )
            for idx, row in enumerate(comp_emission):
                emission[idx] += row

        return emission

    def simulate_component_emission(
        self,
        nside: int,
        component: SkyComponent,
        freqs: Quantity,
        bandpass: Optional[Quantity],
        output_unit: Union[str, Unit],
        fwhm: Optional[Quantity],
    ) -> Quantity:
        """Returns the simulated sky emission for a component."""

        simulation_strategies = get_simulation_protocol(component)

        if freqs.size > 1:
            if bandpass is not None and freqs.shape != bandpass.shape:
                raise ValueError("freqs and bandpass must have the same shape")

            return simulation_strategies.bandpass(
                component,
                freqs,
                bandpass,
                output_unit=output_unit,
                fwhm=fwhm,
                nside=nside,
            )

        return simulation_strategies.delta(
            component, freqs, output_unit=output_unit, fwhm=fwhm, nside=nside
        )


simulator = SkySimulator()

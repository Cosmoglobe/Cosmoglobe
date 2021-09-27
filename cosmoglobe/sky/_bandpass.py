from typing import Callable, Dict, Union

from astropy.units import Quantity, Unit, UnitTypeError
import numpy as np

from cosmoglobe.utils.utils import str_to_astropy_unit
from cosmoglobe.sky._intensity_derivatives import INTENSITY_DERIVATIVE_MAPPINGS


def get_normalized_bandpass(freqs: Quantity, bandpass: Quantity) -> Quantity:
    """Normalizes a bandpass to units of unity under integration."""

    return bandpass / np.trapz(bandpass, freqs)


def get_bandpass_coefficient(
    freqs: Quantity,
    bandpass: Quantity,
    input_unit: Union[str, Unit],
    output_unit: Union[str, Unit],
) -> Quantity:
    """Returns the bandpass coefficient."""

    if isinstance(input_unit, str):
        input_unit = str_to_astropy_unit(input_unit)
    if isinstance(output_unit, str):
        output_unit = str_to_astropy_unit(output_unit)

    units = {"in": input_unit, "out": output_unit}

    intensity_derivatives: Dict[str, Callable[[Quantity], Quantity]] = {}

    for key, unit in units.items():
        for intensity_deriv in INTENSITY_DERIVATIVE_MAPPINGS:
            intensity_unit = str_to_astropy_unit(intensity_deriv)
            if intensity_unit._is_equivalent(unit):
                intensity_derivatives[key] = INTENSITY_DERIVATIVE_MAPPINGS[
                    intensity_deriv
                ]
                break
        else:
            raise UnitTypeError("unrecognized unit for intensity derivatives")

    coefficient = np.trapz(
        bandpass * intensity_derivatives["in"](freqs), freqs
    ) / np.trapz(bandpass * intensity_derivatives["out"](freqs), freqs)

    return coefficient

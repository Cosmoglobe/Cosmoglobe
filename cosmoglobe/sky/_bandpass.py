from typing import Union

from astropy.units import Quantity, Unit
import numpy as np

from cosmoglobe.sky._intensity_derivative import get_intensity_derivative


def get_normalized_bandpass(freqs: Quantity, bandpass: Quantity) -> Quantity:
    """Normalizes a bandpass to units of unity under integration.

    Parameters
    ----------
    freqs
        Frequencies corresponding to bandpass weights.
    bandpass
        The bandpass profile.

    Returns
    -------
        Bandpass with unit integral over np.trapz.
    """

    return bandpass / np.trapz(bandpass, freqs)


def get_bandpass_coefficient(
    freqs: Quantity,
    bandpass: Quantity,
    input_unit: Union[str, Unit],
    output_unit: Union[str, Unit],
) -> Quantity:
    """Returns the bandpass coefficient.

    Parameters
    ----------
    freqs
        Frequencies corresponding to bandpass weights.
    bandpass
        The bandpass profile.
    input_unit
        The unit of the cosmoglobe data.
    ouput_unit
        The requested output unit of the simulated emission

    Returns
    -------
    coefficient
        Bandpass coefficient representing the unitconversion between the
        input/ouput unit over the bandpass.
    """

    if isinstance(input_unit, str):
        in_unit = input_unit
    else:
        in_unit = input_unit.to_string().replace(" ", "")
    if isinstance(output_unit, str):
        out_unit = output_unit
    else:
        out_unit = output_unit.to_string().replace(" ", "")

    in_intensity_derivative = get_intensity_derivative(in_unit)
    out_intensity_derivative = get_intensity_derivative(out_unit)

    coefficient = np.trapz(bandpass * in_intensity_derivative(freqs), freqs) / np.trapz(
        bandpass * out_intensity_derivative(freqs), freqs
    )

    return coefficient

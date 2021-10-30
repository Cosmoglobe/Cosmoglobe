from dataclasses import dataclass
from typing import Dict, Optional

from astropy.units.quantity import Quantity
import numpy as np

from cosmoglobe.refactored_sky.enums import SkyComponentLabel, SkyComponentType
from cosmoglobe.refactored_sky.SEDs import SpectralEnergyDistribution


@dataclass
class SkyComponent:
    """Abstract base class for a sky component."""

    label: SkyComponentLabel
    type: SkyComponentType
    amp: Quantity
    freq_ref: Quantity
    spectral_parameters: Optional[Dict[str, Quantity]] = None
    SED: Optional[SpectralEnergyDistribution] = None
    catalog: Optional[np.ndarray] = None


@dataclass
class Diffuse(SkyComponent):
    """Diffuse Sky component"""

    spectral_parameters: Optional[Dict[str, Quantity]] = None
    SED: Optional[SpectralEnergyDistribution] = None
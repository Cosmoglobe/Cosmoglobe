from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Protocol

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import NDArray


class FreqScalingFunc(Protocol):
    def __call__(
        self,
        freqs: Quantity[u.Hz] | Quantity[u.m],
        freq_ref: Quantity[u.Hz] | Quantity[u.m],
        **spectral_params: dict[str, float | Quantity | NDArray[np.float_]],
    ) -> float | NDArray[np.float_]:
        ...


@dataclass
class SkyComponent(ABC):
    amp_ref: Quantity[u.K] | Quantity[u.KR_J] | Quantity[u.Jy / u.sr]
    freq_ref: Quantity[u.Hz] | Quantity[u.m]
    spectral_params: dict[str, float | Quantity | NDArray[np.float_]]
    freq_scaling_func: FreqScalingFunc

    def get_delta_emission(
        self, freqs: Quantity[u.Hz] | Quantity[u.m]
    ) -> Quantity[u.K] | Quantity[u.KR_J] | Quantity[u.Jy / u.sr]:
        ...

    def get_bandpass_emission(
        self, freqs: Quantity[u.Hz] | Quantity[u.m], weights: NDArray[np.float_]
    ) -> Quantity[u.K] | Quantity[u.KR_J] | Quantity[u.Jy / u.sr]:
        ...


class Diffuse(SkyComponent):
    def get_delta_emission(
        self, freqs: Quantity[u.Hz] | Quantity[u.m]
    ) -> Quantity[u.K] | Quantity[u.KR_J] | Quantity[u.Jy / u.sr]:

        scale_factor = self.freq_scaling_func(
            freqs=freqs, 
            freq_ref=self.freq_ref,
            **self.spectral_params
        )
        return self.amp_ref * scale_factor

    def get_bandpass_emission(
        self, freqs: Quantity[u.Hz] | Quantity[u.m], weights: NDArray[np.float_]
    ) -> Quantity[u.K] | Quantity[u.KR_J] | Quantity[u.Jy / u.sr]:
        ...


@dataclass
class PointSource(SkyComponent):
    point_source_catalog: Quantity[u.deg]

    def get_delta_emission(
        self, freqs: Quantity[u.Hz] | Quantity[u.m]
    ) -> Quantity[u.K] | Quantity[u.KR_J] | Quantity[u.Jy / u.sr]:

        scale_factor = self.freq_scaling_func(
            freqs=freqs, 
            freq_ref=self.freq_ref,
            **self.spectral_params
        )

        scaled_source_points = self.amp_ref * scale_factor

        emission = bin_point_sources(
            
        )
        return self.amp_ref * scale_factor

    def get_bandpass_emission(
        self, freqs: Quantity[u.Hz] | Quantity[u.m], weights: NDArray[np.float_]
    ) -> Quantity[u.K] | Quantity[u.KR_J] | Quantity[u.Jy / u.sr]:
        ...
from dataclasses import dataclass
from enum import Enum, auto
from band import Band
from parameter_collection import ParameterCollection

@dataclass
class SmoothingScaleParameters:
    fwhm: float
    fwhm_postproc: float
    lmax: int
    nside: int
    pixwin: str

@dataclass
class DatasetParameters(ParameterCollection):
    data_directory: str = None
    include_bands: list[Band] = None
    processing_maskfile: str = None
    source_maskfile: str = None
    smoothing_scales: list[SmoothingScaleParameters] = None

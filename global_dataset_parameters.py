from dataclasses import dataclass
from enum import Enum, auto
from band import Band

@dataclass
class SmoothingScaleParameters:
    fwhm: float
    fwhm_postproc: float
    lmax: int
    nside: int
    pixwin: str

@dataclass
class GlobalDatasetParameters:
    data_directory: str
    include_bands: list[Band]
    processing_maskfile: str
    source_maskfile: str
    smoothing_scales: list[SmoothingScaleParameters]

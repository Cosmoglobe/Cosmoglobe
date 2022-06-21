from __future__ import annotations

from enum import Enum, auto
from pydantic import BaseModel

from band import Band

class SmoothingScaleParameters(BaseModel):
    fwhm: float
    fwhm_postproc: float
    lmax: int
    nside: int
    pixwin: str

class DatasetParameters(BaseModel):
    data_directory: str = None
    include_bands: list[Band] = None
    processing_maskfile: str = None
    source_maskfile: str = None
    smoothing_scales: list[SmoothingScaleParameters] = None

from __future__ import annotations

from enum import Enum, auto
from functools import partial
from pydantic import BaseModel

from band import Band

class SmoothingScaleParameters(BaseModel):
    fwhm: float
    fwhm_postproc: float
    lmax: int
    nside: int
    pixwin: str

class DatasetParameters(BaseModel):
    """
    A container for the general Commander parameters that define the data used
    by Commander in any way. Also contains a list of Band containers, which
    contain band-specific parameters.
    """
    data_directory: str = None
    include_bands: list[Band] = None
    processing_maskfile: str = None
    source_maskfile: str = None
    smoothing_scales: list[SmoothingScaleParameters] = None

    @classmethod
    def _get_parameter_handling_dict(cls):

        def default_handler(field_name, paramfile_dict):
            paramfile_param = field_name.upper()
            try:
                return paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: Dataset parameter {} not found in parameter file".format(e))
                return None

        def band_handler(field_name, paramfile_dict):
            bands = []
            num_bands = int(paramfile_dict['NUMBAND'])
            for i in range(1, num_bands+1):
                bands.append(Band.create_band_params(paramfile_dict, i))
            return bands

        def smoothing_scale_handler(field_name, paramfile_dict):
            pass

        field_names = cls.__fields__.keys()
        handling_dict = {}
        for field_name in field_names:
            if field_name == 'include_bands':
                handling_dict[field_name] = partial(band_handler, field_name)
            elif field_name == 'smoothing_scales':
                handling_dict[field_name] = partial(
                    smoothing_scale_handler, field_name)
            else:
                handling_dict[field_name] = partial(
                    default_handler, field_name)
        return handling_dict

    @classmethod
    def create_dataset_params(cls, paramfile_dict):
        handling_dict = cls._get_parameter_handling_dict()
        param_vals = {}
        for field_name, handling_function in handling_dict.items():
            param_vals[field_name] = handling_function(paramfile_dict)
        return DatasetParameters(**param_vals)

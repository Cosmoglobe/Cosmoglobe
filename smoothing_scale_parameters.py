from __future__ import annotations

from functools import partial

from pydantic import BaseModel

class SmoothingScaleParameters(BaseModel):
    fwhm: float
    fwhm_postproc: float
    lmax: int
    nside: int
    pixwin: str

    @classmethod
    def _get_parameter_handling_dict(cls):
        """
        Create a mapping between the container field names and the appropriate
        functions to handle those fields.

        The functions in the output dict will take a parameter file dictionary
        and the smoothing scale number, and will return whatever is appropriate
        for that field. 

        Output:
            dict[str, Callable]: Mapping between field names and their handlers.
        """

        def default_handler(field_name, paramfile_dict, smoothing_scale_num):
            paramfile_param = (
                'SMOOTHING_SCALE_' + field_name.upper() +
                '{:02d}'.format(smoothing_scale_num))
            try:
                return paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: Smoothing scale parameter {} not found in parameter file".format(e))

        field_names = cls.__fields__.keys()
        handling_dict = {}
        for field_name in field_names:
            handling_dict[field_name] = partial(
                default_handler, field_name)
        return handling_dict
        
    @classmethod
    def create_smoothing_scale_params(
            cls,
            paramfile_dict: dict[str, str],
            smoothing_scale_num: int) -> SmoothingScaleParameters:
        """
        Factory class method for a SmoothingScaleParameters instance.

        Input:
            paramfile_dict[str, str]: A dict (typically created by
                parameter_parser._paramfile_to_dict) mapping the keys found in
                a Commander parameter file to the values found in that same
                file.
        Output:
            SmoothingScaleParameters: Parameter container for parameters
            specific to a given Commander smoothing scale instance.
        """
        handling_dict = cls._get_parameter_handling_dict()
        param_vals = {}
        for field_name, handling_function in handling_dict.items():
            param_vals[field_name] = handling_function(paramfile_dict,
                                                       smoothing_scale_num)
        return SmoothingScaleParameters(**param_vals)

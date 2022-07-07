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
            paramfile_dict: dict[str, Any],
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

    def serialize_to_paramfile_dict(self, smoothing_scale_num):
        """
        Creates a mapping from Commander parameter names to the values in the
        SmoothingScaleParameters instance, with all lower-level parameter
        collections similarly serialized.

        Note the values in this mapping are basic types, not strings. This
        means they will have to be processed further before they are ready for
        a Commander parameter file. The keys, however, need no more processing.

        Input:
            smoothing_scale_num[int]: The number of the smoothing scale
            instance in the Commander file context.

        Output:
            dict[str, Any]: Mapping from Commander parameter file names to the
                parameter values.
        """

        paramfile_dict = {}
        for field_name, value in self.__dict__.items():
            paramfile_dict[
                'SMOOTHING_SCALE_{}{:02}'.format(field_name.upper(),
                                                 smoothing_scale_num)] = value
        return paramfile_dict

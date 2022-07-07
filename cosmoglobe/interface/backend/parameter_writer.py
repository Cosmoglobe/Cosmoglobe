from __future__ import annotations

from enum import Enum

from general_parameters import GeneralParameters

class ParameterWriter:
    """
    Class responsible for writing parameters in a GeneralParameters instance to file.


    General mode of operation: Will call the
    GeneralParameters.serialize_to_paramfile_dict() instance method, which
    returns a parameter name to value mapping. This class is responsible for
    validating/cleaning the values in this dict so that they make sense for
    Commander. The keys of the dict are already in the right format.
    """

    def __init__(self, general_parameters: GeneralParameters):
        self.general_parameters = general_parameters

    def _stringify_param(self, param: Any) -> str:
        """
        Creates a "Commander-acceptable" string from the input parameter value.

        Can handle strings, Nones, bools, floats, ints, and Enums.

        Input:
            param (Any): The parameter value to input into a Commander
                parameter file.

        Returns:
            string: The Commander-prepared version of the parameter value.
        """
        if isinstance(param, str):
            return param
        if param is None:
            return 'none'
        if isinstance(param, bool):
            return '.{}.'.format(str(param).lower())
        if isinstance(param, float):
            return ('{:e}'.format(param)).lower().replace('e', 'd')
        if isinstance(param, int):
            return str(param)
        if isinstance(param, Enum):
            return param.value
        else:
            return param

    def write_paramfile(self, fname: str):
        """
        Write the GeneralParameters instance provided upon initialization to
        file.

        Input:
            fname (str): The filename of the written file.
        """
        paramfile_dict = self.general_parameters.serialize_to_paramfile_dict()

        with open(fname, 'w') as f:
            for key, value in paramfile_dict.items():
                f.write("{} = {}\n".format(key, self._stringify_param(value)))

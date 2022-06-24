from typing import Union

from general_parameters import GeneralParameters

class ParameterParser:
    """
    This class is responsible for a) parsing a native Commander parameter file
    and b) generating an instantiated data structure which contains the
    information in the parameter file.

    The way the data structures currently work is that the top parameter
    container is a GeneralParameters instance. This contains parameters
    represented as floats and strings etc, as well as other, more specialized
    parameter containers:

                        GeneralParameters
                          /          \
                         /            \
                        /              \
            ModelParameters       DatasetParameters
                 /                          \
                /                            \
          Component 1                      Band 1
          Component 2                      Band 2
          Component 3                      Band 3
           .                                 .
           .                                 .
           .                                 .

    Each of these nodes contain parameters specific to them, in addition to
    point to the nodes further down, and are implemented as inheritors of the
    pydantic BaseModels. The Component and Band classes are parameter
    containers that will typically have more than one instance. Some of the
    parameters are small-ish classes in their own right, not detailed here. For
    example, there is a MonopolePrior class used by the Component classes that
    contain all the information for monopole priors.
    """

    def __init__(self, paramfile, defaults_dir):
        self.defaults_dir = defaults_dir
        self.paramfile_dict = self._paramfile_to_dict(paramfile=paramfile)
        self.context = None

    def _process_line(self, line: str) -> dict[str, Union[str, None]]:
        """
        Turn a single line from a Commander parameter file into a key, value
        pair.

        If the line is empty, or a comment, an empty dict is returned.
        If the line is an '@' directive, either
            - an empty dict is returned if the directive is START or END
            - a dict containing all parameters in the indicated defaults file
              if the directive is DEFAULT
        The key, value pairs are generally the same as the input lines,
        verbatim, with these exceptions:
         -  If the value would be '.true.' or '.false.' (i.e a Fortran bool), it will
            output 'true' and 'false', respectively.
         -  If the value would be 'none', it will output a python None
         -  If the value would be a float containing a 'd', it will output the
            same float but with an 'e' instead (e.g "2.7d0" -> "2.7e0")
         -  If the key contains '&&' or '&&&', it will replace those with the
            current directive context (indicated by @START).
        The above replacements are mostly done in order to make these strings
        cooperate with the pydantic type converter/validator.

        Input:
            line (str): A single line from a Commander parameter file.

        Output:
            dict[Str, Str|None]: A dict where the keys are the parameter names
                and the values are their values. See above for details on what
                exactly will be output.
        """
        params = {}
        if line.strip().startswith('#') or line.strip().startswith('*') or line.strip() == '' or line.strip().startswith('\n'):
            return params
        elif line.strip().startswith('@'):
            if line.strip().startswith('@DEFAULT'):
                k, v = line.split(' ')[:2]
                params.update(
                    self._paramfile_to_dict('{}/{}'.format(self.defaults_dir, v.strip())))
            elif line.strip().startswith('@START'):
                k, v = line.split(' ')[:2]
                self.context = int(v.strip())
            elif line.strip().startswith('@END'):
                self.context = None
            return params

        k, v = line.split("=")[:2]
        k = k.strip()
        v = v.strip().split(' ')[0]
        if '&&&' in k:
            k = k.replace('&&&', '{:03d}'.format(self.context))
        elif '&&' in k:
            k = k.replace('&&', '{:02d}'.format(self.context))
        # Changing fortran-style bools to something that can be parsed by pydantic
        if v.startswith('.') and v.endswith('.'):
            v = v[1:-1]
        # Changing fortran-style scientific notation to something that can be
        # parsed by pydantic
        if len(v) > 2 and 'd' in v.lower() and (v[0].isdigit() or v[1].isdigit()):
            v = v.lower().replace('d', 'e')
        if v.lower() == 'none':
            v = None
        params[k] = v
        return params


    def _paramfile_to_dict(self, paramfile: str) -> dict[str, Union[str, None]]:
        """
        Creates a dictionary from a Commander parameter file.

        Input:
            paramfile (str): The full path to the parameter file.
        Output:
            dict[str, str|None]: A mapping of all parameters (anything marked
                with a '=') in the parameter file to their values. Any @DEFAULT
                directives are followed, and @START and @END contexts are used
                to replace ampersands. See the docs for
                ParameterParser._process_line for more info.
        """
        params = {}
        with open(paramfile, 'r') as f:
            line = f.readline()
            while line:
                params.update(self._process_line(line))
                line = f.readline()
        return params

    def classify_params(self):
        return GeneralParameters.create_gen_params(self.paramfile_dict)

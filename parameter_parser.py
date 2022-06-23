from typing import Union

from general_parameters import GeneralParameters
from dataset_parameters import DatasetParameters
from model_parameters import ModelParameters
from band import Band
from cg_sampling_group import CGSamplingGroup
from component import Component, MonopolePrior, MonopolePriorType


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


    def get_collection_to_paramfile_mapping(self,
                                            parameter_collection,
                                            prepend_string='',
                                            append_string='') -> dict[str, str]:
        """
        Get a mapping of internal parameter collection parameters to
        parameterfile parameters.

        Generally, we assume that the names of the fields in the parameter
        collections are the same as the parameter file names in uppercase. It
        is possible to prepend and append this with custom strings.

        Input:
            parameter_collection: An BaseModel instance representing some
                parameter collection (see the documentation for the
                ParameterParser class).
            prepend_string (str): String to be prepended to the parameter file
                parameter string.
            append_string (str): String to be appended to the parameter file
                parameter string.

        Output:
            dict[str, str]: A mapping from the collection field name to the
                parameterfile parameter name, with the optional strings
                pre-or-appended to the latter.

        """
        parameter_mapping = {}
        parameter_list = (parameter_collection.__fields__.keys())

        for parameter in parameter_list:
            parameter_mapping[parameter] = (prepend_string +
                                            parameter.upper() +
                                            append_string)

        return parameter_mapping


    def create_gen_params(self):
        """
        Creates a GeneralParameters instance given the parameter file provided
        when the ParameterParser was instantiated.


        Output:
            GeneralParameters: Container of the top-level Commander
                parameterfile parameters, as well as the lower-level parameter
                containers. See the ParameterParser docs for more information.
        """
        param_mapping = self.get_collection_to_paramfile_mapping(GeneralParameters)
        param_vals = {}
        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'init_chains':
                num_init_chains = self.paramfile_dict['NUM_INIT_CHAINS']
                init_chain_list = []
                for i in range(int(num_init_chains)):
                    init_chain_list.append(self.paramfile_dict['INIT_CHAIN{:02d}'.format(i+1)])
                param_vals[collection_param] = init_chain_list
                continue
            elif collection_param == 'model_parameters':
                param_vals[collection_param] = self.create_model_params()
                continue
            elif collection_param == 'dataset_parameters':
                param_vals[collection_param] = self.create_dataset_params()
                continue
            try:
                param_vals[collection_param] = self.paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: General parameter {} not found in parameter file".format(e))

        return GeneralParameters(**param_vals)


    def create_model_params(self):
        """
        Creates a ModelParameters instance given the parameter file provided
        when the ParameterParser was instantiated.


        Output:
            ModelParameters: Container of the model-specific Commander
                parameterfile parameters, as well as the Component parameter
                containers that belong to it. See the ParameterParser docs for
                more information.
        """

        param_vals = {}
        param_mapping = self.get_collection_to_paramfile_mapping(ModelParameters)
        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'cg_sampling_groups':
                # This must be done *after* the components have been created and
                # populated in order to point to them.
                continue
            if collection_param == 'signal_components':
                param_vals['signal_components'] = []
                num_components = int(self.paramfile_dict['NUM_SIGNAL_COMPONENTS'])
                for i in range(1, num_components+1):
                    param_vals['signal_components'].append(
                        self.create_component_params(i))
                continue
            try:
                param_vals[collection_param] = self.paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: Model parameter {} not found in parameter file".format(e))

        # Populate CG sampling groups
        param_vals['cg_sampling_groups'] = []
        num_sampling_groups = int(self.paramfile_dict['NUM_CG_SAMPLING_GROUPS'])
        for i in range(1, num_sampling_groups+1):
            param_vals['cg_sampling_groups'].append(
                self.create_cg_sampling_group(i, param_vals['signal_components']))

        return ModelParameters(**param_vals)


    def create_cg_sampling_group(self,
                                 cg_sampling_group_num: int,
                                 components: list[Component]) -> CGSamplingGroup:
        """
        Creates a CG sampling group instance given the parameter file provided
        when the ParameterParser was instantiated.

        Input:
            cg_sampling_group_num (int): The number associated with this group
                in the parameter file
            components list[Component]: An already instantiated list of
                Component parameter containers. The output CGSamplingGroup
                instance will link to the relevant components in this list.

        Output:
            CGSamplingGroup: Container of Commander parameterfile parameters
                pertaining to a specific cg sampling group (typically indicated by
                CG_SAMPLING_GROUP_***_&& parameter names).
        """

        param_vals = {}

        param_mapping = self.get_collection_to_paramfile_mapping(
            CGSamplingGroup,
            'CG_SAMPLING_GROUP_', '{:02d}'.format(cg_sampling_group_num))

        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'components': continue
            try:
                param_vals[collection_param] = self.paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: CG sampling group parameter {} not found in parameter file".format(e))
        # Link the components to the sampling groups
        compnames = self.paramfile_dict['CG_SAMPLING_GROUP{:02d}'.format(cg_sampling_group_num)]
        compnames = compnames.replace("'", "")
        compnames = compnames.split(',')
        param_vals['components'] = []
        for compname in compnames:
            for component in components:
                if component.label == compname:
                    param_vals['components'].append(component)
                    break
            else:
                raise ValueError("Component {} specified in CG sampling group but is not defined in parameter file")
        return CGSamplingGroup(**param_vals)


    def create_dataset_params(self):
        """
        Creates a DatasetParameters instance given the parameter file provided
        when the ParameterParser was instantiated.

        Output:
            DatasetParameters: Container of the dataset-specific Commander
                parameterfile parameters, as well as the Band parameter
                containers that belong to it. See the ParameterParser docs for
                more information.
        """

        param_vals = {}
        param_mapping = self.get_collection_to_paramfile_mapping(DatasetParameters)
        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'include_bands':
                param_vals['include_bands'] = []
                num_bands = int(self.paramfile_dict['NUMBAND'])
                for i in range(1, num_bands+1):
                    param_vals['include_bands'].append(self.create_band_params(i))
                continue
            if collection_param == 'smoothing_scales':
                continue
            try:
                param_vals[collection_param] = self.paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: Dataset parameter {} not found in parameter file".format(e))

        return DatasetParameters(**param_vals)
    

    def create_band_params(self, band_num):
        """
        Creates a Band instance given the parameter file provided when the
        ParameterParser was instantiated.

        Output:
            Band: Container of Commander parameterfile parameters pertaining to
                a specific band (typically indicated by BAND_***_&&& parameter
                names).
        """

        param_vals = {}

        param_mapping = self.get_collection_to_paramfile_mapping(
            Band, 'BAND_', '{:03d}'.format(band_num))
        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'noise_rms_smooth':
                i = 1
                param_vals[collection_param] = []
                while ('BAND_NOISE_RMS{:03d}_SMOOTH{:02d}'.format(band_num, i)
                       in self.paramfile_dict.keys()):
                    param_vals[collection_param].append(
                        self.paramfile_dict[
                            'BAND_NOISE_RMS{:03d}_SMOOTH{:02d}'.format(band_num, i)])
                    i+=1
                continue
            if collection_param == 'tod_detector_list':
                param_vals[collection_param] = self.paramfile_dict[paramfile_param].split(',')
                continue
            if collection_param == 'include_band':
                param_vals[collection_param] = self.paramfile_dict['INCLUDE_BAND{:03d}'.format(band_num)]
                continue
            try:
                param_vals[collection_param] = self.paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: Band parameter {} not found in parameter file".format(e))
        return Band(**param_vals)


    def _parse_monoprior_params(self, monoprior_string):
        """
        Creates a MonopolePrior instance given the monopole prior parameter
        string found in a Commander parameter file.

        Output:
            MonopolePrior: Contains the parameters relevant for a single
                MonopolePrior definition in the Commander parameter file.
        """
        if monoprior_string == 'none' or monoprior_string is None:
            return None
        monoprior_type, monoprior_params = monoprior_string.split(':')
        monoprior_params = monoprior_params.split(',')
        monoprior_type = MonopolePriorType(monoprior_type)
        if monoprior_type == MonopolePriorType.BANDMONO:
            monopole_prior = MonopolePrior(type=monoprior_type,
                                           label=monoprior_params[0])
        elif monoprior_type == MonopolePriorType.CROSSCORR:
            monopole_prior = MonopolePrior(type=monoprior_type,
                                           corrmap=monoprior_params[0],
                                           nside=monoprior_params[1],
                                           fwhm=monoprior_params[2].replace('d', 'e'),
                                           thresholds=monoprior_params[3:])
        elif monoprior_type == MonopolePriorType.MONOPOLE_MINUS_DIPOLE:
            monopole_prior = MonopolePrior(type=monoprior_type,
                                           mask=monoprior_params[0])
        return monopole_prior


    def create_component_params(self, component_num):
        """
        Creates a Component instance given the parameter file provided when the
        ParameterParser was instantiated.

        Output:
            Component: Container of Commander parameterfile parameters
                pertaining to a specific component (typically indicated by
                COMP_***_&& parameter names).
        """

        param_vals = {}

        param_mapping = self.get_collection_to_paramfile_mapping(
            Component, 'COMP_', '{:02d}'.format(component_num))
        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'ctype':
                param_vals[collection_param] = self.paramfile_dict[
                    "COMP_TYPE{:02d}".format(component_num)]
                continue
            if collection_param == 'cclass':
                param_vals[collection_param] = self.paramfile_dict[
                    "COMP_CLASS{:02d}".format(component_num)]
                continue
            if collection_param == 'monopole_prior':
                try:
                    param_vals[collection_param] = self._parse_monoprior_params(
                        self.paramfile_dict[paramfile_param])
                except KeyError as e:
                    print("Warning: Component parameter {} not found in parameter file".format(e))
                continue
            try:
                param_vals[collection_param] = self.paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: Component parameter {} not found in parameter file".format(e))
        return Component(**param_vals)

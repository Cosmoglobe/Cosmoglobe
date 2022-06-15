from general_parameters import GeneralParameters
from dataset_parameters import DatasetParameters
from model_parameters import ModelParameters
from dataclasses import asdict


class ParameterParser:

    def __init__(self, paramfile, defaults_dir):
        self.paramfile = paramfile
        self.defaults_dir = defaults_dir
        self.paramfile_dict = self._paramfile_to_dict()
        print(self.paramfile_dict)

    def _paramfile_to_dict(self):
        params = {}
        with open(self.paramfile, 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('#') or line.startswith('*') or line.strip() == '' or line.startswith('\n'):
                    line = f.readline()
                    continue
                elif line.startswith('@'):
                    line = f.readline()
                    continue
                print(line)
                k, v = line.split("=")[:2]
                k = k.strip()
                v = v.strip().split(' ')[0]
                params[k] = v
                line = f.readline()
        return params


    def get_general_collection_paramfile_mapping(self, general_parameters):
        general_parameter_mapping = {}
        general_parameter_list = asdict(general_parameters).keys()

        for parameter in general_parameter_list:
            general_parameter_mapping[parameter] = parameter.upper()

        return general_parameter_mapping

#    def get_dataset_collection_paramfile_mapping(self):
#        pass
#
#    def get_model_collection_paramfile_mapping(self):
#        pass
#

    def create_gen_params(self):
        gen_params = GeneralParameters()
        param_mapping = self.get_general_collection_paramfile_mapping(gen_params)
        for collection_name, paramfile_name in param_mapping.items():
            if collection_name == 'init_chains':
                num_init_chains = self.paramfile_dict['NUM_INIT_CHAINS']
                init_chain_list = []
                for i in range(int(num_init_chains)):
                    init_chain_list.append(self.paramfile_dict['INIT_CHAIN{:02d}'.format(i+1)])
                gen_params.set_param(collection_name, init_chain_list)
                continue
            try:
                gen_params.set_param(collection_name, self.paramfile_dict[paramfile_name])
            except KeyError as e:
                print("Warning: {} not found in parameter file".format(e))

#        # if some logic
#            gen_params.dataset_params = self.create_dataset_params()
#            gen_params.model_params = self.create_model_params()
        return gen_params


#    def create_model_params(self):
#        model_params = ModelParameters()
#        param_mapping = ModelParameters.get_parameter_repr_dict()
#        for paramfile_name, collection_name in param_mapping:
#            model_params.set_param(collection_name, self.paramfile_dict[paramfile_name])
#        # if some logic
#            for i in num_components:
#                model_params.bands.append(self.create_band(...))
#
#    def create_dataset_params(self):
#        dataset_params = DatasetParameters()
#        param_mapping = DatasetParameters.get_parameter_repr_dict()
#        for paramfile_name, collection_name in param_mapping:
#            dataset_params.set_param(collection_name, self.paramfile_dict[paramfile_name])
#        # if some logic
#            for i in num_bands:
#                dataset_params.components.append(self.create_component(...))

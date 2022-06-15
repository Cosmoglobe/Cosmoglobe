from general_parameters import GeneralParameters
from dataset_parameters import DatasetParameters
from model_parameters import ModelParameters
from dataclasses import asdict


class ParameterParser:

    def __init__(self, paramfile, defaults_dir):
        self.defaults_dir = defaults_dir
        self.paramfile_dict = self._paramfile_to_dict(paramfile=paramfile)
        self.context = None

    def _process_line(self, line):
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
        if k.endswith('&&'):
            if k.endswith('&&&'):
                k = '{}{:03d}'.format(k[:-3], self.context)
            else:
                k = '{}{:02d}'.format(k[:-2], self.context)
        params[k] = v
        return params


    def _paramfile_to_dict(self, paramfile):
        params = {}
        with open(paramfile, 'r') as f:
            line = f.readline()
            while line:
                params.update(self._process_line(line))
                line = f.readline()
        return params


    def get_collection_to_paramfile_mapping(self, parameter_collection):
        parameter_mapping = {}
        parameter_list = asdict(parameter_collection).keys()

        for parameter in parameter_list:
            parameter_mapping[parameter] = parameter.upper()

        return parameter_mapping


    def create_gen_params(self):
        gen_params = GeneralParameters()
        param_mapping = self.get_collection_to_paramfile_mapping(gen_params)
        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'init_chains':
                num_init_chains = self.paramfile_dict['NUM_INIT_CHAINS']
                init_chain_list = []
                for i in range(int(num_init_chains)):
                    init_chain_list.append(self.paramfile_dict['INIT_CHAIN{:02d}'.format(i+1)])
                gen_params.set_param(collection_param, init_chain_list)
                continue
            elif collection_param == 'model_parameters':
                gen_params.set_param(collection_param, self.create_model_params())
                continue
            elif collection_param == 'dataset_parameters':
                gen_params.set_param(collection_param, self.create_dataset_params())
                continue
            try:
                gen_params.set_param(collection_param, self.paramfile_dict[paramfile_param])
            except KeyError as e:
                print("Warning: General parameter {} not found in parameter file".format(e))

        return gen_params


    def create_model_params(self):
        model_params = ModelParameters()
        param_mapping = self.get_collection_to_paramfile_mapping(model_params)
        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'cg_sampling_groups':
                continue
            if collection_param == 'signal_components':
                continue
            try:
                model_params.set_param(collection_param, self.paramfile_dict[paramfile_param])
            except KeyError as e:
                print("Warning: Model parameter {} not found in parameter file".format(e))

        return model_params


    def create_dataset_params(self):
        dataset_params = DatasetParameters()
        param_mapping = self.get_collection_to_paramfile_mapping(dataset_params)
        for collection_param, paramfile_param in param_mapping.items():
            if collection_param == 'include_bands':
                continue
            if collection_param == 'smoothing_scales':
                continue
            try:
                dataset_params.set_param(collection_param, self.paramfile_dict[paramfile_param])
            except KeyError as e:
                print("Warning: Dataset parameter {} not found in parameter file".format(e))

        return dataset_params

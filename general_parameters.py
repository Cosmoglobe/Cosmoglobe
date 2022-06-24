from __future__ import annotations

from enum import Enum, unique
from typing import Union, Callable
from functools import partial

from pydantic import BaseModel

from dataset_parameters import DatasetParameters
from model_parameters import ModelParameters

@unique
class Operation(Enum):
    SAMPLE = 'sample'
    OPTIMIZE = 'optimize'

@unique
class ChainStatus(Enum):
    APPEND = 'append'
    NEW = 'new'

@unique
class MjysrConvention(Enum):
    IRAS = 'IRAS'

class GeneralParameters(BaseModel):
    """
    A container for general parameters needed by Commander.

    By "general" parameters is meant parameters that control the overall
    operation of the run, and output options. In addition the container will
    point to a DatasetParameters instance and a ModelParameters instance, which
    in turn contain all other parameters in them (or in their subclasses).
    """

    # The global CG parameters are not included as they're apparently outdated,
    # according to HKE. We operate only with CG sampling groups
    operation: Operation = None
    chain_status: ChainStatus = None

    verbosity: int = None
    num_gibbs_iter: int = None
    chain_status: str = None
    init_chains: Union[list[str], list[None]] = None

    base_seed: int = None
    num_gibbs_steps_per_tod_sample: int = None
    sample_only_polarization: bool = None
    sample_signal_amplitudes: bool = None
    sample_spectral_indices: bool = None
    sample_powspec: bool = None
    enable_tod_analysis: bool = None
    tod_output_4D_map_every_nth_iter: int = None
    tod_output_auxiliary_map_every_nth_iter: int = None
    tod_include_zodi: bool = None
    tod_num_bp_proposals_per_iter: int = None
    fftw3_magic_numbers: str = None
    enable_tod_simulations: bool = None
    sims_output_directory: str = None
    resample_cmb: bool = None
    first_sample_for_cmb_resamp: int = None
    last_sample_for_cmb_resamp: int = None
    num_subsamp_per_main_sample: int = None
    set_all_noise_maps_to_mean: bool = None
    num_index_cycles_per_iteration: int = None
    ignore_gain_and_bandpass_corr: bool = None
    
    # "Output options"
    thinning_factor: int = None
    nside_chisq: int = None
    polarization_chisq: bool = None
    output_directory: str = None
    output_mixing_matrix: bool = None
    output_residual_maps: bool = None
    output_chisq_map: bool = None
#    output_every_nth_cg_iteration: int # Unclear if needed
#    output_cg_precond_eigenvals: bool # Unclear if needed
    output_input_model: bool = None
    output_debug_seds: bool = None
    output_signals_per_band: bool = None
    mjysr_convention: MjysrConvention = None
    t_cmb: float = None

    dataset_parameters: DatasetParameters = None
    model_parameters: ModelParameters = None

    @classmethod
    def _get_parameter_handling_dict(cls) -> dict[str, Callable]:
        """
        Create a mapping between the container field names and the appropriate
        functions to handle those fields.

        The functions in the output dict will take a parameter file dictionary
        as the only argument, and will return whatever is appropriate for that
        field.

        Output:
            dict[str, Callable]: Mapping between field names and their handlers.
        """

        def default_handler(field_name: str, paramfile_dict: dict[str, str]) -> str:
            paramfile_param = field_name.upper()
            try:
                return paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: General parameter {} not found in parameter file".format(e))
                return None

        def model_param_handler(field_name, paramfile_dict):
            return ModelParameters.create_model_params(paramfile_dict)

        def dataset_param_handler(field_name, paramfile_dict):
            return DatasetParameters.create_dataset_params(paramfile_dict)

        def init_chain_handler(field_name, paramfile_dict):
            num_init_chains = paramfile_dict['NUM_INIT_CHAINS']
            init_chain_list = []
            for i in range(int(num_init_chains)):
                init_chain_list.append(paramfile_dict['INIT_CHAIN{:02d}'.format(i+1)])
            return init_chain_list

        field_names = cls.__fields__.keys()
        handling_dict = {}
        for field_name in field_names:
            if field_name == 'init_chains':
                handling_dict[field_name] = partial(
                    init_chain_handler, field_name)
            elif field_name == 'model_parameters':
                handling_dict[field_name] = partial(
                    model_param_handler, field_name)
            elif field_name == 'dataset_parameters':
                handling_dict[field_name] = partial(
                    dataset_param_handler, field_name)
            else:
                handling_dict[field_name] = partial(
                    default_handler, field_name)
        return handling_dict

    @classmethod
    def create_gen_params(cls,
                          paramfile_dict: dict[str, str]) -> GeneralParameters:
        """
        Factory class method for a GeneralParameters instance.

        Input:
            paramfile_dict[str, str]: A dict (typically created by
                parameter_parser._paramfile_to_dict) mapping the keys found in
                a Commander parameter file to the values found in that same
                file.
        Output:
            GeneralParameters: Parameter container for the general Commander
                parameters, as well as lower-level parameter containers such as
                ModelParameters and DatasetParameters.
        """
        handling_dict = cls._get_parameter_handling_dict()
        param_vals = {}
        for field_name, handling_function in handling_dict.items():
            param_vals[field_name] = handling_function(paramfile_dict)
        return GeneralParameters(**param_vals)

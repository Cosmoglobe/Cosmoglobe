from enum import Enum, unique
from typing import Union

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

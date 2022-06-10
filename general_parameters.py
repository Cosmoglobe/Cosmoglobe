from dataclasses import dataclass
from enum import Enum

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

@dataclass
class GeneralParameters:
    # The global CG parameters are not included as they're apparently outdated,
    # according to HKE. We operate only with CG sampling groups
    operation: Operation
    chain_status: ChainStatus

    verbosity: int
    num_gibbs_iter: int
    chain_status: str
    init_chains: list[str]

    base_seed: int
    num_gibbs_steps_per_tod_sample: int
    sample_only_polarization: bool
    sample_signal_amplitudes: bool
    sample_spectral_indices: bool
    sample_powspec: bool
    enable_tod_analysis: bool
    tod_output_4D_map_every_nth_iter: int
    tod_output_auxiliary_map_every_nth_iter: int
    tod_include_zodi: bool
    tod_num_bp_proposals_per_iter: int
    fftw3_magic_numbers: str
    enable_tod_simulations: bool
    sims_output_directory: str
    resample_cmb: bool
    first_sample_for_cmb_resamp: int
    last_sample_for_cmb_resamp: int
    num_subsamp_per_main_sample: int
    set_all_noise_maps_to_mean: bool
    num_index_cycles_per_iteration: int
    ignore_gain_and_bandpass_corr: bool
    
    # "Output options"
    thinning_factor: int
    nside_chisq: int
    polarization_chisq: bool
    output_directory: str
    output_mixing_matrix: bool
    output_residual_maps: bool
    output_chisq_map: bool
#    output_every_nth_cg_iteration: int # Unclear if needed
#    output_cg_precond_eigenvals: bool # Unclear if needed
    output_input_model: bool
    output_debug_seds: bool
    output_signals_per_band: bool
    mjysr_convention: MjysrConvention
    t_cmb: float

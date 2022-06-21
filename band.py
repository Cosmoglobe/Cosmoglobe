from __future__ import annotations

from enum import Enum, auto, unique
from typing import Union

from parameter_collection import ParameterCollection
from unit import Unit

@unique
class BandpassModel(Enum):
    POWLAW_TILT = auto()
    ADDITIVE_SHIFT = auto()

@unique
class BandpassType(Enum):
    LFI = auto()
    HFI_CMB = auto()
    QUIET = auto()
    WMAP = auto()
    DELTA = auto()

@unique
class BeamType(Enum):
    B_L = auto()
    FEBECOP = auto()

@unique
class NoiseFormat(Enum):
    RMS = auto()
    QUCOV = auto()

class Band(ParameterCollection):
    bandpass_model: BandpassModel
    bandpass_type: BandpassType
    bandpass_file: str
    beam_type: BeamType
    beam_b_l_file: str
    beam_b_ptsrc_file: str
    default_bp_delta: float
    default_gain: float
    default_noiseamp: float
    gain_apod_fwhm: float
    gain_calib_comp: str
    gain_lmax: int
    gain_lmin: int
    label: str
    lmax: int
    mapfile: str
    maskfile: str
    maskfile_calib: str
    noise_format: NoiseFormat
    noise_uniformize_fsky: float
    noisefile: str
    reg_noisefile: str
    noise_rms_smooth: list[str]
    nside: int
    nominal_freq: float
    obs_period: int
    pixel_window: str
    polarization: bool
    samp_bandpass: bool
    samp_gain: bool
    samp_noise_amp: bool
    unit: Unit
    tod_bp_init_prop: str
    tod_detector_list: list[str]
    tod_filelist: str
    tod_init_from_hdf: Union[str, bool]
    tod_main_procmask: str
    tod_small_procmask: str
    tod_start_scanid: int 
    tod_end_scanid: int
    tod_tot_numscan: int
    tod_flag: int
    tod_rimo: str
    include_band: bool
    component_sensitivity: str = "broadband" # Only option found in the files
    gain_apod_mask: str = "fullsky" # Only option found in the files

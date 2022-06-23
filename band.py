from __future__ import annotations

from enum import Enum, auto, unique
from typing import Union
from pydantic import BaseModel

from unit import Unit

@unique
class BandpassModel(Enum):
    POWLAW_TILT = 'powlaw_tilt'
    ADDITIVE_SHIFT = 'additive_shift' 

@unique
class BandpassType(Enum):
    LFI = 'LFI'
    HFI_CMB = 'HFI_CMB'
    HFI = 'HFI'
    QUIET = 'QUIET'
    WMAP = 'WMAP'
    DELTA = 'DELTA'

@unique
class BeamType(Enum):
    B_L = 'b_l' 
    FEBECOP = 'febecop'

@unique
class NoiseFormat(Enum):
    RMS = 'rms'
    QUCOV = 'qucov'

class Band(BaseModel):
    """
    A container for the parameters that define a given band used in Commander.
    Typically these are the ones called BAND_***_&&& where &&& is replaced by
    the band number in question.
    """
    bandpass_model: BandpassModel
    bandpass_type: BandpassType
    bandpassfile: str
    beamtype: BeamType
    beam_b_l_file: str
    beam_b_ptsrc_file: Union[str, None]
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
    reg_noisefile: Union[str, None]
    noise_rms_smooth: list[str] = None
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

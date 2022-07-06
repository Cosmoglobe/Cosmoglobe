from __future__ import annotations

from enum import Enum, auto, unique
from functools import partial
from typing import Union, Callable
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

    @classmethod
    def _get_parameter_handling_dict(cls) -> dict[str, Callable]:
        """
        Create a mapping between the container field names and the appropriate
        functions to handle those fields.

        The functions in the output dict will take a parameter file dictionary
        and the band number as arguments, and will return whatever is
        appropriate for that field.

        Output:
            dict[str, Callable]: Mapping between field names and their handlers.
        """

        def default_handler(field_name, paramfile_dict, band_num):
            paramfile_param = field_name.upper() + '{:03d}'.format(band_num)
            if field_name != 'include_band':
                paramfile_param = 'BAND_' + paramfile_param
            try:
                return paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: Band parameter {} not found in parameter file".format(e))
                return None

        def noise_rms_smooth_handler(field_name, paramfile_dict, band_num):
            i = 1
            noise_rms_smooth = []
            while True:
                try:
                    noise_rms_smooth.append(
                        paramfile_dict[
                            'BAND_NOISE_RMS{:03d}_SMOOTH{:02d}'.format(band_num, i)])
                    i+= 1
                except KeyError:
                    break
            return noise_rms_smooth

        def tod_detector_list_handler(field_name, paramfile_dict, band_num):
            paramfile_param = 'BAND_' + field_name.upper() + '{:03d}'.format(band_num)
            return paramfile_dict[paramfile_param].split(',')

        field_names = cls.__fields__.keys()
        handling_dict = {}
        for field_name in field_names:
            if field_name == 'noise_rms_smooth':
                handling_dict[field_name] = partial(
                    noise_rms_smooth_handler, field_name)
            elif field_name == 'tod_detector_list':
                handling_dict[field_name] = partial(
                    tod_detector_list_handler, field_name)
            else:
                handling_dict[field_name] = partial(
                    default_handler, field_name)
        return handling_dict

    @classmethod
    def create_band_params(cls,
                           paramfile_dict: dict[str, Any],
                           band_num: int) -> Band:
        """
        Factory class method for a Band instance.

        Input:
            paramfile_dict[str, str]: A dict (typically created by
                parameter_parser._paramfile_to_dict) mapping the keys found in
                a Commander parameter file to the values found in that same
                file.
            band_num (int): The number of the band to be instantiated.
        Output:
            Band: Parameter container for a band-specific set of
                Commander parameters.
        """

        handling_dict = cls._get_parameter_handling_dict()
        param_vals = {}
        for field_name, handling_function in handling_dict.items():
            param_vals[field_name] = handling_function(paramfile_dict, band_num)
        return Band(**param_vals)


    def serialize_to_paramfile_dict(self, band_num: int) -> dict[str, Any]:
        """
        Creates a mapping from Commander parameter names to the values in the
        Band instance, with all lower-level parameter collections
        similarly serialized.

        Note the values in this mapping are basic types, not strings. This
        means they will have to be processed further before they are ready for
        a Commander parameter file. The keys, however, need no more processing.

        Input:
            band_num[int]: The number of the band instance in the Commander
            file context.

        Output:
            dict[str, Any]: Mapping from Commander parameter file names to the
                parameter values.
        """
        paramfile_dict = {}
        for field_name, value in self.__dict__.items():
            if field_name == 'noise_rms_smooth':
                for i, band_noise_rms_smooth in enumerate(value):
                    paramfile_dict[
                        'BAND_NOISE_RMS{:03d}_SMOOTH{:02d}'.format(
                            band_num, i+1)] = band_noise_rms_smooth
            elif field_name == 'tod_detector_list':
                paramfile_dict[
                    'BAND_{}{:03d}'.format(field_name.upper(), band_num)] = (
                        ','.join(value)
                    )
            else:
                paramfile_dict[
                    'BAND_{}'.format(field_name.upper())] = value
        return paramfile_dict

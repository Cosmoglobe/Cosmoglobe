from __future__ import annotations

from enum import Enum, auto, unique
from pydantic import BaseModel

from parameter_collection import ParameterCollection
from unit import Unit

@unique
class ComponentLabel(Enum):
    # This class might be redundant, likely the label can be anything you want.
    SYNCH = 'synch'
    DUST = 'dust'
    FF = 'ff'
    CMB = 'cmb'
    AME = 'ame'
    MD = 'md'
    RADIO = 'radio'

@unique
class ComponentType(Enum):
    POWER_LAW = 'power_law'
    MBB = 'MBB'
    FREEFREE = 'freefree'
    CMB = 'cmb'
    SPINDUST2 = 'spindust2'
    MD = 'md'
    RADIO = 'radio'

@unique
class ComponentClass(Enum):
    DIFFUSE = 'diffuse'
    PTSRC = 'ptsrc'
    TEMPLATE = 'template'

@unique
class MonopolePriorType(Enum):
    BANDMONO = 'bandmono'
    CROSSCORR = 'crosscorr'
    MONOPOLE_MINUS_DIPOLE = 'monopole-dipole'

@unique
class ClType(Enum):
    GAUSS = 'gauss'
    NONE = 'none'
    SINGLE_L = 'single_l'
    BINNED = 'binned'
    POWER_LAW = 'power_law'
    EXP = 'exp'
    POWER_LAW_GAUSS = 'power_law_gauss'

@unique
class PolType(Enum):
    TEB = 1 # All three joint
    T_EB = 2 # EB joint
    T_E_B = 3 # None joint


class UniformPrior(BaseModel):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high


class GaussPrior(BaseModel):
    def __init__(self, mean: float, rms: float):
        self.mean = mean
        self.rms = rms


class MonopolePrior(BaseModel):
    def __init__(self, m_type: MonopolePriorType, **kwargs):
        self.type = m_type

        if self.type == MonopolePriorType.BANDMONO:
            self.label = kwargs['label'] 
        elif self.type == MonopolePriorType.CROSSCORR:
            self.corrmap = kwargs['corrmap']
            self.nside = kwargs['nside']
            self.fwhm = kwargs['fwhm']
            self.thresholds = kwargs['thresholds']
        elif self.type == MonopolePriorType.MONOPOLE_MINUS_DIPOLE:
            self.mask = kwargs['mask']

class Component(ParameterCollection):
    # Note: The comp_CG parameters are, according to HKE, outdated, and because
    # it messes with a potential future CG class if we include them, I will
    # leave them out.  Components checked so far: Synch, dust, freefree, cmb, ame, monodipole, radio

    # All 'pixreg' parameters I have encountered are not in the code, so I don't include them here.

    label: ComponentLabel
    ctype: ComponentType
    cclass: ComponentClass
    polarization: bool
    init_from_hdf: str

    nside: int # Only in synch, dust, freefree, cmb, ame, radio
    nu_ref_t: float # Only in synch, dust, freefree, cmb, ame, radio
    nu_ref_p: float # Only in synch, dust, freefree, cmb, ame, radio
    cl_poltype: PolType # Only in synch, dust, freefree, cmb, ame, radio

    monopole_prior: MonopolePrior # Only in synch, dust, freefree, cmb, ame
    deflation_mask: str # Only in synch, dust, freefree, cmb, ame
    l_apod: int # Only in synch, dust, freefree, cmb, ame
    lmin_amp: int # Only in synch, dust, freefree, cmb, ame
    lmax_amp: int # Only in synch, dust, freefree, cmb, ame
    lmax_ind: int # Only in synch, dust, freefree, cmb, ame
    output_fwhm: float # Only in synch, dust, freefree, cmb, ame
    unit: Unit # Only in synch, dust, freefree, cmb, ame
    mask: str # Only in synch, dust, freefree, cmb, ame
    cl_type: ClType # Only in synch, dust, freefree, cmb, ame
    cl_beta_prior_mean: float # Only in synch, dust, freefree, cmb, ame
    cl_beta_prior_rms: float # Only in synch, dust, freefree, cmb, ame
    cl_l_pivot: int # Only in synch, dust, freefree, cmb, ame
    cl_default_amp_t: float # Only in synch, dust, freefree, cmb, ame
    cl_default_amp_e: float # Only in synch, dust, freefree, cmb, ame
    cl_default_amp_b: float # Only in synch, dust, freefree, cmb, ame
    cl_default_beta_t: float # Only in synch, dust, freefree, cmb, ame
    cl_default_beta_e: float # Only in synch, dust, freefree, cmb, ame
    cl_default_beta_b: float # Only in synch, dust, freefree, cmb, ame
    input_amp_map: str # Only in synch, dust, freefree, cmb, ame
    prior_amp_map: str # Only in synch, dust, freefree, cmb, ame
    output_eb_map: bool # Only in synch, dust, freefree, cmb, ame

    apply_jeffreys_prior: bool # Only synch, dust, freefree, ame, radio

    indmask: str # Only synch, dust, freefree, ame

    prior_amp_lmax: float # Only freefree, ame

    alpha_nu_min: float # Only ame, radio
    alpha_nu_max: float # Only ame, radio
    default_alpha: float # Only ame, radio
    prior_uni_alpha: UniformPrior # Only ame, radio
    prior_gauss_alpha: GaussPrior # Only ame, radio

    beta_nu_min: float # Only synch, dust, radio
    beta_nu_max: float # Only synch, dust, radio
    default_beta: float # Only synch, dust, radio
    prior_uni_beta: UniformPrior # Only synch, dust, radio
    prior_gauss_beta: GaussPrior # Only synch, dust, radio

    md_mono_from_prior: str # Only monodipole
    md_definition_file: str # Only monodipole

    nu_ref: float # Only radio
    catalog: str # Only radio
    init_catalog: str # Only radio
    ptsrc_template: str # Only radio
    output_ptsrc_template: bool # Only radio
    amp_rms_scale_factor: float # Only radio
    min_dist_between_src: float # Only radio
    poltype: PolType # Only radio
    apply_positivity_prior: bool # Only radio
    burn_in_on_first_sample: bool # Only radio

    nu_p_poltype: PolType # Only ame
    input_nu_p_map: str # Only ame
    nu_p_smoothing_scale: float # Only ame
    default_nu_p: float # Only ame
    prior_uni_nu_p: UniformPrior # Only ame
    prior_gauss_nu_p: GaussPrior # Only ame
    nu_p_nu_min: float # Only ame
    nu_p_nu_max: float # Only ame
    nu_p_almsamp_init: str # Only ame
    alpha_poltype: PolType # Only ame
    input_alpha_map: str # Only ame
    alpha_smoothing_scale: float # Only ame
    alpha_almsamp_init: str # Only ame
    sed_template: str # Only ame
    
    cl_default_theta_t: float # Only freefree
    cl_default_theta_e: float # Only freefree
    cl_default_theta_b: float # Only freefree
    em_poltype: PolType # Only freefree
    input_em_map: str # Only freefree
    em_smoothing_scale: float # Only freefree
    default_em: float # Only freefree
    prior_uni_em: UniformPrior # Only freefree
    prior_gauss_em: GaussPrior # Only freefree
    em_nu_min: float # Only freefree
    em_nu_max: float # Only freefree
    t_e_poltype: PolType # Only freefree
    input_t_e_map: str # Only freefree
    t_e_smoothing_scale: float # Only freefree
    default_t_e: float # Only freefree
    prior_uni_t_e: UniformPrior # Only freefree
    prior_gauss_t_e: GaussPrior # Only freefree
    t_e_nu_min: float # Only freefree
    t_e_nu_max: float # Only freefree
    t_e_almsamp_init: float # Only freefree

    input_beta_map: str # Only synch, dust
    beta_smoothing_scale: float # Only synch, dust
    beta_poltype: PolType # Only synch, dust

    input_t_map: str # Only dust
    default_t: float # Only dust
    prior_uni_t: UniformPrior # Only dust
    prior_gauss_t: GaussPrior # Only dust
    t_smoothing_scale: float # Only dust
    t_poltype: PolType # Only dust
    t_nu_max: float # Only dust
    t_nu_min: float # Only dust

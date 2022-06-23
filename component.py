from __future__ import annotations

from enum import Enum, auto, unique
from pydantic import BaseModel
from typing import Union

from unit import Unit

@unique
class ComponentType(Enum):
    POWER_LAW = 'power_law'
    MBB = 'MBB'
    FREEFREE = 'freefree'
    CMB = 'cmb'
    SPINDUST2 = 'spindust2'
    MD = 'md'
    RADIO = 'radio'
    CMB_RELQUAD = 'cmb_relquad'

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
    TEB = '1' # All three joint
    T_EB = '2' # EB joint
    T_E_B = '3' # None joint


class UniformPrior(BaseModel):
    low: float
    high: float


class GaussPrior(BaseModel):
    mean: float
    rms: float


class MonopolePrior(BaseModel):
    type: MonopolePriorType
    label: str = None # Needed for MonopolePriorType.BANDMONO
    corrmap: str = None # Needed for MonopolePriorType.CROSSCORR
    nside: int = None # Needed for MonopolePriorType.CROSSCORR
    fwhm: float = None # Needed for MonopolePriorType.CROSSCORR
    thresholds: list[float] = None # Needed for MonopolePriorType.CROSSCORR
    mask: str = None # Needed for MonopolePriorType.MONOPOLE_MINUS_DIPOLE

class Component(BaseModel):
    """
    A container for the parameters that define a given component used in Commander.
    Typically these are the ones called COMP_***_&&& where &&& is replaced by
    the band number in question.
    """
    # Note: The comp_CG parameters are, according to HKE, outdated, and because
    # it messes with a potential future CG class if we include them, I will
    # leave them out.  Components checked so far: Synch, dust, freefree, cmb, ame, monodipole, radio

    # All 'pixreg' parameters I have encountered are not in the code, so I don't include them here.

    label: str
    ctype: ComponentType
    cclass: ComponentClass
    polarization: bool
    init_from_hdf: Union[str, None]

    nside: int = None # Only in synch, dust, freefree, cmb, ame, radio
    nu_ref_t: float = None # Only in synch, dust, freefree, cmb, ame, radio
    nu_ref_p: float = None # Only in synch, dust, freefree, cmb, ame, radio
    cl_poltype: PolType = None # Only in synch, dust, freefree, cmb, ame, radio

    monopole_prior: MonopolePrior = None # Only in synch, dust, freefree, cmb, ame
    deflation_mask: str = None # Only in synch, dust, freefree, cmb, ame
    l_apod: int = None # Only in synch, dust, freefree, cmb, ame
    lmin_amp: int = None # Only in synch, dust, freefree, cmb, ame
    lmax_amp: int = None # Only in synch, dust, freefree, cmb, ame
    lmax_ind: int = None # Only in synch, dust, freefree, cmb, ame
    output_fwhm: float = None # Only in synch, dust, freefree, cmb, ame
    unit: Unit = None # Only in synch, dust, freefree, cmb, ame
    mask: str = None # Only in synch, dust, freefree, cmb, ame
    cl_type: ClType = None # Only in synch, dust, freefree, cmb, ame
    cl_beta_prior_mean: float = None # Only in synch, dust, freefree, cmb, ame
    cl_beta_prior_rms: float = None # Only in synch, dust, freefree, cmb, ame
    cl_l_pivot: int = None # Only in synch, dust, freefree, cmb, ame
    cl_default_amp_t: float = None # Only in synch, dust, freefree, cmb, ame
    cl_default_amp_e: float = None # Only in synch, dust, freefree, cmb, ame
    cl_default_amp_b: float = None # Only in synch, dust, freefree, cmb, ame
    cl_default_beta_t: float = None # Only in synch, dust, freefree, cmb, ame
    cl_default_beta_e: float = None # Only in synch, dust, freefree, cmb, ame
    cl_default_beta_b: float = None # Only in synch, dust, freefree, cmb, ame
    input_amp_map: str = None # Only in synch, dust, freefree, cmb, ame
    prior_amp_map: str = None # Only in synch, dust, freefree, cmb, ame
    output_eb_map: bool = None # Only in synch, dust, freefree, cmb, ame

    apply_jeffreys_prior: bool = None # Only synch, dust, freefree, ame, radio

    indmask: str = None # Only synch, dust, freefree, ame

    prior_amp_lmax: float = None # Only freefree, ame

    alpha_nu_min: float = None # Only ame, radio
    alpha_nu_max: float = None # Only ame, radio
    default_alpha: float = None # Only ame, radio
    prior_uni_alpha: UniformPrior = None # Only ame, radio
    prior_gauss_alpha: GaussPrior = None # Only ame, radio

    beta_nu_min: float = None # Only synch, dust, radio
    beta_nu_max: float = None # Only synch, dust, radio
    default_beta: float = None # Only synch, dust, radio
    prior_uni_beta: UniformPrior = None # Only synch, dust, radio
    prior_gauss_beta: GaussPrior = None # Only synch, dust, radio

    md_mono_from_prior: str = None # Only monodipole
    md_definition_file: str = None # Only monodipole

    nu_ref: float = None # Only radio
    catalog: str = None # Only radio
    init_catalog: str = None # Only radio
    ptsrc_template: str = None # Only radio
    output_ptsrc_template: bool = None # Only radio
    amp_rms_scale_factor: float = None # Only radio
    min_dist_between_src: float = None # Only radio
    poltype: PolType = None # Only radio
    apply_positivity_prior: bool = None # Only radio
    burn_in_on_first_sample: bool = None # Only radio

    nu_p_poltype: PolType = None # Only ame
    input_nu_p_map: str = None # Only ame
    nu_p_smoothing_scale: float = None # Only ame
    default_nu_p: float = None # Only ame
    prior_uni_nu_p: UniformPrior = None # Only ame
    prior_gauss_nu_p: GaussPrior = None # Only ame
    nu_p_nu_min: float = None # Only ame
    nu_p_nu_max: float = None # Only ame
    nu_p_almsamp_init: str = None # Only ame
    alpha_poltype: PolType = None # Only ame
    input_alpha_map: str = None # Only ame
    alpha_smoothing_scale: float = None # Only ame
    alpha_almsamp_init: str = None # Only ame
    sed_template: str = None # Only ame
    
    cl_default_theta_t: float = None # Only freefree
    cl_default_theta_e: float = None # Only freefree
    cl_default_theta_b: float = None # Only freefree
    em_poltype: PolType = None # Only freefree
    input_em_map: str = None # Only freefree
    em_smoothing_scale: float = None # Only freefree
    default_em: float = None # Only freefree
    prior_uni_em: UniformPrior = None # Only freefree
    prior_gauss_em: GaussPrior = None # Only freefree
    em_nu_min: float = None # Only freefree
    em_nu_max: float = None # Only freefree
    t_e_poltype: PolType = None # Only freefree
    input_t_e_map: str = None # Only freefree
    t_e_smoothing_scale: float = None # Only freefree
    default_t_e: float = None # Only freefree
    prior_uni_t_e: UniformPrior = None # Only freefree
    prior_gauss_t_e: GaussPrior = None # Only freefree
    t_e_nu_min: float = None # Only freefree
    t_e_nu_max: float = None # Only freefree
    t_e_almsamp_init: float = None # Only freefree

    input_beta_map: str = None # Only synch, dust
    beta_smoothing_scale: float = None # Only synch, dust
    beta_poltype: PolType = None # Only synch, dust

    input_t_map: str = None # Only dust
    default_t: float = None # Only dust
    prior_uni_t: UniformPrior = None # Only dust
    prior_gauss_t: GaussPrior = None # Only dust
    t_smoothing_scale: float = None # Only dust
    t_poltype: PolType = None # Only dust
    t_nu_max: float = None # Only dust
    t_nu_min: float = None # Only dust
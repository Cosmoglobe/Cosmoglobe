from pydantic import BaseModel

from enum import Enum, auto
from cg_sampling_group import CGSamplingGroup
from component import Component

class ModelParameters(BaseModel):
    """
    A container for the general Commander parameters that define the models used in
    Commander. Also contains a list of Component containers, which contain
    model-specific parameters, and a list of CGSamplingGroup containers, each
    of which defines a CG sampling group.
    """
    instrument_param_file: str = None
    init_instrument_from_hdf: str = None
    cg_sampling_groups: list[CGSamplingGroup] = None
    signal_components: list[Component] = None

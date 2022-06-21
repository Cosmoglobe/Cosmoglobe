from pydantic import BaseModel

from enum import Enum, auto
from cg_sampling_group import CGSamplingGroup
from component import Component

class ModelParameters(BaseModel):
    instrument_param_file: str = None
    init_instrument_from_hdf: str = None
    cg_sampling_groups: list[CGSamplingGroup] = None
    signal_components: list[Component] = None

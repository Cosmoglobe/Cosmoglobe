from dataclasses import dataclass
from enum import Enum, auto
from cg_sampling_group import CGSamplingGroup
from component import Component

@dataclass
class GlobalModelParameters:
    instrument_param_file: str
    init_instrument_from_hdf: str
    cg_sampling_groups: list[CGSamplingGroup]
    signal_components: list[Component]

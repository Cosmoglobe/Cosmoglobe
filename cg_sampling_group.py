from dataclasses import dataclass
from enum import Enum, auto
from component import Component

@dataclass
class CGSamplingGroup:
    mask: str
    maxiter: int
    components: list[Component]

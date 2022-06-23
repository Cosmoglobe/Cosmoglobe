from enum import Enum, auto
from pydantic import BaseModel

from component import Component

class CGSamplingGroup(BaseModel):
    mask: str
    maxiter: int
    components: list[Component]

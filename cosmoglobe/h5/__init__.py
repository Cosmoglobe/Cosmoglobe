from enum import Enum, auto
from cosmoglobe.h5._exceptions import (
    ChainKeyError,
    ChainSampleError,
    ChainComponentNotFoundError, 
    ChainFormatError
)

PARAMETER_GROUP_NAME = "parameters"


class ChainVersion(Enum):
    """The version number of the chain."""

    OLD = auto()
    NEW = auto()

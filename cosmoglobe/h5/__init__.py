from enum import Enum, auto


PARAMETER_GROUP_NAME = "parameters"


class ChainVersion(Enum):
    """The version number of the chain."""

    OLD = auto()
    NEW = auto()

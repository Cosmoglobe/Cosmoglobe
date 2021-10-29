from enum import Enum, auto

class SkyComponentType(Enum):
    """Enums representing a type of sky component."""

    DIFFUSE = auto()
    POINTSOURCE = auto()
    LINE = auto()


class SkyComponentLabel(Enum):
    """Enums representing the label of a sky component."""

    AME = "ame"
    CMB = "cmb"
    DUST = "dust"
    FF = "ff"
    RADIO = "radio"
    SYNCH = "synch"
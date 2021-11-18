from enum import Enum


class SkyComponentLabel(Enum):
    """Enums representing labeled emission sources.

    A `SkyComponent` implementation should represent the emission from one of
    the following classes.
    """

    AME = "ame"
    CIB = "cib"
    CMB = "cmb"
    CO = "co"
    DUST = "dust"
    FF = "ff"
    PAH = "pah"
    RADIO = "radio"
    SYNCH = "synch"
    SZ = "sz"
    ZODI = "zodi"

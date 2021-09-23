from astropy.units import Unit

from cosmoglobe.sky.components import AME, CMB, Dust, FreeFree, Radio, Synchrotron


NO_SMOOTHING = 0.0 * Unit("arcmin")
DEFAULT_OUTPUT_UNIT = Unit("uK")
DEFAULT_FREQ_UNIT = Unit("GHz")

COSMOGLOBE_COMPS = {
    comp.label: comp  # type: ignore
    for comp in [
        AME,
        CMB,
        Dust,
        FreeFree,
        Radio,
        Synchrotron,
    ]
}

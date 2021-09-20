from cosmoglobe.sky.components import AME, CMB, Dust, FreeFree, Radio, Synchrotron

COSMOGLOBE_COMPS = {
    comp.label: comp
    for comp in [
        AME,
        CMB,
        Dust,
        FreeFree,
        Radio,
        Synchrotron,
    ]
}

from enum import Enum
from typing import Dict, Type

from cosmoglobe.sky.base_components import SkyComponent

from cosmoglobe.sky.components.ame import AME
from cosmoglobe.sky.components.synchrotron import Synchrotron
from cosmoglobe.sky.components.dust import Dust
from cosmoglobe.sky.components.freefree import FreeFree
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.radio import Radio


class SkyComponentLabel(Enum):
    """Enums representing available unique labels for sky components."""

    AME = "ame"
    CMB = "cmb"
    DUST = "dust"
    FF = "ff"
    RADIO = "radio"
    SYNCH = "synch"


class CosmoglobeSkyModel:
    """The sky components making up Cosmoglobe Sky Model."""

    components: Dict[str, Type[SkyComponent]] = {}

    def register_component(
        self, label: SkyComponentLabel, component: Type[SkyComponent]
    ) -> None:
        """Registers a sky component."""

        if label.value in self.components:
            raise ValueError("component {label} is already registered to the model.")

        self.components[label.value] = component

    def __repr__(self) -> str:
        return str(list(self.components.keys()))


cosmoglobe_sky_model = CosmoglobeSkyModel()
cosmoglobe_sky_model.register_component(SkyComponentLabel.AME, AME)
cosmoglobe_sky_model.register_component(SkyComponentLabel.CMB, CMB)
cosmoglobe_sky_model.register_component(SkyComponentLabel.DUST, Dust)
cosmoglobe_sky_model.register_component(SkyComponentLabel.FF, FreeFree)
cosmoglobe_sky_model.register_component(SkyComponentLabel.RADIO, Radio)
cosmoglobe_sky_model.register_component(SkyComponentLabel.SYNCH, Synchrotron)

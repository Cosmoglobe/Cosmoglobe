from typing import Dict, Type

from cosmoglobe.sky.base_components import SkyComponent
from cosmoglobe.sky.components.ame import AME
from cosmoglobe.sky.components.synchrotron import Synchrotron
from cosmoglobe.sky.components.dust import Dust
from cosmoglobe.sky.components.freefree import FreeFree
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.radio import Radio


class CosmoglobeComponents:
    """Factory keeping a registry of components in the Cosmoglobe Sky Model."""

    def __init__(self) -> None:
        """Initializes the component registry."""

        self.components: Dict[str, Type[SkyComponent]] = {}

    def register(self, component: Type[SkyComponent]) -> None:
        """Registers a sky component."""

        self.components[component.label] = component  # type: ignore


factory = CosmoglobeComponents()

factory.register(AME)
factory.register(Synchrotron)
factory.register(Dust)
factory.register(FreeFree)
factory.register(CMB)
factory.register(Radio)

COSMOGLOBE_COMPS = factory.components

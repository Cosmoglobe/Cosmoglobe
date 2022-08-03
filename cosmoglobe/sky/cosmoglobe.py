from dataclasses import dataclass, field
from typing import Dict, List, Type

from cosmoglobe.sky._base_components import SkyComponent
from cosmoglobe.sky._exceptions import (
    ComponentNotFoundError,
    CosmoglobeModelError,
    ModelNotFoundError,
)
from cosmoglobe.sky.components.ame import SpinningDust
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.dust import ModifiedBlackbody
from cosmoglobe.sky.components.freefree import LinearOpticallyThin
from cosmoglobe.sky.components.radio import AGNPowerLaw
from cosmoglobe.sky.components.synchrotron import PowerLaw


@dataclass
class CosmoglobeModel:
    """The sky components making up Cosmoglobe Sky Model."""

    version: str
    components: List[Type[SkyComponent]]

    def __post_init__(self) -> None:
        """Makes sure that no duplicates of a component label exists."""

        if any(self.components.count(comp) > 1 for comp in self.components):
            raise CosmoglobeModelError(
                "components list to CosmoglobeModel cant contain multiple "
                "representations of the same labeled component."
            )

    def __getitem__(self, component_name: str) -> Type[SkyComponent]:
        """Returns a sky component from the cosmoglobe model."""

        for component in self.components:
            if component.label.value == component_name:
                return component
        raise ComponentNotFoundError(f"component {component_name} not found in model.")


@dataclass
class CosmoglobeModelRegistry:
    """Container for registered sky model versions."""

    REGISTRY: Dict[str, CosmoglobeModel] = field(default_factory=dict)

    def register_model(self, model: CosmoglobeModel) -> None:
        """Adds a new sky model to the registry."""

        if (version := model.version) in self.REGISTRY:
            raise ValueError(f"model by version {version} is already registered.")

        self.REGISTRY[version] = model

    def get_model(self, version: str) -> CosmoglobeModel:
        """Returns a registered sky model."""
        try:
            return self.REGISTRY[version]
        except KeyError:
            raise ModelNotFoundError(
                f"model {version} was not found in the registry. "
                f"Available models are: {list(self.REGISTRY.keys())}"
            )


cosmoglobe_registry = CosmoglobeModelRegistry()
cosmoglobe_registry.register_model(
    CosmoglobeModel(
        version="BeyondPlanck",
        components=[
            SpinningDust,
            CMB,
            ModifiedBlackbody,
            LinearOpticallyThin,
            AGNPowerLaw,
            PowerLaw,
        ],
    )
)

DEFAULT_COSMOGLOBE_MODEL = cosmoglobe_registry.get_model("BeyondPlanck")


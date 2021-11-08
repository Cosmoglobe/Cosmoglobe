from typing import Dict, List, Optional, Type
from dataclasses import dataclass

from cosmoglobe.sky._exceptions import ModelNotFoundError, ComponentNotFoundError

from cosmoglobe.sky.base_components import SkyComponent
from cosmoglobe.sky.components.ame import AME
from cosmoglobe.sky.components.synchrotron import Synchrotron
from cosmoglobe.sky.components.dust import ThermalDust
from cosmoglobe.sky.components.freefree import FreeFree
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.radio import Radio


@dataclass
class CosmoglobeModelInfo:
    """Information object for the Cosmoglobe Sky Model.

    The Cosmoglobe Sky Model will dynamically evolve with the inclusion
    of new datasets.

    TODO: Expand the information contained.
    """

    version: str
    data_url: str


@dataclass
class CosmoglobeModel:
    """The sky components making up Cosmoglobe Sky Model."""

    components: List[Type[SkyComponent]]
    info: CosmoglobeModelInfo

    def __getitem__(self, key: str) -> Type[SkyComponent]:
        """Returns a SkyComponent class."""
        for component in self.components:
            if component.label.value == key:
                return component
        else:
            raise ComponentNotFoundError(f"component {key} not found in sky model.")

    def __str__(self) -> str:
        repr = "Cosmoglobe Sky Model\n"
        repr += f"--------------------\n"
        repr += f"version: {self.info.version}\n"
        repr += (
            f"components: {', '.join([comp.label.value for comp in self.components])}\n"
        )

        return repr


class CosmoglobeRegistry:
    """Container for registered sky models."""

    def __init__(self) -> None:
        self.models: Dict[str, CosmoglobeModel] = {}
        self.default_model: Optional[CosmoglobeModel] = None

    def register_model(self, name: str, model: CosmoglobeModel) -> None:
        """Adds a new sky model to the registry."""

        if name in self.models:
            raise ValueError(f"model by name {name} is already registered.")

        self.models[name] = model

    def get_model(self, name: str) -> CosmoglobeModel:
        """Returns a registered sky model."""
        try:
            return self.models[name]
        except KeyError:
            raise ModelNotFoundError(
                f"model {name} was not found in the registry. "
                f"Available models are: {list(self.models.keys())}"
            )

    def set_default_model(self, name: str) -> None:
        """Sets the default sky model."""

        self.default_model = self.models[name]


cosmoglobe_registry = CosmoglobeRegistry()
cosmoglobe_registry.register_model(
    name="BeyondPlanck",
    model=CosmoglobeModel(
        components=[AME, CMB, ThermalDust, FreeFree, Radio, Synchrotron],
        info=CosmoglobeModelInfo(
            version="BeyondPlanck",
            data_url="http://cosmoglobe.uio.no/BeyondPlanck/",
        ),
    ),
)

cosmoglobe_registry.set_default_model("BeyondPlanck")
DEFAULT_COSMOGLOBE_MODEL = cosmoglobe_registry.default_model

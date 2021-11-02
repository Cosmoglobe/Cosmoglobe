from typing import Dict, List, Type
from dataclasses import dataclass

from cosmoglobe.sky._exceptions import ModelNotFoundError

from cosmoglobe.sky.base_components import SkyComponent
from cosmoglobe.sky.components.ame import AME
from cosmoglobe.sky.components.synchrotron import Synchrotron
from cosmoglobe.sky.components.dust import ThermalDust
from cosmoglobe.sky.components.freefree import FreeFree
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.radio import Radio


@dataclass
class SkyModelInfo:
    """Information object for the Cosmoglobe Sky Model.

    The Cosmoglobe Sky Model will dynamically evolve with the inclusion
    of new datasets.
    """

    version: str
    data_url: str


@dataclass
class CosmoglobeSkyModel:
    """The sky components making up Cosmoglobe Sky Model."""

    components: List[Type[SkyComponent]]
    info: SkyModelInfo

    @property
    def labels(self) -> List[str]:
        return [comp.label.value for comp in self.components]

    def __str__(self) -> str:
        repr = "========Cosmoglobe Sky Model========\n"
        repr += f"version: {self.info.version}\n"
        repr += f"components: {', '.join(self.labels)}"

        return repr


class SkyModelRegistry:
    """Container for registered sky models."""

    def __init__(self) -> None:
        self.models: Dict[str, CosmoglobeSkyModel] = {}

    def register_model(self, name: str, model: CosmoglobeSkyModel) -> None:
        """Adds a new sky model to the registry."""

        if name in self.models:
            raise ValueError(f"model by name {name} is already registered.")

        self.models[name] = model

    def get_model(self, name: str) -> CosmoglobeSkyModel:
        """Returns a registered sky model."""
        try:
            return self.models[name]
        except KeyError:
            raise ModelNotFoundError(
                f"model {name} was not found in the registry. "
                f"Available models are: {list(self.models.keys())}"
            )


skymodel_registry = SkyModelRegistry()
skymodel_registry.register_model(
    name="BeyondPlanck",
    model=CosmoglobeSkyModel(
        components=[AME, CMB, ThermalDust, FreeFree, Radio, Synchrotron],
        info=SkyModelInfo(
            version="BeyondPlanck",
            data_url="http://cosmoglobe.uio.no/data_releases/BeyondPlanck/BP10/",
        ),
    ),
)

DEFAULT_SKY_MODEL = skymodel_registry.get_model("BeyondPlanck")

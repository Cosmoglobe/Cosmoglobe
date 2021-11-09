from typing import Dict, List, Optional, Type
from dataclasses import dataclass

from cosmoglobe.sky._base_components import SkyComponent
from cosmoglobe.sky._exceptions import ModelNotFoundError, ComponentNotFoundError
from cosmoglobe.sky.components.ame import AME
from cosmoglobe.sky.components.synchrotron import Synchrotron
from cosmoglobe.sky.components.dust import ThermalDust
from cosmoglobe.sky.components.freefree import FreeFree
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.radio import Radio


@dataclass
class CosmoglobeModel:
    """The sky components making up Cosmoglobe Sky Model."""

    version: str
    components: List[Type[SkyComponent]]

    def __getitem__(self, component_name: str) -> Type[SkyComponent]:
        """Returns a sky component from the cosmoglobe model."""

        for component in self.components:
            if component.label.value == component_name:
                return component
        raise ComponentNotFoundError(
            f"component {component_name} not found in model."
        )


class CosmoglobeRegistry:
    """Container for registered sky model versions."""

    def __init__(self) -> None:
        self.models: Dict[str, CosmoglobeModel] = {}
        self._default_model: Optional[CosmoglobeModel] = None

    @property
    def default_model(self) -> CosmoglobeModel:
        """The default cosmoglobe model."""

        if self._default_model is None:
            raise ValueError(
                "default model has not yet been set. A default can be set "
                "using `set_default_model`"
            )

        return self._default_model

    def set_default_model(self, version: str) -> None:
        """Sets the default sky model."""

        self._default_model = self.models[version]

    def register_model(self, model: CosmoglobeModel) -> None:
        """Adds a new sky model to the registry."""

        if (version := model.version) in self.models:
            raise ValueError(f"model by version {version} is already registered.")

        self.models[version] = model

    def get_model(self, version: str) -> CosmoglobeModel:
        """Returns a registered sky model."""
        try:
            return self.models[version]
        except KeyError:
            raise ModelNotFoundError(
                f"model {version} was not found in the registry. "
                f"Available models are: {list(self.models.keys())}"
            )


cosmoglobe_registry = CosmoglobeRegistry()
cosmoglobe_registry.register_model(
    CosmoglobeModel(
        version="BeyondPlanck",
        components=[
            AME,
            CMB,
            ThermalDust,
            FreeFree,
            Radio,
            Synchrotron,
        ],
    )
)

cosmoglobe_registry.set_default_model("BeyondPlanck")
DEFAULT_COSMOGLOBE_MODEL = cosmoglobe_registry.default_model

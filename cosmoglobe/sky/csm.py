from enum import Enum
from typing import Dict, List, Type
from dataclasses import dataclass

from cosmoglobe.sky.base_components import SkyComponent

from cosmoglobe.sky.components.ame import AME
from cosmoglobe.sky.components.synchrotron import Synchrotron
from cosmoglobe.sky.components.dust import Dust
from cosmoglobe.sky.components.freefree import FreeFree
from cosmoglobe.sky.components.cmb import CMB
from cosmoglobe.sky.components.radio import Radio


class SkyComponentLabel(Enum):
    """Enums representing unique sky component labels."""

    AME = "ame"
    CMB = "cmb"
    CIB = "cib"
    DUST = "dust"
    FF = "ff"
    RADIO = "radio"
    SYNCH = "synch"
    SZ = "sz"
    ZODI = "zodi"


class DataRelease(Enum):
    """Enums representing a data Cosmoglobe data release."""

    BP10 = LATEST = "http://cosmoglobe.uio.no/data_releases/BeyondPlanck/BP10/"


@dataclass
class ModelInfo:
    """Information object for the Cosmoglobe Sky Model.

    The Cosmoglobe Sky Model will dynamically evolve with the inclusion
    of new datasets.
    """

    version: DataRelease


@dataclass
class CosmoglobeSkyModel:
    """The sky components making up Cosmoglobe Sky Model."""

    components: Dict[SkyComponentLabel, Type[SkyComponent]]
    info: ModelInfo

    @property
    def comp_labels(self) -> List[str]:
        return [comp.value for comp in self.components.keys()]
        
    def __str__(self) -> str:
        repr = "========Cosmoglobe Sky Model========\n"
        repr += f"version: {self.info.version}\n"
        repr += f"components: {', '.join(self.comp_labels)}"

        return repr


class RegisteredSkyModels:
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

        return self.models[name]


REGISTERED_SKY_MODELS = RegisteredSkyModels()
REGISTERED_SKY_MODELS.register_model(
    name="BeyondPlanck",
    model=CosmoglobeSkyModel(
        components={
            SkyComponentLabel.AME: AME,
            SkyComponentLabel.CMB: CMB,
            SkyComponentLabel.DUST: Dust,
            SkyComponentLabel.FF: FreeFree,
            SkyComponentLabel.RADIO: Radio,
            SkyComponentLabel.SYNCH: Synchrotron,
        },
        info=ModelInfo(version=DataRelease.LATEST),
    ),
)

DEFUALT_SKY_MODEL = REGISTERED_SKY_MODELS.get_model("BeyondPlanck")

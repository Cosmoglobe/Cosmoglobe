from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, TYPE_CHECKING

from astropy.units import Unit

from cosmoglobe.sky._base_components import SkyComponent

if TYPE_CHECKING:
    from cosmoglobe.sky._chain_context import ChainContext


@dataclass
class ChainContextRegistry:
    """Class that book-keeps and registeres contexts for components."""

    functions: Dict[Type[SkyComponent], List[ChainContext]] = field(default_factory=dict)
    mappings: Dict[Type[SkyComponent], Dict[str, str]] = field(default_factory=dict)
    units: Dict[Type[SkyComponent], Dict[str, Unit]] = field(default_factory=dict)

    def register_class_context(
        self,
        sky_component_class: Type[SkyComponent],
        functions: Optional[List[ChainContext]] = None,
        mappings: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, Unit]] = None,
    ) -> None:
        """Registers a specific context for a sky component implementation."""

        if functions is not None:
            for context in functions:
                if sky_component_class not in self.functions:
                    self.functions[sky_component_class] = []
                self.functions[sky_component_class].append(context)

        if mappings is not None:
            if sky_component_class not in self.mappings:
                self.mappings[sky_component_class] = {}
            self.mappings[sky_component_class].update(mappings)

        if units is not None:
            if sky_component_class not in self.units:
                self.units[sky_component_class] = {}
            self.units[sky_component_class].update(units)

    def get_functions(
        self, sky_component_class: Type[SkyComponent]
    ) -> List[ChainContext]:
        """Returns the context."""

        if sky_component_class not in self.functions:
            raise KeyError(f"No context is registered for class {sky_component_class=}")

        return [context for context in self.functions[sky_component_class]]

    def get_parameter_mappings(
        self, sky_component_class: Type[SkyComponent]
    ) -> Dict[str, str]:
        """Returns the mappings between notation for a component."""

        if sky_component_class not in self.mappings:
            raise KeyError(f"No mappings are registered for {sky_component_class=}")

        return self.mappings[sky_component_class]

    def get_units(self, sky_component_class: Type[SkyComponent]) -> Dict[str, Unit]:
        """Returns the mappings between notation for a component."""

        if sky_component_class not in self.units:
            raise KeyError(f"No units are registered for {sky_component_class=}")

        return self.units[sky_component_class]

from __future__ import annotations
from typing import Dict, List, Type, TYPE_CHECKING, Union

from astropy.units import Unit

from cosmoglobe.sky.cosmoglobe import DEFAULT_COSMOGLOBE_MODEL
from cosmoglobe.sky._base_components import SkyComponent

if TYPE_CHECKING:
    from cosmoglobe.sky._chain_context import ChainContext


class ChainContextRegistry:
    """Class that book-keeps and registeres contexts for components."""

    def __init__(self):
        self._context: Dict[Type[SkyComponent], List[ChainContext]] = {
            component: [] for component in DEFAULT_COSMOGLOBE_MODEL.components
        }
        self._mappings: Dict[Type[SkyComponent], Dict[str, str]] = {
            component: {} for component in DEFAULT_COSMOGLOBE_MODEL.components
        }
        self._units: Dict[Type[SkyComponent], Dict[str, str]] = {
            component: {} for component in DEFAULT_COSMOGLOBE_MODEL.components
        }

    def register_context(
        self, components: List[Type[SkyComponent]], context: ChainContext
    ) -> None:
        """Registers a specific context for a set of components."""

        if not components:
            for component in self._context.keys():
                self._context[component].append(context)
            return

        for component in components:
            if component not in self._context:
                raise ValueError(
                    "Cannot register a context for a non cosmoglobe component"
                )
            self._context[component].append(context)

    def register_mapping(
        self, components: List[Type[SkyComponent]], mapping: Dict[str, str]
    ) -> None:
        """Registers a specific mapping between chain and sky model notation."""

        if not components:
            for component in self._mappings.keys():
                self._mappings[component].update(mapping)
            return

        for component in components:
            if component not in self._mappings:
                raise ValueError(
                    "Cannot register a mapping for a non cosmoglobe component"
                )
            self._mappings[component].update(mapping)

    def register_units(
        self, components: List[Type[SkyComponent]], units: Dict[str, Union[str, Unit]]
    ) -> None:
        """Registers a units for values from the chain."""

        if not components:
            for component in self._units.keys():
                self._units[component].update(units)
            return

        for component in components:
            if component not in self._units:
                raise ValueError(
                    "Cannot register a unit for a non cosmoglobe component"
                )
            self._units[component].update(units)

    def get_context(self, component: Type[SkyComponent]) -> List[ChainContext]:
        """Returns the context."""

        if component not in self._context:
            raise KeyError(f"No context is registered for {component=}")

        return [context for context in self._context[component]]

    def get_mappings(self, component: Type[SkyComponent]) -> Dict[str, str]:
        """Returns the mappings between notation for a component."""

        if component not in self._mappings:
            raise KeyError(f"No mappings are registered for {component=}")

        return self._mappings[component]

    def get_units(self, component: Type[SkyComponent]) -> Dict[str, str]:
        """Returns the mappings between notation for a component."""

        if component not in self._units:
            raise KeyError(f"No units are registered for {component=}")

        return self._units[component]

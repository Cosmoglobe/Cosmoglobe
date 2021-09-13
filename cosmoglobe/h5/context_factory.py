from typing import Dict, List, Type, TYPE_CHECKING

from cosmoglobe.sky import COSMOGLOBE_COMPS

if TYPE_CHECKING:
    from cosmoglobe.h5.context import ChainContext


class ChainContextFactory:
    """Factory that book-keeps and registeres contexts for components."""

    def __init__(self):
        self._context = {component: [] for component in COSMOGLOBE_COMPS}
        self._mappings = {component: {} for component in COSMOGLOBE_COMPS}
        self._units = {component: {} for component in COSMOGLOBE_COMPS}

    def register_context(
        self, components: List[str], context: Type["ChainContext"]
    ) -> None:
        """Registers a specific context for a set of components."""

        if not components:
            for component in self._context.keys():
                self._context[component].append(context())
            return

        for component in components:
            if component in self._context:
                self._context[component].append(context())
            else:
                raise ValueError(
                    "Cannot register a context for a non cosmoglobe component"
                )

    def register_mapping(self, components: List[str], mapping: Dict[str, str]) -> None:
        """Registers a specific mapping between chain and sky model notation."""

        if not components:
            for component in self._mappings.keys():
                self._mappings[component].update(mapping)
            return

        for component in components:
            if component in self._mappings:
                self._mappings[component].update(mapping)
            else:
                raise ValueError(
                    "Cannot register a mapping for a non cosmoglobe component"
                )

    def register_units(self, components: List[str], units: Dict[str, str]) -> None:
        """Registers a units for values from the chain."""

        if not components:
            for component in self._units.keys():
                self._units[component].update(units)
            return

        for component in components:
            if component in self._units:
                self._units[component].update(units)
            else:
                raise ValueError(
                    "Cannot register a unit for a non cosmoglobe component"
                )

    def get_pre_context(self, component: str) -> List["ChainContext"]:
        """Returns the context prior to extracting from chain."""
        if component in self._context:
            return [context for context in self._context.get(component) if context.PRE]
        else:
            raise KeyError(f"No context is registered for {component=}")

    def get_post_context(self, component: str) -> List["ChainContext"]:
        """Returns the context post to extracting from chain."""
        if component in self._context:
            return [
                context for context in self._context.get(component) if not context.PRE
            ]
        else:
            raise KeyError(f"No context is registered for {component=}")

    def get_mappings(self, component: str) -> Dict[str, str]:
        """Returns the mappings between notation for a component."""
        if component in self._mappings:
            return self._mappings[component]
        else:
            raise KeyError(f"No mappings are registered for {component=}")

    def get_units(self, component: str) -> Dict[str, str]:
        """Returns the mappings between notation for a component."""
        if component in self._units:
            return self._units[component]
        else:
            raise KeyError(f"No units are registered for {component=}")

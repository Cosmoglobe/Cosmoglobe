class ModelNotFoundError(Exception):
    """Raised when a model is not found in the registry."""


class ComponentNotFoundError(Exception):
    """Raised if a component is missing from the model."""


class CosmoglobeModelError(Exception):
    """Raised if there is an issue with the initialization of a CosmoglobeModel."""


class ComponentError(Exception):
    """Raised if there is an issue with a component in the sky model."""


class NsideError(Exception):
    """Raised if there is an issue with the nside of a map."""

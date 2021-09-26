class ComponentNotFoundError(Exception):
    """Raised if a component is missing from the model."""


class ComponentError(Exception):
    """Raised if there is an issue with a component in the sky model."""


class NsideError(Exception):
    """Raised if there is an issue with the nside of a map."""

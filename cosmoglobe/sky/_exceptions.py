class ModelComponentNotFoundError(Exception):
    """Raised if a component is missing from the model."""


class ModelComponentError(Exception):
    """Raised if there is an issue with a component in the model."""


class SkyModelComponentError(Exception):
    """Raised if there is an issue with a component in the model."""


class NsideError(Exception):
    """Raised if there is an issue with the nside of a map."""

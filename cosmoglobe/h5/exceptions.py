class ChainItemNotFoundError(Exception):
    """Raised when an item is missing from the chain."""


class ChainSampleError(Exception):
    """Raised when an item is missing from the chain."""


class ChainComponentNotFoundError(Exception):
    """Raised when an item is missing from the chain."""


class ChainFormatError(Exception):
    """Raised when there is a format error with the chain."""

    def __init__(self, message="Chain is empty or missing the 'parameters' group"):
        self.message = message
        super().__init__(self.message)

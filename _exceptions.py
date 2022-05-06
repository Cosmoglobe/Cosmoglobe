
class OutputDirectoryException(Exception):
    """Raised if there is an issue related to the output directory."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

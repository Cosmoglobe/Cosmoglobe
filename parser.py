from __future__ import annotations

from pathlib import Path

from _exceptions import OutputDirectoryException


class Parser:
    """Parser for Commander3 parameterfiles."""

    def __init__(self, file: Path) -> None:
        self.file = file
        if not self.file.is_file():
            raise FileNotFoundError(
                f"parameterfile '{self.file.absolute()!s}' not found"
            )

    def parse_parameterfile(self):
        ...

    def parse_monte_carlo_options(self):
        ...

    def parse_included_datasets(self):
        ...

    def parse_comp_parameters(self):
        ...

    def get_output_dir(self) -> Path:
        with open(self.file, "r") as file:
            for line in file:
                if line.lower().startswith("output_directory"):
                    output_dir = line.split("=")[1]
                    break
            else:
                raise OutputDirectoryException(
                    "OUTPUT_DIRECTORY field missing in parameterfile."
                )
        return Path(output_dir)


if __name__ == "__main__":
    file = Path("param_tesst.txt")
    parser = Parser(file)

    print(parser.get_output_dir())

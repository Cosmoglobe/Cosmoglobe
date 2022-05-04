from dataclasses import dataclasses, field
import typing

@dataclasses
class Algorithm_params:
    parameter: str
    value: "typing.Any"



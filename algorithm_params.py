from dataclasses import dataclasses, field
from parameters import parameter
import typing

import pydantic

@dataclasses
class Algorithm_params(parameter):
    parameter: str
    value: "typing.Any"

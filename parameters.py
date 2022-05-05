from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Union

import pydantic

class parameter(pydantic.BaseModel):
    cpar: str
    value: Union[int, float, str]
    comment: Optional[str]
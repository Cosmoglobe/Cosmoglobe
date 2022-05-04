from dataclasses import dataclass, field
from __future__ import annotations

@dataclass
class parameter:
    key: str
    value: int | float | str
    comment: str=''
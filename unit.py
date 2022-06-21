from __future__ import annotations

from enum import Enum, unique
from pydantic import BaseModel

@unique
class Unit(Enum):
    MK_CMB: 'mK_CMB' 
    UK_CMB: 'uK_CMB'

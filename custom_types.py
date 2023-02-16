from __future__ import annotations
from typing import NoReturn, NewType

import numpy as np


class Point:
    def __init__(self, x: float | int, y: float | int) -> NoReturn:
        self.x = x
        self.y = y


Vector = NewType("Vector", list)
Image = NewType("Image", np.array)

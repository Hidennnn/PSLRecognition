from __future__ import annotations
from typing import NoReturn, NewType, List

import numpy as np


class Point:
    """
    Class to represent 2D points.
    """

    def __init__(self, x: float | int, y: float | int) -> NoReturn:
        self.x = x
        self.y = y


# Type Vector is a list contain objects Point,
# generated in function vector_of_points and used as input in function distance.
Vector = NewType("Vector", List[Point])

# Type Image is a np.array which representing loaded images.
Image = NewType("Image", np.array)

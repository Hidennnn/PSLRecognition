from __future__ import annotations
from typing import NoReturn, NewType

import numpy as np


class Point:
    """
    Class to represent 2D points.
    """

    def __init__(self, x: float | int, y: float | int) -> NoReturn:
        self.x = x
        self.y = y


# Type Vector is a np.array which contains objects Point, generated in function make_vector_of_points.
# Used as input in function make_distances.
Vector = NewType("Vector", np.ndarray)

# Type Distances is a np.array which contains distances between Points, generated in function make_distances.
# Used as input to neural network.
Distances = NewType("Distances", np.ndarray)

# Type Image is a np.array which representing loaded images.
Image = NewType("Image", np.ndarray)



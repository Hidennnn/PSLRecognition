from __future__ import annotations
from typing import List, NamedTuple, Tuple, Any
from math import sqrt

from custom_types import Point, Vector, Distances, Image
from custom_exceptions import PoseNotDetectedError, LeftHandNotDetectedError, RightHandNotDetectedError, \
    VectorIsNoneError
from utils import open_img

import cv2
import mediapipe as mp
import numpy as np


def make_vector_of_points(source: Image | str) -> Vector:
    """
    Function to detect elbows, shoulders and hands landmarks and return as vector of 2D points.

    :param source: Image or path to Image.
    :raises PoseNotDetectedError: Pose landmarks are not detected.
    :raises RightHandNotDetectedError: Right-hand landmarks are not detected.
    :raises LeftHandNotDetectedError: Left-hand landmarks are not detected.
    :raises PathToImageIsIncorrectError: Path passed to function is not path to Image.
    :raises ImageNotExistsError: Image is None.
    :return: Vector of points if detected. Otherwise, None.
    """

    img = open_img(source)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp.solutions.holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(img_rgb)

    test_detection(results)

    vector = make_vector(results)

    return vector


def test_detection(results: NamedTuple) -> Tuple[Any, Any, Any]:
    """
    Function to check if Holistic model from MediaPipe detected pose, left hand and right hand.
    Help function to make_vector_of_points.

    :param results: Results of Holistic model detection.
    :raises PoseNotDetectedError: Pose landmarks are not detected.
    :raises RightHandNotDetectedError: Right-hand landmarks are not detected.
    :raises LeftHandNotDetectedError: Left-hand landmarks are not detected.
    :return: Pose, left-hand and right-hand landmarks if detected.
    """

    try:
        pose = results.pose_landmarks.landmark
    except AttributeError:
        raise PoseNotDetectedError

    try:
        right_hand = results.right_hand_landmarks.landmark
    except AttributeError:
        raise RightHandNotDetectedError

    try:
        left_hand = results.left_hand_landmarks.landmark
    except AttributeError:
        raise LeftHandNotDetectedError

    return pose, right_hand, left_hand


def make_vector(results: Tuple[Any, Any, Any]) -> Vector:
    """
    Function to make vector based on detected points. (Help function for make_vector_of_points).
    Help function to make_vector_of_points.

    :param results: Results of Holistic model detection.
    :return: Vector of 2D points.
    """

    pose, right_hand, left_hand = results

    vector = Vector(np.array([]))

    for point in pose[11:15]:
        np.append(vector, [Point(point.x, point.y)])

    for point in right_hand:
        np.append(vector, [Point(point.x, point.y)])

    for point in left_hand:
        np.append(vector, [Point(point.x, point.y)])

    return vector


def euclidean_distance(point1: Point, point2: Point) -> float:
    """
    Function to compute Euclidean distance between 2 points in 2D.

    :param point1: First point.
    :param point2: Second point.
    :return: Euclidean distance.
    """

    return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def make_distance(vector: Vector) -> Distances:
    """
    Function to compute distances between every point in Vector.

    :param vector: Vector of 2D points.
    :return: List of computed distances.
    """

    if not vector:
        raise VectorIsNoneError

    distances = Distances(np.array([]))
    for first_index in range(45):
        for second_index in range(first_index + 1, 46):
            np.append(distances, euclidean_distance(vector[first_index], vector[second_index]))

    return distances

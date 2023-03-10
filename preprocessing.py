from __future__ import annotations
from typing import List, NamedTuple, Tuple, Any

from custom_types import Point, Vector, Image
from custom_exceptions import PathToImageNotExistsError, ImageNotExistsError

import cv2
import mediapipe as mp
from math import sqrt


def euclidean_distance(point1: Point, point2: Point) -> float:
    """
    Function compute Euclidean distance between 2 points in 2D.
    :param point1: First point.
    :param point2: Second point.
    :return: Euclidean distance.
    """
    return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def test_detection(results: NamedTuple) -> Tuple[Any, Any, Any] | None:
    """
    Function checks if Holistic model from MediaPipe detected pose, left hand and right hand.

    :param results: Results of Holistic model detection.
    :return: If Holistic detect everything what is required, landmarks of pose, left hand and right hand are returned
        seperetly. Otherwise, None is returned.
    """
    try:
        pose = results.pose_landmarks.landmark
    except AttributeError:
        print("Holistic doesn't detect pose.")
        return None, None, None

    try:
        left_hand = results.left_hand_landmarks.landmark
    except AttributeError:
        print("Holistic doesn't detect left hand.")
        return None, None, None

    try:
        right_hand = results.right_hand_landmarks.landmark
    except AttributeError:
        print("Holistic doesn't detect right hand pose.")
        return None, None, None

    return pose, left_hand, right_hand


def vector_of_points(source: str | Image) -> Vector | None:
    """
    Function detect characteristic points of elbows, shoulders, and hands and return as vector of points in 2D
    :param source: Path to image or Image.
    :return: Vector of points or None if something wasn't detected.
    """
    if isinstance(source, str):
        image = cv2.imread(source)
        if not image:
            raise PathToImageNotExistsError
    else:
        image = source
        if image is None:
            raise ImageNotExistsError

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(img_rgb)

    pose, left_hand, right_hand = test_detection(results)

    if not pose:
        return

    vector = Vector([])
    for number in range(11, 15):
        vector.append(Point(pose[number].x, pose[number].y))

    for number in range(len(right_hand)):
        vector.append(Point(right_hand[number].x, right_hand[number].y))

    for number in range(len(left_hand)):
        vector.append(Point(left_hand[number].x, left_hand[number].y))

    return vector


def distance(vector: Vector) -> List[float]:
    """
    Function compute distance between every point in vector.
    :param vector: Vector of 2D points.
    :return: List of computed distances.
    """
    dist = []
    for x in range(45):
        for z in range(x + 1, 46):
            dist.append(euclidean_distance(vector[x], vector[z]))

    return dist

from __future__ import annotations
from typing import List, NamedTuple, Tuple, Any

from custom_types import Point, Vector, Image
from custom_exceptions import PoseNotDetectedError, LeftHandNotDetectedError, RightHandNotDetectedError
from utils import open_img

import cv2
import mediapipe as mp
from math import sqrt


def euclidean_distance(point1: Point, point2: Point) -> float:
    """
    Function to compute Euclidean distance between 2 points in 2D.
    :param point1: First point.
    :param point2: Second point.
    :return: Euclidean distance.
    """
    return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def test_detection(results: NamedTuple) -> Tuple[Any, Any, Any] | None:
    """
    Function to check if Holistic model from MediaPipe detected pose, left hand and right hand.

    :param results: Results of Holistic model detection.
    :return: Pose, left-hand and right-hand landmarks if detected.
    """

    try:
        pose = results.pose_landmarks.landmark
    except AttributeError:
        raise PoseNotDetectedError

    try:
        left_hand = results.left_hand_landmarks.landmark
    except AttributeError:
        raise LeftHandNotDetectedError

    try:
        right_hand = results.right_hand_landmarks.landmark
    except AttributeError:
        raise RightHandNotDetectedError

    return pose, left_hand, right_hand


def vector_of_points(source: Image | str) -> Vector | None:
    """
    Function to detect elbows, shoulders and hands landmarks and return as vector of 2D points.

    :param source: Image or path to Image.
    :return: Vector of points if detected. Otherwise, None.
    """

    img = open_img(source)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp.solutions.holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(img_rgb)

    pose, left_hand, right_hand = test_detection(results)

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

from __future__ import annotations
from typing import List, NamedTuple, Tuple, Any
import os

from custom_types import Point, Vector

import cv2
import mediapipe as mp
from math import sqrt


def euclidean_distance(point1: Point, point2: Point) -> float:
    return sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def test_detection(results: NamedTuple) -> Tuple[Any, Any, Any] | None:
    try:
        pose = results.pose_landmarks.landmark
    except AttributeError:
        print("Holistic doesn't detect pose.")
        return

    try:
        left_hand = results.left_hand_landmarks.landmark
    except AttributeError:
        print("Holistic doesn't detect left hand.")
        return

    try:
        right_hand = results.right_hand_landmarks.landmark
    except AttributeError:
        print("Holistic doesn't detect right hand pose.")
        return

    return pose, left_hand, right_hand


def vector_of_points(source: str) -> Vector | None:
    if os.path.exists(source):
        img = cv2.imread(source)
    else:
        print("Source doesn't exist")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    dist = []
    for x in range(45):
        for z in range(x + 1, 46):
            dist.append(euclidean_distance(vector[x], vector[z]))

    return dist

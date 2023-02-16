from __future__ import annotations
from typing import NoReturn, Tuple, Any

from custom_types import Image

import cv2
import mediapipe as mp
import numpy as np


def center_of_image(image: Image) -> Tuple[float | Any, float | Any]:
    (h, w) = image.shape[:2]
    return w / 2, h / 2


def rescale_frame(frame: np.array, percent: int = 100) -> Image:
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def drawing_points_video(source: str | int, rescale: int = 100, min_detection_confidence: float = 0.5,
                         min_tracking_confidence: float = 0.5, window_name: str = "", points_color: tuple = (0, 0, 255),
                         points_thickness: float | int = 1, points_radius: float | int = 1,
                         connect_color: tuple = (50, 255, 0), connect_thickness: float | int = 1,
                         connect_radius: float | int = 1) -> NoReturn | None:
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Camera doesn't work or file doesn't exist")
        return

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        drawing_points_frame(img, rescale, min_detection_confidence, min_tracking_confidence, window_name,
                             points_color, points_thickness, points_radius, connect_color, connect_thickness,
                             connect_radius, waitkey=1, destroy_windows=False)

    cv2.destroyWindow(window_name)


def drawing_points_frame(source: str | Image, rescale: int = 100, min_detection_confidence: float = 0.5,
                         min_tracking_confidence: float = 0.5, window_name: str = "", points_color: tuple = (0, 0, 255),
                         points_thickness: float | int = 1, points_radius: float | int = 1,
                         connect_color: tuple = (50, 255, 0), connect_thickness: float | int = 1,
                         connect_radius: float | int = 1, waitkey: int = 0, destroy_windows: bool = True) -> Image:
    if isinstance(source, str):
        img = cv2.imread(source)
    else:
        img = source

    img = rescale_frame(img, rescale)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence
    ) as holistic:
        results = holistic.process(img_rgb)

    if results.pose_landmarks:
        with mp.solutions.holistic as holistic, mp.solutions.pose as pose, mp.solutions.drawing_utils as draw:
            draw.draw_landmarks(img, results.pose_landmarks, pose.POSE_CONNECTIONS,
                                draw.DrawingSpec(
                                    color=points_color, thickness=points_thickness, circle_radius=points_radius
                                ),
                                draw.DrawingSpec(
                                    color=connect_color, thickness=connect_thickness, circle_radius=connect_radius
                                )
                                )

            draw.draw_landmarks(img, results.left_hand_landmarks, holistic.HAND_CONNECTIONS,
                                draw.DrawingSpec(
                                    color=points_color, thickness=points_thickness, circle_radius=points_radius
                                ),
                                draw.DrawingSpec(
                                    color=connect_color, thickness=connect_thickness, circle_radius=connect_radius
                                )
                                )

            draw.draw_landmarks(img, results.right_hand_landmarks, holistic.HAND_CONNECTIONS,
                                draw.DrawingSpec(
                                    color=points_color, thickness=points_thickness, circle_radius=points_radius
                                ),
                                draw.DrawingSpec(
                                    color=connect_color, thickness=connect_thickness, circle_radius=connect_radius
                                )
                                )

    cv2.imshow(window_name, img)
    cv2.waitKey(waitkey)

    if destroy_windows:
        cv2.destroyWindow(window_name)

    return img


def photo_mirror(source: str | Image, destination_path: str) -> NoReturn | None:

    if isinstance(source, str):
        img = cv2.imread(source)
        if not img:
            print("Photo doesn't exist")
            return
    else:
        img = source

    imgX = np.flip(img, axis=1)
    cv2.imwrite(destination_path, imgX)


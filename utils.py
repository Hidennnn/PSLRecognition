from __future__ import annotations
from typing import NoReturn, Tuple

from custom_types import Image
from custom_exceptions import PathToImageNotExistError, PathToVideoNotExistError, ImageNotExistError

import cv2
import mediapipe as mp
import numpy as np


def center_of_image(image: Image) -> Tuple[float, float]:
    """
    Function to compute center of image.

    :param image: Image which center we want to compute.

    :return: center: Center coordinate (x,y).
    """
    try:
        (h, w) = image.shape[:2]
        return w / 2, h / 2
    except AttributeError:
        raise ImageNotExistError


def rescale_image(image: Image, percent: int = 100) -> Image:
    """
    Function rescales image.

    :param image: Image which we want to rescale. If image is None, AttributeError is raised.
    :param percent: Percent of original size returned image will be.
    :return: rescaled_image: Rescaled image which is percent of original size.
    """

    try:
        width = int(image.shape[1] * percent / 100)
        height = int(image.shape[0] * percent / 100)
        dim = (width, height)

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    except AttributeError:
        raise ImageNotExistError


def drawing_points_video(source: str | int, rescale: int = 100, min_detection_confidence: float = 0.5,
                         min_tracking_confidence: float = 0.5, window_name: str = "", points_color: tuple = (0, 0, 255),
                         points_thickness: float | int = 1, points_radius: float | int = 1,
                         connect_color: tuple = (50, 255, 0), connect_thickness: float | int = 1,
                         connect_radius: float | int = 1) -> NoReturn:
    """
    Function detects, draws characteristic points and shows input video. Return None on error.

    :param source: Source of video.
    :param rescale: Showed video will be rescale% of original size.
    :param min_detection_confidence: Minimum detection confidence for detecting model (Holistic mediapipe).
    :param min_tracking_confidence: Minimum tracking confidence for detecting model (Holistic mediapipe).
    :param window_name: Name of window which shows output video.
    :param points_color: Color of characteristic points.
    :param points_thickness: Thickness of characteristic points.
    :param points_radius: Radius of characteristic points.
    :param connect_color: Color of characteristic points' connections.
    :param connect_thickness: Thickness of characteristic points' connections.
    :param connect_radius: Radius of characteristic points' connections.
    :return: Return None on error. Otherwise, NoReturn.
    """

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise PathToVideoNotExistError

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        drawing_points_image(img, rescale, min_detection_confidence, min_tracking_confidence, window_name,
                             points_color, points_thickness, points_radius, connect_color, connect_thickness,
                             connect_radius, waitkey=1, destroy_window=False)

    cv2.destroyWindow(window_name)


def drawing_points_image(source: str | Image, rescale: int = 100, min_detection_confidence: float = 0.5,
                         min_tracking_confidence: float = 0.5, window_name: str = "", points_color: tuple = (0, 0, 255),
                         points_thickness: float | int = 1, points_radius: float | int = 1,
                         connect_color: tuple = (50, 255, 0), connect_thickness: float | int = 1,
                         connect_radius: float | int = 1, waitkey: int = 0, destroy_window: bool = True) -> Image:
    """
    Function detects, draws characteristic points and shows input image by certain time. Return None on error.

    :param source: Source of image.
    :param rescale: Showed image will be rescale% of original size.
    :param min_detection_confidence: Minimum detection confidence for detecting model (Holistic mediapipe).
    :param min_tracking_confidence: Minimum tracking confidence for detecting model (Holistic mediapipe).
    :param window_name: Name of window which shows output video.
    :param points_color: Color of characteristic points.
    :param points_thickness: Thickness of characteristic points.
    :param points_radius: Radius of characteristic points.
    :param connect_color: Color of characteristic points' connections.
    :param connect_thickness: Thickness of characteristic points' connections.
    :param connect_radius: Radius of characteristic points' connections.
    :param waitkey: How much time image will be showed. On default will be showed until press any key.
    :param destroy_window: Set if you want to destroy window where image will be showed.
    :return: Image with characteristic points.
    """
    if isinstance(source, str):
        img = cv2.imread(source)
        if not img:
            raise PathToImageNotExistError
    else:
        img = source
        if not img:
            raise ImageNotExistError

    img = rescale_image(img, rescale)
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

    if destroy_window:
        cv2.destroyWindow(window_name)

    return img


def image_mirror(source: str | Image, destination_path: str = None) -> Image:
    """
    Function mirrors and save input image if you want to.
    :param source: Source to image or Image.
    :param destination_path: If you give destination path to save, mirrored image will be saved there.
    :return: Mirrored image.
    """

    if isinstance(source, str):
        img = cv2.imread(source)
        if not img:
            raise PathToImageNotExistError
    else:
        img = source
        if not img:
            raise ImageNotExistError

    img_x = np.flip(img, axis=1)

    if destination_path:
        cv2.imwrite(destination_path, img_x)

    return img_x

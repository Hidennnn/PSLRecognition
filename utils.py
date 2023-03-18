from __future__ import annotations
from typing import NoReturn, Tuple

from custom_types import Image
from custom_exceptions import PathToImageIsIncorrectError, PathToVideoIsIncorrectError, \
                                ImageNotExistsError, CameraIndexIsIncorrect

import cv2
import mediapipe as mp
import numpy as np


def open_img(source: Image | str) -> Image:
    """
    Function to check if correct Image is passed to function or open Image from path.

    :param source: Image or path to Image.
    :return: Checked or opened Image.
    """

    if isinstance(source, str):
        img = cv2.imread(source)
        if not img:
            raise PathToImageIsIncorrectError
    else:
        img = source
        if not img:
            raise ImageNotExistsError

    return img


def center_of_img(source: Image | str) -> Tuple[float, float]:
    """
    Function to compute Image center.

    :param source: Image or path to Image.
    :return: Image center coordinates in format (x,y).
    """

    source = open_img(source)

    (h, w) = source.shape[:2]
    return w / 2, h / 2


def rescale_img(source: Image | str, rescale_factor: int = 100) -> Image:
    """
    Function to rescale image.

    :param source: Image or path to Image.
    :param rescale_factor: What percentage of the size of the original Image will be the Image after rescaling.
    :return: rescaled_image: Rescaled Image.
    """

    img = open_img(source)

    width = int(img.shape[1] * rescale_factor / 100)
    height = int(img.shape[0] * rescale_factor / 100)

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def detect_and_draw_landmarks(img: Image, min_detection_confidence: float = 0.5,
                              min_tracking_confidence: float = 0.5, points_color: tuple = (0, 0, 255),
                              points_thickness: float | int = 1, points_radius: float | int = 1,
                              connect_color: tuple = (50, 255, 0), connect_thickness: float | int = 1,
                              connect_radius: float | int = 1) -> Image | None:
    """
    Function to detect and draw landmarks on Image.

    :param img: Image.
    :param min_detection_confidence: Minimum detection confidence for detecting model (Holistic mediapipe).
    :param min_tracking_confidence: Minimum tracking confidence for detecting model (Holistic mediapipe).
    :param points_color: Color of landmarks.
    :param points_thickness: Thickness of landmarks.
    :param points_radius: Radius of landmarks.
    :param connect_color: Color of landmarks' connections.
    :param connect_thickness: Thickness of landmarks' connections.
    :param connect_radius: Landmarks' connections.
    :return: Image with drawn landmarks if detected. Otherwise, without drawn landmarks
    """

    with mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence
    ) as holistic:
        results = holistic.process(img)

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

    return img


def drawing_points_video(source: str | int, rescale: int = 100, window_name: str = "",
                         min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                         points_color: tuple = (0, 0, 255), points_thickness: float | int = 1,
                         points_radius: float | int = 1, connect_color: tuple = (50, 255, 0),
                         connect_thickness: float | int = 1, connect_radius: float | int = 1) -> NoReturn:
    """
    Function to detect, draw landmarks and show input video.

    :param source: Path to video or camera number.
    :param rescale: What percentage of the size of the original video will be the output video.
    :param window_name: Name of window with output video.
    :param min_detection_confidence: Minimum detection confidence for detecting model (Holistic mediapipe).
    :param min_tracking_confidence: Minimum tracking confidence for detecting model (Holistic mediapipe).
    :param points_color: Color of landmarks.
    :param points_thickness: Thickness of landmarks.
    :param points_radius: Radius of landmarks.
    :param connect_color: Color of landmarks' connections.
    :param connect_thickness: Thickness of landmarks' connections.
    :param connect_radius: Landmarks' connections.
    :return: NoReturn.
    """

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        if isinstance(source, str):
            raise PathToVideoIsIncorrectError
        else:
            raise CameraIndexIsIncorrect

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        #landmarks for one frame is like for Image.
        drawing_points_img(img, rescale, window_name, min_detection_confidence, min_tracking_confidence, points_color,
                           points_thickness, points_radius, connect_color, connect_thickness, connect_radius,
                           wait_key=1, destroy_window=False)

    cv2.destroyWindow(window_name)


def drawing_points_img(source: Image | str, rescale: int = 100, window_name: str = "",
                       min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                       points_color: tuple = (0, 0, 255), points_thickness: float | int = 1,
                       points_radius: float | int = 1, connect_color: tuple = (50, 255, 0),
                       connect_thickness: float | int = 1, connect_radius: float | int = 1, wait_key: int = 0,
                       destroy_window: bool = True) -> Image:
    """
    Function to detect, drawn landmarks and show input Image by certain time.

    :param source: Image or path to Image.
    :param rescale: What percentage of the size of the original Image will be the output Image.
    :param window_name: Name of window with output video.
    :param min_detection_confidence: Minimum detection confidence for detecting model (Holistic mediapipe).
    :param min_tracking_confidence: Minimum tracking confidence for detecting model (Holistic mediapipe).
    :param points_color: Color of landmarks.
    :param points_thickness: Thickness of landmarks.
    :param points_radius: Radius of landmarks.
    :param connect_color: Color of landmarks' connections.
    :param connect_thickness: Thickness of landmarks' connections.
    :param connect_radius: Landmarks' connections.
    :param wait_key: How much time Image will be showed. On default will be showed until press any key.
    :param destroy_window: Set if you want to destroy window where Image will be showed.
    :return: Image with drawn landmarks.
    """

    img = open_img(source)

    img = rescale_img(img, rescale)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = detect_and_draw_landmarks(
                                    img_rgb, min_detection_confidence, min_tracking_confidence, points_color,
                                    points_thickness, points_radius, connect_color, connect_thickness, connect_radius
                                    )

    cv2.imshow(window_name, img)
    cv2.waitKey(wait_key)

    if destroy_window:
        cv2.destroyWindow(window_name)

    return img


def img_mirror(source: str | Image, destination_path: str = None) -> Image:
    """
    Function to mirror and save input image if you want to.

    :param source: Image or path to Image.
    :param destination_path: If you give destination path to save, mirrored image will be saved there.
    :return: Mirrored Image.
    """

    img = open_img(source)
    img_mirrored = np.flip(img, axis=1)

    if destination_path:
        cv2.imwrite(destination_path, img_mirrored)

    return img_mirrored

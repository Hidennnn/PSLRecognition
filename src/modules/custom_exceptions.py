from __future__ import annotations

from typing import NoReturn


class PoseNotDetectedError(Exception):
    """
    Raise when pose landmarks is not detected.
    """
    def __init__(self) -> NoReturn:
        self.message = "Pose is not detected."
        super().__init__(self.message)


class LeftHandNotDetectedError(Exception):
    """
    Raise when left-hand landmarks is not detected.
    """
    def __init__(self) -> NoReturn:
        self.message = "Left hand is not detected."
        super().__init__(self.message)


class RightHandNotDetectedError(Exception):
    """
    Raise when right-hand landmarks is not detected.
    """
    def __init__(self) -> NoReturn:
        self.message = "Right hand is not detected."
        super().__init__(self.message)


class PathToImageIsIncorrectError(Exception):
    """
    Raise when images with certain path doesn't exist. In the most cases it is problem with incorrect path in
    cv2.imread().
    """
    def __init__(self) -> NoReturn:
        self.message = "Image on this path doesn't exist."
        super().__init__(self.message)


class PathToVideoIsIncorrectError(Exception):
    """
    Raise when video with certain path doesn't exist. In the most cases it is problem with cv2.VideoCapture()
    """
    def __init__(self) -> NoReturn:
        self.message = "Video on this path doesn't exist."
        super().__init__(self.message)


class CameraIndexIsIncorrect(Exception):
    """
    Raise when camera with given index is not exist.
    """

    def __init__(self) -> NoReturn:
        self.message = "Camera with given index is not exist."
        super().__init__(self.message)


class ImageNotExistsError(Exception):
    """
    Raise when input image not exist (by default image is None).
    """

    def __init__(self) -> NoReturn:
        self.message = "Image doesn't exist. Check if input image isn't None."
        super().__init__(self.message)


class CSVFilesExist(Exception):
    """
    Raise when you want to create CSV files which exist already.
    """

    def __init__(self) -> NoReturn:
        self.message = "CSV files exist, so you can't initiate them. (append_mode is False)"
        super().__init__(self.message)


class CSVFilesNotExist(Exception):
    """
    Raise when you want to append data to not existed CSV files.
    """
    def __init__(self) -> NoReturn:
        self.message = "CSV files don't exist, but you choose append mode."
        super().__init__(self.message)


class BadNumberOfFileNames(Exception):
    """
    Raise when you input bad number of files.
    """
    def __init__(self, goal: int, current: int) -> NoReturn:

        if goal > 1 and current > 1:
            self.message = f"Function require {goal} files, but {current} are given."
        elif goal > 1 and 0 <= current <= 1:
            self.message = f"Function require {goal} files, but {current} is given."
        elif 0 <= goal <= 1 and current > 1:
            self.message = f"Function require {goal} file, but {current} are given."
        elif 0 <= goal <= 1 and 0 <= current <= 1:
            self.message = f"Function require {goal} file, but {current} is given."
        else:
            self.message = "Error about number of file names are raise, but something bad happen with communicate."

        super().__init__(self.message)


class VectorIsNoneError(Exception):
    """
    Raise when your vector is None.
    """
    def __init__(self) -> NoReturn:
        self.message = "Vector is None."
        super().__init__(self.message)


from __future__ import annotations
from typing import List, Tuple, NoReturn
import os

from utils import rescale_img, center_of_image, image_mirror
from custom_types import Image

import cv2


def make_augmentation(database: str, rescale_ratios: List[int], images_in_classes: List[int], flip_degrees: List[int],
                      labels: List[str]) -> NoReturn:
    """
    Function to make augmentation (rescale, mirror and flip).
    :param database: Path to database where image which you want augmented are.
    :param rescale_ratios: Ratios of rescale.
    :param images_in_classes: List which shows how many images are in certain class. It is help to generate names.
    :param flip_degrees: List with degrees of flip.
    :param labels: Name of classes.
    :return: No return.
    """
    class_index = 0
    for folder in os.listdir(database):
        for file in os.listdir(f"{database}\\{folder}"):
            file_base = labels[class_index]

            original_image = cv2.imread(f"{database}\\{folder}\\{file}")
            mirrored_image = image_mirror(original_image,
                                          f"{database}\\{folder}\\{file_base}_{images_in_classes[class_index]}.jpg")

            images_in_classes[class_index] += 1

            center = center_of_image(original_image)

            manage_augmentation(original_image, center, database, folder, flip_degrees,
                                file_base, class_index, images_in_classes, rescale_ratios)

            manage_augmentation(mirrored_image, center, database, folder, flip_degrees,
                                file_base, class_index, images_in_classes, rescale_ratios)

    class_index += 1


def manage_augmentation(original_image: Image, center: Tuple[float | int, float | int], database: str, folder: str,
                        flip_degrees: List[int], file_base: str, class_index: int, images_in_classes: List[int],
                        rescale_ratios: List[int]) -> NoReturn:

    """
    Function to make augmentation of 1 image.
    :param original_image: Image which we want to augment.
    :param center: Center of image.
    :param database: Path to database.
    :param folder: Folder of database where are original image.
    :param flip_degrees: Degrees of flip.
    :param file_base: First part of file name.
    :param class_index: Index of class.
    :param images_in_classes: List which shows how many images are in certain class. It is help to generate names.
    :param rescale_ratios: Ratios to rescale image.
    :return: No return.
    """

    manage_flip(original_image, center, f"{database}\\{folder}",
                flip_degrees, file_base, class_index, images_in_classes)

    for rescale_ratio in rescale_ratios:
        image_rescaled = resize_and_save_image(original_image, rescale_ratio,
                                               f"{database}\\{folder}\\{file_base}_{images_in_classes[class_index]}.jpg")
        images_in_classes[class_index] += 1
        center = center_of_image(image_rescaled)
        manage_flip(original_image, center, f"{database}\\{folder}",
                    flip_degrees, file_base, class_index, images_in_classes)


def resize_and_save_image(original_image: Image, size_ratio: int, name_of_new_image: str) -> Image:
    """
    Function resize and save image.
    :param original_image: Original image as Image type.
    :param size_ratio: Ratio of rescale
    :param name_of_new_image: Name of new image as only image name or path.
    :return: Resized image.
    """
    image_resized = rescale_img(original_image, size_ratio)
    cv2.imwrite(name_of_new_image, image_resized)

    return image_resized


def manage_flip(original_image: Image, center: Tuple[float | int, float | int], directory: str, flip_degrees: List[int],
                file_base: str, class_index: int, images_in_classes: List[int]) -> NoReturn:
    """
    Function to manage flip of 1 image.
    :param original_image: Original image as Image type.
    :param center: Center of image.
    :param directory: Directory where you want to save image.
    :param flip_degrees: How many degrees you want to flip file.
    :param file_base: Base of filename. Important to generate name.
    :param class_index: Index of class.
    :param images_in_classes: Images in each class. Important to generate name.
    :return: No return.
    """
    for flip_degree in flip_degrees:
        flip_and_save_photo(original_image, center,
                            f"{directory}\\{file_base}_{images_in_classes[class_index]}.jpg",
                            flip_degree)
        images_in_classes[class_index] += 1


def flip_and_save_photo(original_image: Image, center_of_original_image: Tuple[float],
                        name_of_new_photo: str, flip_degree: int) -> NoReturn:
    

    rotate_matrix = cv2.getRotationMatrix2D(center_of_original_image, flip_degree, 1)
    image_flipped = cv2.warpAffine(original_image, rotate_matrix, center_of_original_image)
    cv2.imwrite(name_of_new_photo, image_flipped)

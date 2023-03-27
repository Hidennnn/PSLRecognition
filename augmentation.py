from __future__ import annotations
from typing import List, Tuple, NoReturn
import os

from utils import rescale_img, center_of_img, img_mirror, open_img
from custom_types import Image

import cv2


def make_augmentation(database: str, img_in_classes: List[int], labels: List[str], rescale_ratios: List[int],
                      flip_degrees: List[int]) -> NoReturn:
    """
    Function to make augmentation by rescale, mirror and flip.

    :param database: Path to database with Images which you want augmented.
    :param labels: Class names.
    :param img_in_classes: Number of Images in each class. Important to generate name.
    :param rescale_ratios: Rescale ratios is values which means what percentage of the size of the original Image will
    be the Image after rescaling.
    :param flip_degrees: By how many degrees you want to flip Image.
    :raises PathToImageIsIncorrectError: Path passed to function is not path to Image.
    :raises ImageNotExistsError: Image is None.
    :return: No return.
    """

    class_index = 0
    for folder in os.listdir(database):
        for file in os.listdir(f"{database}\\{folder}"):
            file_base = labels[class_index]

            original_image = open_img(f"{database}\\{folder}\\{file}")
            mirrored_image = img_mirror(original_image,
                                        f"{database}\\{folder}\\{file_base}_{img_in_classes[class_index]}.jpg")

            img_in_classes[class_index] += 1

            center = center_of_img(original_image)

            manage_augmentation(original_image, database, folder, file_base, class_index, img_in_classes, center,
                                rescale_ratios, flip_degrees)

            manage_augmentation(mirrored_image, database, folder, file_base, class_index, img_in_classes, center,
                                rescale_ratios, flip_degrees)

    class_index += 1


def manage_augmentation(original_image: Image, database: str, folder: str, file_base: str, class_index: int,
                        images_in_classes: List[int], center: Tuple[float | int, float | int],
                        rescale_ratios: List[int], flip_degrees: List[int]) -> NoReturn:
    """
    Function to make augmentation on 1 Image and save files in database. Help function to make_augmentation.

    :param original_image: Original Image which will be augmented.
    :param database: Path to database.
    :param folder: Database folder where are original Image.
    :param file_base: First part of file name.
    :param class_index: Class index based on dictionary with labels.
    :param images_in_classes: Number of Images in each class. Important to generate name.
    :param center: Center of original Image. Important to flip.
    :param rescale_ratios: Rescale ratios is values which means what percentage of the size of the original Image will
    be the Image after rescaling.
    :param flip_degrees: By how many degrees you want to flip Image.
    :return: No return.
    """

    manage_flip(original_image, f"{database}\\{folder}", file_base, class_index, images_in_classes, center,
                flip_degrees)

    for rescale_ratio in rescale_ratios:
        image_rescaled = resize_and_save_image(original_image, rescale_ratio,
                                               f"{database}\\{folder}\\{file_base}_{images_in_classes[class_index]}.jpg")
        images_in_classes[class_index] += 1
        center = center_of_img(image_rescaled)
        manage_flip(original_image, f"{database}\\{folder}", file_base, class_index, images_in_classes, center,
                    flip_degrees)


def manage_flip(original_image: Image, directory: str, file_base: str, class_index: int, images_in_classes: List[int],
                center: Tuple[float | int, float | int], flip_degrees: List[int]) -> NoReturn:
    """
    Function to manage flip of 1 Image. Help function to manage_augmentation.

    :param original_image: Original Image which will be augmented.
    :param directory: Path to folder in database where Image will be saved.
    :param file_base: Base of filename. Important to generate name.
    :param class_index: Index of class.
    :param images_in_classes: Number of Images in each class. Important to generate name.
    :param center: Center of original Image. Important to flip.
    :param flip_degrees: By how many degrees you want to flip Image.
    :return: No return.
    """
    for flip_degree in flip_degrees:
        flip_and_save_photo(original_image, f"{directory}\\{file_base}_{images_in_classes[class_index]}.jpg", center,
                            flip_degree)
        images_in_classes[class_index] += 1


def flip_and_save_photo(original_image: Image, name_of_new_photo: str, center: Tuple[float],
                        flip_degree: int) -> NoReturn:
    """
    Function to flip and save_photo. Help function to manage_flip.

    :param original_image: Original Image which will be augmented.
    :param name_of_new_photo: Name of augmented Image.
    :param center: Center of original Image. Important to flip.
    :param flip_degree: By how many degrees you want to flip Image.
    :return: NoReturn
    """

    rotate_matrix = cv2.getRotationMatrix2D(center, flip_degree, 1)
    image_flipped = cv2.warpAffine(original_image, rotate_matrix, (original_image.shape[1], original_image.shape[0]))
    cv2.imwrite(name_of_new_photo, image_flipped)


def resize_and_save_image(original_image: Image, rescale_ratio: int, new_image_name: str) -> Image:
    """
    Function to resize and save image. Help function to manage_augmentation.

    :param original_image: Original Image which will be augmented.
    :param rescale_ratio: Rescale ratio is values which means what percentage of the size of the original Image will
    be the Image after rescaling.
    :param new_image_name: New Image name.
    :return: Resized image.
    """
    image_resized = rescale_img(original_image, rescale_ratio)
    cv2.imwrite(new_image_name, image_resized)

    return image_resized

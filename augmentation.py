from __future__ import annotations
from typing import List, Tuple, NoReturn
import os

from utils import rescale_image, center_of_image, photo_mirror
from custom_types import Image

import cv2


def make_augmentation(database: str, rescale_ratios: List[int], photos_in_classes: List[int], flip_degrees: List[int],
                      labels: List[str]) -> NoReturn:

    class_index = 0
    for folder in os.listdir(database):
        for file in os.listdir(f"{database}\\{folder}"):
            file_base = labels[class_index]

            original_image = cv2.imread(f"{database}\\{folder}\\{file}")
            mirrored_image = photo_mirror(original_image,
                                          f"{database}\\{folder}\\{file_base}_{photos_in_classes[class_index]}.jpg")

            photos_in_classes[class_index] += 1

            center = center_of_image(original_image)

            manage_augmentation(original_image, center, database, folder, flip_degrees,
                                file_base, class_index, photos_in_classes, rescale_ratios)

            manage_augmentation(mirrored_image, center, database, folder, flip_degrees,
                                file_base, class_index, photos_in_classes, rescale_ratios)

    class_index += 1


def manage_augmentation(original_image: Image, center: Tuple[float | int, float | int], database: str, folder: str,
                        flip_degrees: List[int], file_base: str, class_index: int, photos_in_classes: List[int],
                        rescale_ratios: List[int]) -> NoReturn:

    manage_flip(original_image, center, f"{database}\\{folder}",
                flip_degrees, file_base, class_index, photos_in_classes)

    for rescale_ratio in rescale_ratios:
        image_rescaled = resize_and_save_photo(original_image, rescale_ratio,
                                               f"{database}\\{folder}\\{file_base}_{photos_in_classes[class_index]}.jpg")
        photos_in_classes[class_index] += 1
        center = center_of_image(image_rescaled)
        manage_flip(original_image, center, f"{database}\\{folder}",
                    flip_degrees, file_base, class_index, photos_in_classes)


def resize_and_save_photo(original_image: Image, size_ratio: int, name_of_new_photo: str) -> Image:
    image_resized = rescale_image(original_image, size_ratio)
    cv2.imwrite(name_of_new_photo, image_resized)

    return image_resized


def manage_flip(original_image: Image, center: Tuple[float | int, float | int], directory: str, flip_degrees: List[int],
                file_base: str, class_index: int, photos_in_classes: List[int]) -> NoReturn:

    for flip_degree in flip_degrees:
        flip_and_save_photo(original_image, center,
                            f"{directory}\\{file_base}_{photos_in_classes[class_index]}.jpg",
                            flip_degree)
        photos_in_classes[class_index] += 1


def flip_and_save_photo(original_image: Image, center_of_original_image: Tuple[float],
                        name_of_new_photo: str, flip_degree: int) -> NoReturn:
    rotate_matrix = cv2.getRotationMatrix2D(center_of_original_image, flip_degree, 1)
    image_flipped = cv2.warpAffine(original_image, rotate_matrix, center_of_original_image)
    cv2.imwrite(name_of_new_photo, image_flipped)

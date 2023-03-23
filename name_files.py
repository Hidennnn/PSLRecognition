from typing import List, NoReturn
import os

import numpy as np


def rename_files(path_to_base: str, path_to_not_named_images: str, labels: List[str]) -> NoReturn:
    """
    Function to name Images and move them to database. Images need to be sorted in good queue - images of next class
    need to be after Images of before class.

    :param path_to_base: Path to database where Images will be moved.
    :param path_to_not_named_images: Path to folder with not named Images.
    :param labels: List with images' class labels.
    :return: No return.
    """

    list_of_numbers = np.array([])

    for folder in os.listdir(path_to_base):
        np.append(list_of_numbers, [len(os.listdir(f"{path_to_base}\\{folder}"))])

    count = 0
    for file in os.listdir(path_to_not_named_images):
        os.rename(
                f"{path_to_not_named_images}\\{file}",
                f"{path_to_base}\\{labels[count]}\\{labels[count]}_{list_of_numbers[count]}.jpg"
                )
        list_of_numbers[count] += 1
        count += 1
        if count == 27:
            count = 0

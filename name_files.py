from typing import List, NoReturn
import os


def rename_files(path_to_base: str, path_to_not_named_images: str, labels: List[str]) -> NoReturn:
    """
    Function to name images and move them to database. Files need to be sorted in good queue - images of next class need 
    to be after images of before class. 
    class.
    :param path_to_base: Path to database where images will be moved.
    :param path_to_not_named_images: Path to not named images.
    :param labels: List with labels of images' class.
    :return: No return.
    """
    list_of_numbers = []

    for folder in os.listdir(path_to_base):
        list_of_numbers.append(len(os.listdir(f"{path_to_base}\\{folder}")))

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

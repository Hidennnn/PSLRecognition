from typing import List, NoReturn
import os


def rename_files(path_to_base: str, path_to_not_named_photos: str, labels: List[str]) -> NoReturn:
    list_of_numbers = []

    for folder in os.listdir(path_to_base):
        list_of_numbers.append(len(os.listdir(f"{path_to_base}\\{folder}")))

    count = 0
    for file in os.listdir(path_to_not_named_photos):
        os.rename(
                f"{path_to_not_named_photos}\\{file}",
                f"{path_to_base}\\{labels[count]}\\{labels[count]}_{list_of_numbers[count]}.jpg"
                )
        list_of_numbers[count] += 1
        count += 1
        if count == 27:
            count = 0

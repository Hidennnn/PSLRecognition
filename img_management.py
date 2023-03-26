from __future__ import annotations
from os.path import exists
import os
import csv
from typing import List, Any, NoReturn, Tuple

from preprocessing import make_vector_of_points, make_distance
from custom_types import Vector
from custom_exceptions import CSVFilesExist, CSVFilesNotExist, BadNumberOfFileNames, PoseNotDetectedError, \
                            LeftHandNotDetectedError, RightHandNotDetectedError


def load_img(source: str, error_file_name: str, labels: dict, vector_file_name: str = "vector.csv",
             distance_features_file_name: str = "vector_and_distances\distance_features.csv",
             distance_labels_file_name: str = "vector_and_distances\distance_labels.csv", img_names_file_name: str = "files.csv",
             append_photos: bool = True) -> NoReturn:
    """
    Function to detect landmarks on Images from source and compute distances between every Point. Then Vector,
    Distances and Image names to csv files. Vector, Distances, Distances labels and Image name on the same index
    in files are from the same Image. Images have to be sort in folders by labels in source.

    :param distance_labels_file_name:
    :param source: Path to place where folders with Images are.
    :param error_file_name: File to save Images names on which landmark won't been detected.
    :param labels: List of class to which images belong.
    :param vector_file_name: Name of file where vector of points will be saved.
    :param distance_features_file_name: Name of file where distances will be saved.
    :param distance_labels_file_name: Name of file where Distances labels will be saved.
    :param img_names_file_name: Name of file where file names will be saved.
    :param append_photos: On False function initiates new csv files. On True function appends to existed files.
    :raises CSVFilesExist: Append_photos is False but your CSV files exist. It is protecting overwriting.
    :raises CSVFilesNotExist: Append_photos is True but your CSV files not exist. It is protecting wasting time by
    chose bad files.
    :raises PathToImageIsIncorrectError: Path passed to function is not path to Image.
    :raises ImageNotExistsError: Image is None.
    :return: No return.
    """

    if (exists(vector_file_name) or exists(distance_features_file_name) or exists(distance_labels_file_name) or exists(
            img_names_file_name)) and not append_photos:
        raise CSVFilesExist

    if not (exists(vector_file_name) and exists(distance_features_file_name) and exists(distance_labels_file_name) and
            exists(img_names_file_name)) and append_photos:
        raise CSVFilesNotExist

    if not append_photos:
        init_csv((vector_file_name, distance_features_file_name, distance_labels_file_name, img_names_file_name))

    for folder in os.listdir(source):
        for file in os.listdir(f"{source}\\{folder}"):
            vector = vector_management(f"{source}\\{folder}\\{file}", file, vector_file_name,
                                       labels[file.split("_")[0]], error_file_name)

            distance_management(vector, distance_features_file_name, labels[file.split("_")[0]])
            write_csv(img_names_file_name, [file])


def init_csv(file_names: Tuple[str, str, str, str]) -> NoReturn:
    """
    Function to initiate 4 csv files where will be saved Vectors, Distances features, Distance labels and Image names.
    Help function to load_img.

    :param file_names: 4 file names.
    :return: No return
    """

    if len(file_names) != 4:
        raise BadNumberOfFileNames(4, len(file_names))

    column_names = [str(x) for x in range(92)]
    column_names.append("label")

    with open(file_names[0], "w", newline="") as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(column_names)

    column_names = [x for x in range(1035)]

    with open(file_names[1], "w", newline="") as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(column_names)

    column_names = [x for x in range(27)]

    with open(file_names[2], "w", newline="") as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(column_names)

    column_names = ["File name"]
    with open(file_names[3], "w", newline="") as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(column_names)


def vector_management(source: str, img_name: str, vector_name: str, class_index: int,
                      error_file_name: str = None) -> Vector | None:
    """
    Function to manage write information about Vectors to csv file. Help function to load_img.

    :param source: Path to Image.
    :param img_name: File name of Image.
    :param vector_name: File name where will be saved Vector.
    :param error_file_name: File name where will be saved Image name on error.
    :param class_index: Index of class to which Image belong.
    :raises PathToImageIsIncorrectError: Path passed to function is not path to Image.
    :raises ImageNotExistsError: Image is None.
    :return: Vector if landmarks is detected. Otherwise, None.
    """

    try:
        vector = make_vector_of_points(source)
    except (PoseNotDetectedError, RightHandNotDetectedError, LeftHandNotDetectedError):
        if error_file_name:
            with open(error_file_name, "a") as error:
                error.write(f"{img_name}\n")
        return

    to_save_in_csv = []
    for point in vector:
        to_save_in_csv.append(point.x)
        to_save_in_csv.append(point.y)

    to_save_in_csv.append(class_index)

    write_csv(vector_name, to_save_in_csv)

    return vector


def distance_management(vector: Vector, distance_name: str, class_index: int) -> NoReturn:
    """
    Function to compute distances between every point in Vector and save Distances to csv file.
    Help function to load_img.

    :param vector: Vector of landmarks.
    :param distance_name: File name where distances will be saved.
    :param class_index: Index of true class of features.
    :return: No return.
    """

    distance_values = make_distance(vector)

    to_save_in_csv = []
    for value in distance_values:
        to_save_in_csv.append(value)

    to_save_in_csv.append(class_index)

    write_csv(distance_name, to_save_in_csv)

def write_csv(directory: str, values: List[Any]) -> NoReturn:
    """
    Function to write values to a csv file. Help function to vector_management and distance_management.

    :param directory: Path to csv file.
    :param values: Values which we want to append.
    :return: NoReturn.
    """

    with open(directory, "a", newline='') as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(values)
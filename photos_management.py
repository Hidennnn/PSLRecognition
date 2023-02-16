from __future__ import annotations
import os
import csv
from typing import List, Any, NoReturn

from preprocessing import vector_of_points, distance
from custom_types import Vector


def load_photos(source: str, error_file_name: str, labels: dict, append_photos: bool = True) -> NoReturn:
    vector_name = "vector.csv"
    distance_name = "distance.csv"
    files_name = "files.csv"

    if not append_photos:
        init_csv([vector_name, distance_name, files_name])

    for folder in os.listdir(source):
        for file in os.listdir(f"{source}\\{folder}"):
            vector = vector_management(f"{source}\\{folder}\\{file}", file, vector_name,
                                       error_file_name, labels[file.split("_")[0]])

            distance_management(vector, distance_name, labels[file.split("_")[0]])
            write_csv(files_name, [file])


def init_csv(files: List[str]) -> NoReturn:
    columns_name = [str(x) for x in range(92)]
    columns_name.append("label")

    with open(files[0], "w", newline='') as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(columns_name)

    columns_name = [str(x) for x in range(1035)]
    columns_name.append("label")

    with open(files[1], "w", newline='') as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(columns_name)

    columns_name = ["File name"]
    with open(files[2], "w", newline='') as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(columns_name)


def write_csv(directory: str, values: List[Any]) -> NoReturn:
    with open(directory, "a", newline='') as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(values)


def vector_management(source: str, file: str, vector_name: str,
                      error_file_name: str, class_index: int) -> Vector | None:
    vector = vector_of_points(source)

    if vector is None:
        with open(error_file_name, "a") as error:
            error.write(f"{file}\n")
        return

    to_save_in_csv = []
    for point in vector:
        to_save_in_csv.append(point.x)
        to_save_in_csv.append(point.y)

    to_save_in_csv.append(class_index)

    write_csv(vector_name, to_save_in_csv)

    return vector


def distance_management(vector: Vector, distance_name: str, class_index: int) -> NoReturn:
    distance_values = distance(vector)

    to_save_in_csv = []
    for value in distance_values:
        to_save_in_csv.append(value)

    to_save_in_csv.append(class_index)

    write_csv(distance_name, to_save_in_csv)

from math import pi
import numpy as np


def detect_edge_using_ht(image: np.ndarray, radius_range: tuple):
    r_min, r_max = radius_range
    row, column = image.shape

    non_zero_edges = np.nonzero(image)

    accumulator_mat = np.zeros([2 * row, 2 * column, r_max + 1])
    for r in radius_range:
        for x, y in zip(*non_zero_edges):
            for theta in range(0, 360):
                a = x - r * np.cos(theta * pi / 180)
                b = y - r * np.sin(theta * pi / 180)
                accumulator_mat[int(a), int(b), int(r)] += 1

    norm_accumulator_mat = np.sum(accumulator_mat, axis=2)
    norm_accumulator_mat = ((norm_accumulator_mat - norm_accumulator_mat.min()) * (
            1.0 / (norm_accumulator_mat.max() - norm_accumulator_mat.min()) * 255))

    accumulator_image = norm_accumulator_mat[: row + 1, : column + 1]
    max_value = np.max(accumulator_mat)
    indices = np.argwhere(accumulator_mat > (max_value * 0.7))

    return indices, accumulator_image

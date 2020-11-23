from math import cos, sin, inf, sqrt, pi
import numpy as np
from PIL import Image
import cv2


# def detect_edge_using_ht(image: np.ndarray, radius_range: tuple):
#     circles = dict()
#     rows, columns = image.shape
#     image_edges = np.nonzero(image)
#     R_min, R_max = radius_range
#
#     acc_matrix = np.zeros((rows + 2 * R_max, columns + 2 * R_max, R_max - R_min + 1))
#     X, Y = np.meshgrid(range(2 * R_max), range(2 * R_max))
#     radii = np.round(np.sqrt((X - R_max) ** 2 + (Y - R_max) ** 2))
#     radii[radii < R_min] = 0
#     radii[radii > R_max] = 0
#     c_x, c_y = np.nonzero(radii)
#
#     for x, y in zip(*image_edges):
#         for a, b in zip(*(c_x, c_y)):
#             r = int(radii[a, b])
#             acc_matrix[a + x - 1, b + y - 1, r - R_min] += 1 / (2 * r * pi)
#
#     import scipy.ndimage.filters as filters
#
#     for i in range(R_max - R_min):
#         single_radius_accumulator = acc_matrix[:, :, i]
#         filtered_accumulator = single_radius_accumulator * (single_radius_accumulator == filters.maximum_filter(single_radius_accumulator, R_min))
#         filtered_accumulator[filtered_accumulator < 0.33] = 0
#         non_zero_accumulator = np.nonzero(filtered_accumulator)
#
#         for x, y in zip(*non_zero_accumulator):
#             c1 = (y - R_max, x - R_max, i + R_min)
#             min_d = inf
#             min_k = -1
#             for key, value in circles.items():
#                 for c2 in value:
#                     d = sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
#                     if d < min_d and d < R_min:
#                         min_d = d
#                         min_k = key
#             if min_k == -1:
#                 circles[len(circles)] = [c1]
#             else:
#                 circles[min_k].append(c1)
#
#     circles_list = []
#     for _, values in circles.items():
#         t_x, t_y, t_r = 0, 0, 0
#         for cir in values:
#             t_x += cir[0]
#             t_y += cir[1]
#             t_r += cir[2]
#         circles_list.append((t_x // len(values), t_y // len(values), t_r // len(values)))
#
#     acc_matrix = np.sum(acc_matrix, 2)
#     acc_matrix *= (255 // np.max(acc_matrix))
#
#     return circles_list, acc_matrix


def circular_detection(image, radius_range):
    '''
    :param image:  image containing circular objects
    :param radiusRange: range of diameters
    :return: 1. accumulator image; 2. input image with objects detected in given range of diameters.
    '''
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
    norm_accumulator_mat = ((norm_accumulator_mat - norm_accumulator_mat.min()) * (1.0 / (norm_accumulator_mat.max() - norm_accumulator_mat.min()) * 255))

    accumulator_image = norm_accumulator_mat[: row + 1, : column + 1]
    max_value = np.max(accumulator_mat)
    indices = np.argwhere(accumulator_mat > (max_value * 0.7))

    return indices, accumulator_image

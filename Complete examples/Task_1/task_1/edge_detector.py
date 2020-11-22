import math

import numpy as np


def apply_edge_detection(image: np.ndarray):
    prewitt_operator = np.asarray([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
    edge_magnitude, edge_orientation = get_edge_magnitude_and_orientation(image, prewitt_operator)
    print(edge_orientation.shape)
    print(edge_orientation)
    return edge_magnitude, edge_orientation


def get_edge_magnitude_and_orientation(f: np.ndarray, gradient_operator: np.ndarray) -> (np.ndarray, np.ndarray):
    m = np.zeros(f.shape)
    theta = np.zeros(f.shape)
    fx = np.zeros(f.shape)
    fy = np.zeros(f.shape)
    gx = gradient_operator
    gy = np.rot90(gradient_operator)
    start_position, end_position = (1, 1), (f.shape[0] - 1, f.shape[1] - 1)
    for i in range(start_position[0], end_position[0]):
        for j in range(start_position[1], end_position[1]):
            temp = f[i - 1:i + 2, j - 1:j + 2]
            fx[i, j] = np.sum(gx * temp) / 3
            fy[i, j] = np.sum(gy * temp) / 3
            m[i, j] = math.sqrt(fx[i, j] ** 2 + fy[i, j] ** 2)
            theta[i, j] = math.atan(fy[i, j] / fx[i, j] if fx[i, j] != 0. else math.inf)
    return m, theta


def apply_nms(m: np.ndarray, theta: np.ndarray):
    start_position, end_position = (1, 1), (m.shape[0] - 1, m.shape[1] - 1)
    output = np.zeros(m.shape)
    theta_degrees = theta * 180. / np.pi
    theta_degrees[theta_degrees < 0] += 180

    for i in range(start_position[0], end_position[0]):
        for j in range(start_position[1], end_position[1]):
            try:
                a = 255
                b = 255
                c = m[i, j]

                # for angle around 0
                if (0 <= theta_degrees[i, j] < 22.5) or (157.5 <= theta_degrees[i, j] <= 180):
                    a = m[i, j + 1]
                    b = m[i, j - 1]
                # for angle around 45
                elif 22.5 <= theta_degrees[i, j] < 67.5:
                    a = m[i + 1, j - 1]
                    b = m[i - 1, j + 1]
                # for angle around 90
                elif 67.5 <= theta_degrees[i, j] < 112.5:
                    a = m[i + 1, j]
                    b = m[i - 1, j]
                # for angle around 135
                elif 112.5 <= theta_degrees[i, j] < 157.5:
                    a = m[i - 1, j - 1]
                    b = m[i + 1, j + 1]

                if (c >= a) and (c >= b):
                    output[i, j] = c
                else:
                    output[i, j] = 0

            except IndexError:
                pass

    return output

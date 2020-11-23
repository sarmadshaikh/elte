import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from task_1.edge_detector import apply_edge_detection, apply_nms
from task_2.otsu_thresholding import mean, get_histogram, q_t, variance_between_classes, recursive_q1_t, \
    recursive_mu1_t, recursive_mu2_t
from task_3.hough_transform import circular_detection


def display_image(image, title=""):
    if isinstance(image, np.ndarray):
        Image.fromarray(image).show(title=title)
    elif isinstance(image, Image.Image):
        image.show(title=title)
    else:
        print("Image should be ndarray or PIL.Image object")


def read_image(image_path: str):
    image = Image.open(image_path)
    print(type(image))
    np_image = np.asarray(image)
    print(np_image.shape)
    return np_image


if __name__ == '__main__':
    import os.path as path
    images = ["blood.png", "cable.png", "cells.png", "circles.png"]
    for image in images:
        input_image = path.join(path.curdir, "test-data", "3_hough", image)
        np_image = read_image(input_image)

        # circles, acc_mat = detect_edge_using_ht(cv2.Canny(np_image, 100, 150), (20, 30))
        # for vertex in circles:
        #     cv2.circle(np_image, (vertex[0], vertex[1]), vertex[2], (255, 0, 0), 2, shift=0, lineType=8);
        #     cv2.rectangle(np_image, (vertex[0], vertex[1]), (vertex[0], vertex[1]), (0, 0, 255), 1)

        indices, accumulator_image = circular_detection(cv2.Canny(np_image, 100, 150), [13, 15])

        for x, y, r in indices:
            cv2.circle(np_image, (y, x), r, (255, 0, 0), thickness=1, lineType=8, shift=0)

        display_image(np_image, "Circle detection image")
        display_image(accumulator_image, "Accumulator image")
        output_image = Image.fromarray(np_image.astype(np.uint8))
        output_image.save("output/" + image[:-4] + "_transformed.jpg")

        accumulator_image = Image.fromarray(accumulator_image.astype(np.uint8))
        accumulator_image.save("output/" + image[:-4] + "_accumulator.jpg")

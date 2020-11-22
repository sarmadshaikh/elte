from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from task_2.otsu_thresholding import mean, get_histogram, variance_between_classes, recursive_q1_t, \
    recursive_mu1_t, recursive_mu2_t


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
    images = ["finger.png", "aluminium.png", "phobos.png", "julia.png"]
    for image in images:
        input_image = path.join(path.curdir, "test-data", image)
        np_image = read_image(input_image)

        img_histogram = get_histogram(np_image)
        img_mean = mean(img_histogram, 0, 0)
        print(img_mean)

        figure, axis = plt.subplots(1, 1, figsize=(30, 15))

        axis.plot([str(i) for i in range(256)], img_histogram)
        figure.suptitle(image)
        plt.show()

        variances_between_classes = np.zeros(254)
        for i in range(1, 255):
            q1_t = recursive_q1_t(img_histogram, i)
            mu1_t = recursive_mu1_t(img_histogram, i)
            mu2_t = recursive_mu2_t(img_histogram, i)
            variances_between_classes[i-1] = variance_between_classes(q1_t, mu1_t, mu2_t)
        max_var = variances_between_classes.max()
        # print("variances between classes:", variances_between_classes)
        # print("Max variance:", max_var)
        threshold = variances_between_classes.argmax() + 1
        print("Threshold at:", threshold)

        threshold_image = np.ndarray(np_image.shape)
        for i in range(np_image.shape[0]):
            for j in range(np_image.shape[1]):
                threshold_image[i, j] = 255 if np_image[i, j] >= threshold else 0

        display_image(threshold_image)

        output_threshold_image = Image.fromarray(threshold_image.astype(np.uint8))
        output_threshold_image.save("output/" + image[:-4] + "_thresholded.png")

from PIL import Image
import numpy as np

from task_1.edge_detector import apply_edge_detection, apply_nms


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
    images = ["julia.png", "motor.png", "circlegrey.png"]
    for image in images:
        input_image = path.join(path.curdir, image)
        np_image = read_image(input_image)

        edge_magnitude, edge_orientation = apply_edge_detection(np_image)
        nms_image = apply_nms(edge_magnitude, edge_orientation)

        display_image(np_image, "Input Image")
        display_image(edge_magnitude, "Image after edge detection")
        display_image(edge_orientation*255, "Edge orientation")
        display_image(nms_image, "Image after nms")

        output_edge_magnitude = Image.fromarray(edge_magnitude.astype(np.uint8))
        output_edge_orientation = Image.fromarray(edge_orientation.astype(np.uint8))
        output_image_after_nms = Image.fromarray(nms_image.astype(np.uint8))
        output_edge_magnitude.save("output/" + image[:-4] + "_edge_magnitude.jpg")
        output_edge_orientation.save("output/" + image[:-4] + "_edge_orientation.jpg")
        output_image_after_nms.save("output/" + image[:-4] + "_image_after_nms.jpg")

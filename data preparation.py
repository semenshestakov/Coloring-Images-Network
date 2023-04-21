import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image


def processed_image(img):
    if not type(img) == Image:
        img = Image.fromarray(np.squeeze(img))

    img = img.resize((224, 224))
    image = np.array(img, dtype=float)

    print(image.shape)
    size = image.shape
    lab = rgb2lab(image / 255.0)
    X, ab = lab[:, :, 0], lab[:, :, 1:]

    ab /= 128  # нормируем выходные значение в диапазон от -1 до 1
    X = X.reshape(1, size[0], size[1], 1)
    ab = ab.reshape(1, size[0], size[1], 2)
    return X, ab, size


def create_data_imagenet():
    ph = Image.open("i1.jpg")
    x, ab, s = processed_image(ph)

    lab_ab_in_rgb(x, ab)


def data_for_vgg():
    if not type(img) == Image:
        img = Image.fromarray(np.squeeze(img))

    img = img.resize((224, 224))
    image = np.array(img, dtype=float)


def lab_ab_in_rgb(lab: np.array, ab: np.array):
    """
    :param lab: grey
    :param ab: many
    :return: np.array shape = 1,224,224,3
    """

    image = np.zeros(shape=(224, 224, 3))
    image[:, :, 0] = np.clip(lab[0][:, :, 0], 0, 100)
    image[:, :, 1:] = np.clip(ab, -128, 127) * 128

    return np.array(lab2rgb(image))


if __name__ == '__main__':
    create_data_imagenet()

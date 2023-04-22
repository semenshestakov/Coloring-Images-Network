import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
from tensorflow.keras.applications import VGG19

vgg19_cl = VGG19()


def processed_image(img):
    image = conver_good_size(img)
    size = image.shape
    lab = rgb2lab(1.0 / 255 * image)
    X, ab = lab[:, :, 0], lab[:, :, 1:]

    ab /= 128
    X = X.reshape(1, size[0], size[1], 1)
    ab = ab.reshape(1, size[0], size[1], 2)
    return X, ab, np.array(image, dtype=int)


def create_data_imagenet():
    ph = Image.open("i1.jpg")
    x, ab, rgb = processed_image(ph)
    a = np.zeros((1, 224, 224, 3))
    a[0, :, :, 0] = x[0, :, :, 0]
    a[0, :, :, 1] = x[0, :, :, 0]
    a[0, :, :, 2] = x[0, :, :, 0]
    return x, ab, rgb


def conver_good_size(img):
    if not type(img) == Image:
        img = Image.fromarray(np.squeeze(img))

    img = img.resize((224, 224))
    image = np.array(img, dtype=float)
    return image


def avg_photo():
    ph = np.array(ph)
    ph = np.expand_dims(ph, axis=0)
    avg = np.sum(ph, axis=3) / 3
    avg.shape = (1, 224, 224, 1)
    return avg, ph


def data_for_vgg(img):
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
    # create_data_imagenet()
    avg_photo()

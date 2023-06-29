from PIL import Image
import pandas as pd
import numpy as np
from skimage.color import rgb2lab, lab2rgb


def conver_good_size(img):
    if type(img) != Image:
        img = Image.fromarray(np.squeeze(img))

    img = img.resize((224, 224))
    image = np.array(img, dtype=float)
    return image


def processed_image(img):
    image = conver_good_size(img)
    size = image.shape
    lab = rgb2lab(1.0 / 255 * image)
    X, ab = lab[:, :, 0], lab[:, :, 1:]

    ab /= 128
    X = X.reshape(1, size[0], size[1], 1)
    ab = ab.reshape(1, size[0], size[1], 2)
    return X, ab, np.array(image, dtype=int) / 255.0


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


class DataSet:

    def __init__(self, path="data", stat="train"):
        self.path = path
        dinfo = pd.read_csv(f"{path}/info.csv").sample(frac=1)
        self.dinfo = dinfo[dinfo["data set"] == stat]

    def get_data(self):
        size = len(self.dinfo.filepaths)
        n = 0
        data_img = np.zeros((size, 224, 224, 3), dtype=np.float32)
        x_data = np.zeros((size, 224, 224, 1), dtype=np.float32)
        y_cl_data = np.zeros((size, 100), dtype=np.float32)
        y_ab_data = np.zeros((size, 224, 224, 2), dtype=np.float32)

        for id_cl, fl, *_ in self.dinfo.values:
            ph = Image.open(f"{self.path}/{fl}")
            x_data[n], y_ab_data[n], data_img[n] = processed_image(ph)
            y_cl_data[n][id_cl] = 1
            n += 1

        return x_data[:n], y_ab_data[:n], y_cl_data[:n], data_img[:n]


    def gen_data(self, batch=64):
        size = len(self.dinfo.filepaths)
        n = 0
        data_img = np.zeros((batch, 224, 224, 3), dtype=np.float32)
        x_data = np.zeros((batch, 224, 224, 1), dtype=np.float32)
        y_cl_data = np.zeros((batch, 100), dtype=np.float32)
        y_ab_data = np.zeros((batch, 224, 224, 2), dtype=np.float32)

        for id_cl, fl, *_ in self.dinfo.values:
            ph = Image.open(f"{self.path}/{fl}")
            if n >= batch:
                yield x_data, y_ab_data, y_cl_data, data_img
                data_img = np.zeros((batch, 224, 224, 3), dtype=np.float32)
                y_cl_data = np.zeros((batch, 100), dtype=np.float32)

                n = 0

            x_data[n], y_ab_data[n], data_img[n] = processed_image(ph)
            y_cl_data[n][id_cl] = 1
            n += 1

        yield x_data[:n], y_ab_data[:n], y_cl_data[:n], data_img[:n]

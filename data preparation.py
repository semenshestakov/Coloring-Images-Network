import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras.datasets import image




def processed_image(img):
    image = np.array(img, dtype=float)
    print(image.shape)
    # assert image.shape == ()
    size = image.shape
    lab = rgb2lab(1.0 / 255 * image)
    X, ab = lab[:, :, 0], lab[:, :, 1:]

    ab /= 128  # нормируем выходные значение в диапазон от -1 до 1
    X = X.reshape(1, size[0], size[1], 1)
    ab = ab.reshape(1, size[0], size[1], 2)
    return X, ab, size


def create_data_imagenet():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape)
    plt.imshow(x_train[0])
    plt.show()


if __name__ == '__main__':
    create_data_imagenet()

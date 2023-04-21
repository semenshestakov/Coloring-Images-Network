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
    lab = rgb2lab(1.0 / 255 * image)
    X, ab = lab[:, :, 0], lab[:, :, 1:]

    ab /= 128  # нормируем выходные значение в диапазон от -1 до 1
    X = X.reshape(1, size[0], size[1], 1)
    ab = ab.reshape(1, size[0], size[1], 2)
    return X, ab, size


def create_data_imagenet():
    ph = Image.open("i1.jpg")
    x,ab,s = processed_image(ph)
    plt.imshow(np.squeeze(x))
    plt.show()
    print(np.max(x))







if __name__ == '__main__':
    create_data_imagenet()

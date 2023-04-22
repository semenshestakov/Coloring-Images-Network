import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
from tensorflow.keras.applications import VGG19

vgg19_cl = VGG19()
def processed_image(img):
    if not type(img) == Image:
        img = Image.fromarray(np.squeeze(img))

    img = img.resize((224, 224))
    image = np.array(img, dtype=float)

    # print(image.shape)
    size = image.shape
    lab = rgb2lab(image / 255.0)
    X, ab = lab[:, :, 0], lab[:, :, 1:]

    ab /= 128  # нормируем выходные значение в диапазон от -1 до 1
    X = X.reshape(1, size[0], size[1], 1)
    ab = ab.reshape(1, size[0], size[1], 2)
    return X, ab, image


def create_data_imagenet():
    ph = Image.open("i1.jpg")
    x, ab, rgb = processed_image(ph)
    a = np.zeros((1,224,224,3))
    a[0,:,:,0] = x[0,:,:,0]
    a[0,:,:,1] = x[0,:,:,0]
    a[0,:,:,2] = x[0,:,:,0]
    return x,ab,rgb
    # i = np.array(a[0],dtype=int)
    #
    # # print(np.max(i), np.min(i), i.shape)
    # plt.imshow(i, cmap='gist_gray')
    # plt.show()
    # a = np.clip(a,0,255)
    # res = vgg19_cl(a)
    # arg = np.argmax(res)
    # print(arg ,res[0,arg])


def avg_photo():
    ph = Image.open("i1.jpg").resize((224,224))
    ph = np.array(ph)
    ph = np.expand_dims(ph,axis=0)
    avg = np.sum(ph,axis=3) / 3
    # avg.shape = (224,224)
    # zer = np.zeros(shape=(1,224,224,3))
    # zer[0,:,:,0] = avg
    # zer[0,:,:,1] = avg
    # zer[0,:,:,2] = avg
    # # print(avg.shape)
    # print(np.argmax(vgg19_cl(zer)))
    # plt.imshow(avg[0])
    # plt.show()
    avg.shape = (1,224,224,1)
    return avg,ph
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


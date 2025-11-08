import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import special
import math
from PIL import Image as im
import TOMOGRAPHY_FUNCTION as tf
import numba
# from google.colab import files

# 画像書き出し
def img_write(fname, img):
    img = img.astype(np.uint8)
    cv2.imwrite(fname, img)
    return 0

# 画像読み込み
def img_read(fname):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    return img

field = np.load("acoustic_vortex_field.npy")

field_obs = field[127]

vortex_fft = np.fft.fftshift(np.fft.fft2(field_obs))

plt.subplot(131,title="Vortex Intensity");plt.imshow(np.abs(field_obs))
plt.subplot(132,title="Vortex Phase");plt.imshow(np.angle(field_obs),"hsv")
plt.subplot(133,title="Vortex FFT");plt.imshow(np.log(np.abs(vortex_fft) + 1),"gray")
plt.show()
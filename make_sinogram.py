# Sinogram 生成用
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

# 正規化(実数のみ)
def normalize(val):
    f = (val-np.amin(val))/(np.amax(val)-np.amin(val))*255
    return f

# 正規化(複素数対応)
def normalize_complex(complex_array: np.ndarray):
    magnitude = np.abs(complex_array)
    min_val = np.amin(magnitude)
    max_val = np.amax(magnitude)

    if max_val == min_val:
        return np.zeros_like(magnitude, dtype=float)
        
    normalized_magnitude = (magnitude - min_val) / (max_val - min_val) * 255.0
    return normalized_magnitude

# 疑似係数行列による投影（ラドン変換）
@numba.jit(nopython=True)
def Radon_Cij(Img,Cij_x, Cij_0, Cij_1, Cij_2):
   Y_size, X_size = Img.shape
   num_angles, num_y, num_x = Cij_x.shape
   Sinogram = np.zeros((num_angles, X_size), dtype=Img.dtype)
   for k in range(num_angles):
      for i in range (Y_size):
         for j in range (X_size):
            x = int(Cij_x[k,i,j])
            if (x>=1 and x<X_size-1):
               # Sinogram.flat[x-1] += Cij_0[k,i,j] * Img[i,j]
               # Sinogram.flat[x] += Cij_1[k,i,j] * Img[i,j]
               # Sinogram.flat[x+1] += Cij_2[k,i,j] * Img[i,j]
               Sinogram[k,x-1] += Cij_0[k,i,j] * Img[i,j]
               Sinogram[k,x] += Cij_1[k,i,j] * Img[i,j]
               Sinogram[k,x+1] += Cij_2[k,i,j] * Img[i,j]
   return Sinogram

# 生成済みの係数行列の読み込み
# 180どまで10step
cij_x = np.load(r"Cij/cij_x_2.npy")
cij_0 = np.load(r"Cij/cij_0_2.npy")
cij_1 = np.load(r"Cij/cij_1_2.npy")
cij_2 = np.load(r"Cij/cij_2_2.npy")

# --- 投影データの作製 (実験データで置き換える) ---
field = np.load("acoustic_vortex_field.npy")
print(field.dtype)
z, y, x = field.shape
num_angle,y_cij,x_cij = cij_x.shape
sinogram = np.zeros((z, num_angle, x), dtype=field.dtype)
for i in range(z):
   img = field[i]
   sinogram[i] = Radon_Cij(img, cij_x, cij_0, cij_1, cij_2)
   if (i%25 == 0):
      print(f"iteration:{i}")

max_amplitude = np.amax(np.abs(sinogram))
sinogram_normalized = (sinogram / max_amplitude) * 255.0
np.save("sinogram_vortex_2.npy", sinogram_normalized)
print(sinogram_normalized.dtype)
# img_write("image/sinogram_20.png",normalize(sinogram))
# plt.imshow(np.abs(sinogram[0]),"gray")
# plt.show()
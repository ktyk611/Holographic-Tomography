import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import special
import math
from PIL import Image as im
import numba
# from google.colab import files

# --- 投影データを計算するための関数を作りたい ---

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

# 投影角度の配列生成
def Projection_Angle(scan_start, scan_stop, scan_step):
   angle = np.linspace(scan_start, scan_stop, scan_step)
   return angle

# 疑似係数行列の生成
# @numba.jit(nopython=True)
def Make_Cij(Y_size, X_size, Angle):
   L = len(Angle)
   Cij_x = np.zeros((L, Y_size, X_size))
   Cij_0 = np.zeros((L, Y_size, X_size))
   Cij_1 = np.zeros((L, Y_size, X_size))
   Cij_2 = np.zeros((L, Y_size, X_size))

   for k in range (L):
      Theta = Angle[k] * 2 * np.pi / 360
      Sin = np.sin(Theta)
      Cos = np.cos(Theta)
      # Sin と Cos のうち大きいほうを a 小さいほうを b
      if (np.abs(Sin) > np.abs(Cos)):
         a = np.abs(Sin)
         b = np.abs(Cos)
      else:
         a = np.abs(Cos)
         b = np.abs(Sin)

      # 0除算を避ける
      if a == 0: a = 1e-6
    
      # 順投影，逆投影を得る画像(x,y)にしたがい各検出確率を求める．
      for i in range (Y_size):
         y = (Y_size/2 - i)
         for j in range (X_size):
            x = (j - X_size/2)
            x0 = x * Cos + y * Sin
            # (x,y)の中心(正方形の中心)が検出される座標を求める
            ix = int(np.floor(x0 + 0.5))
            # 検出器の座標を保存
            Cij_x[k, i, j] = ix + X_size / 2

            if (ix + X_size/2 > 1 and ix + X_size/2 < X_size -2):
               # 検出器s(i-1)に入射する割合
               x05 = (ix - 0.5)
               if ((x05-(x0-(a-b)/2)) > 0):
                  d = x05-(x0-(a-b)/2)
                  Cij_0[k, i, j] = b/(2*a) + d/a
               elif ((x05-(x0-(a+b)/2)) > 0):
                  d = x05-(x0-(a+b)/2)
                  Cij_0[k, i, j] = d*d / (2*a*b) if (a*b > 0) else 0
               
               x05 = (ix + 0.5)
               if ((x0 + (a-b)/2 - x05) > 0):
                  d = x0 + (a-b)/2 - x05
                  Cij_2[k, i, j] = b/(2*a) + d/a
               elif ((x0 + (a+b)/2 - x05) > 0):
                  d = x0 + (a+b)/2 - x05
                  Cij_2[k, i, j] = d*d / (2*a*b) if (a*b > 0) else 0
                
               Cij_1[k, i, j] = (1.0 - Cij_0[k, i, j] - Cij_2[k, i, j])
               Cij_0[k, i, j] = np.clip(Cij_0[k, i, j], 0, 1)
               Cij_1[k, i, j] = np.clip(Cij_1[k, i, j], 0, 1)
               Cij_2[k, i, j] = np.clip(Cij_2[k, i, j], 0, 1)

               total_w = Cij_0[k, i, j] + Cij_1[k, i, j] + Cij_2[k, i, j]
               if total_w > 0:
                  Cij_0[k, i, j] /= total_w
                  Cij_1[k, i, j] /= total_w
                  Cij_2[k, i, j] /= total_w

   return Cij_x, Cij_0, Cij_1, Cij_2

# 疑似係数行列による投影（ラドン変換）
@numba.jit(nopython=True,cache=True)
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

# 疑似係数行列による逆投影
@numba.jit(nopython=True)
def Back_Projection_Cij(Sinogram, Cij_x, Cij_0, Cij_1, Cij_2):
   num_angles,Y_size, X_size = Cij_x.shape
   Recon_Img = np.zeros((Y_size, X_size), dtype=Sinogram.dtype)
   for i in range (Y_size):
      for j in range(X_size):
         sum_val = 0
         for k in range (num_angles):
            x = int(Cij_x[k,i,j])
            if (x>=1 and x<X_size-1):
               sum_val += Sinogram[k,x-1] * Cij_0[k,i,j] + Sinogram[k,x] * Cij_1[k,i,j] + Sinogram[k,x+1] * Cij_2[k,i,j]
         Recon_Img[i,j] = sum_val
   return Recon_Img

# MSE(Mean Squared Error)差分２乗平均
def MSE(original_image, target_image):
   y_size, x_size = original_image.shape
   N = y_size * x_size
   mean_squared_error = np.sum((original_image - target_image)**2)/N
   return mean_squared_error

# PSNR(Peak Signal to Noise Ratio)
def PSNR(mse):
   psnr = 10*np.log10(255**2 / mse)
   return psnr

# 投影データ(Sinogram)の作成
def Radon_rotate(img, scan_stop, sacn_step):
  y_size, x_size = img.shape
  center = (int(y_size/2),int(x_size/2))
  scan_start = 0
  angle = np.linspace(scan_start, scan_stop, sacn_step)
  scale = 1
  sinogram = np.zeros((sacn_step, y_size))
  for i in range(sacn_step):
    # 回転させる（画像と東映の回転の向きは逆）
    retval= cv2.getRotationMatrix2D(center, -angle[i], scale)
    im_rotate = cv2.warpAffine(img,retval,(x_size,y_size))
    # 水平方向に投影(走査)
    for j in range(y_size):
      for k in range(x_size):
        sinogram[i,j] += im_rotate[j,k]
  return sinogram

# Sinogram からの再構成
def Back_Projection_Rotate(sinogram, scan_stop):
  scan_step, x_size = sinogram.shape
  recon_img = np.zeros((x_size, x_size))
  angle = np.linspace(0, scan_stop, scan_step)
  center = (int(x_size/2),int(x_size/2))
  scale = 1

  for i in range(scan_step):
    # dataの引き延ばし
    data_2d = np.tile(sinogram[i], (x_size, 1))
    # 重ね合わせるときの角度を設定（向き合わせのための-90度）
    retval= cv2.getRotationMatrix2D(center, angle[i]-90, scale)
    data_rotate = cv2.warpAffine(data_2d,retval,(x_size, x_size))
    # 重ね合わせ
    recon_img += data_rotate

  return recon_img

# Sinogram を周波数空間でのフィルタリング
def Filtering_Sinogram(sinogram):
  scan_step, x_size = sinogram.shape

  # 周波数空間でのフィルタ生成
  filter = np.zeros((x_size),dtype="complex128")
  for i in range(x_size):
    filter[i] = np.abs(i - x_size/2)

  # 周波数空間でのフィルタリング
  filtered = np.zeros((scan_step,x_size),dtype="complex128")
  for i in range(scan_step):
    sinogram_fft = np.fft.fft(sinogram[i], x_size)
    sinogram_fft_shift = np.fft.fftshift(sinogram_fft)

    filtered_fft = filter * sinogram_fft_shift
    filtered[i] = np.fft.ifft(np.fft.fftshift(filtered_fft), x_size)

  return filtered

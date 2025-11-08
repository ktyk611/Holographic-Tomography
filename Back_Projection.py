import numpy as np
import matplotlib.pyplot as plt
import cv2
import numba

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

# A^Tの代わり (逆投影の関数)
# 疑似係数行列による逆投影
@numba.jit(nopython=True, cache=True)
def Back_Projection_Cij(Sinogram, Cij_x, Cij_0, Cij_1, Cij_2):
   num_angles,Y_size, X_size = Cij_x.shape
   Recon_Img = np.zeros((Y_size, X_size), dtype=Sinogram.dtype) # dtype 指定はOK
   for i in range (Y_size):
      for j in range(X_size):
        
        # Numba に Sinogram の型 (float または complex) を推論させる
        sum_val_init = Sinogram[0, 0] * 0.0
        
        for k in range (num_angles):
            x = int(Cij_x[k,i,j])
            if (x>=1 and x<X_size-1):
               sum_val_init += Sinogram[k,x-1] * Cij_0[k,i,j] + Sinogram[k,x] * Cij_1[k,i,j] + Sinogram[k,x+1] * Cij_2[k,i,j]
        Recon_Img[i,j] = sum_val_init
   return Recon_Img

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

cij_x = np.load(r"Cij/cij_x_2.npy")
cij_0 = np.load(r"Cij/cij_0_2.npy")
cij_1 = np.load(r"Cij/cij_1_2.npy")
cij_2 = np.load(r"Cij/cij_2_2.npy")
sinogram = np.load(r"sinogram/sinogram_unwrap.npy")
layer, angle, x = sinogram.shape
# sinogram1 = Filtering_Sinogram(sinogram[int(layer/2)])
# sinogram2 = Filtering_Sinogram(sinogram[int(layer-1)])
recon = np.zeros((layer,x,x))
for k in range (layer):
   recon[k] = Back_Projection_Cij(sinogram[k],cij_x,cij_0,cij_1,cij_2)
   if (k%25==0):
      print(f"iteration:{k}")

# np.save("result/2025_1031/BP/Back_Projection_0,90.npy",recon)

plt.subplot(221,title=f"intensity,{0.15/128*(layer/2)*100:.1f}cm");plt.imshow(np.abs(recon[int(layer/2)]),"hsv");plt.colorbar(label='Phase')
plt.subplot(222,title=f"phase, {0.15/128*(layer/2)*100:.1f}cm");plt.imshow(np.angle(recon[int(layer/2)]),"hsv");plt.colorbar(label='Phase')
plt.subplot(223,title=f"intensity, {0.15/128*(layer-1)*100:.1f}cm");plt.imshow(np.abs(recon[int(layer-1)]),"hsv");plt.colorbar(label='Phase')
plt.subplot(224,title=f"phase, {0.15/128*(layer-1)*100:.1f}cm");plt.imshow(np.angle(recon[int(layer-1)]),"hsv");plt.colorbar(label='Phase')
plt.suptitle(f"Filtered Back Projection \n projection angle:0,90")
plt.show()
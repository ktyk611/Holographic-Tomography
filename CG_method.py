import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import special
import numba

# Aの代わり(順投影の関数)
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

# 生成済みの係数行列の読み込み
# 180どまで60step
cij_x = np.load(r"Cij/cij_x_3.npy")
cij_0 = np.load(r"Cij/cij_0_3.npy")
cij_1 = np.load(r"Cij/cij_1_3.npy")
cij_2 = np.load(r"Cij/cij_2_3.npy")

sinogram = np.load(r"sinogram/sinogram_unwrap.npy")
layer, angle, x = sinogram.shape

# --- 共役勾配法(係数行列) Conjugate Gradient Method ---
# 共役勾配法（反復法）の繰り返し回数
N = 4
# g, d は再構成画像
img_y, img_x = layer, x
rec = np.zeros((layer, img_y, img_x), dtype="complex128")
f_history = np.zeros((layer,N))
for k in range(layer):
   g = Back_Projection_Cij(sinogram[127], cij_x,cij_0,cij_1,cij_2)
   d = g.copy()
   img_rec = np.zeros((img_y, img_x), dtype=g.dtype)

   for i in range(N):
      # 反復計算 (逐次近似画像再構成 p119)
      Cd = Radon_Cij(d, cij_x, cij_0, cij_1, cij_2)
      alp = np.dot(g.conj().flatten(), g.flatten()) / np.dot(Cd.conj().flatten(), Cd.flatten())
      f_ne = img_rec + alp * d
      g_ne = g - alp * Back_Projection_Cij(Cd, cij_x,cij_0,cij_1,cij_2)
      beta = np.dot(g_ne.conj().flatten(), g_ne.flatten()) / np.dot(g.conj().flatten(), g.flatten())
      d_ne = g_ne + beta * d
      # パラメータの更新
      g = g_ne
      d = d_ne
      img_rec = f_ne
      f_history[k,i] = np.abs(np.sum(f_ne - d)/(img_y*img_x))
   if (k%25 == 0):
      print(f"iteration:{k}")

   rec[k] = img_rec

# np.save("CG_method_recon_0,90.npy",rec)
# rec = np.load("result/2025_1031/CG/CG_method_recon_0,90.npy")
plt.subplot(111,title=f"z:{0.15/1*60*100:.1f}cm");plt.plot(np.log10(f_history[int(layer/2)] + 1));plt.xlabel("iteration(N)");plt.ylabel("log|A(x)-b|")
plt.show()
# 再構成画像の表示
plt.subplot(221,title=f"intensity,{0.15/128*(layer/2)*100:.1f}cm ");plt.imshow(np.abs(rec[int(layer/2)]));plt.colorbar(label='Amplitude')
plt.subplot(222,title=f"phase,{0.15/128*(layer/2)*100:.1f}cm");plt.imshow(np.angle(rec[int(layer/2)]),"hsv");plt.colorbar(label='Phase')
plt.subplot(223,title=f"intensity, {0.15/128*(layer-1)*100:.1f}cm ");plt.imshow(np.abs(rec[int(layer-1)]));plt.colorbar(label='Amplitude')
plt.subplot(224,title=f"phase, {0.15/128*(layer-1)*100:.1f}cm");plt.imshow(np.angle(rec[int(layer-1)]),"hsv");plt.colorbar(label='Phase')
plt.suptitle(f"CG method iteration:{N} \n projection angle:0, 90")
plt.show()
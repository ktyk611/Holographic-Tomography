import numpy as np
import numba

# 疑似係数行列の生成
@numba.jit(nopython=True, cache=True)
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
               Cij_0[k, i, j] = max(0.0, min(Cij_0[k, i, j], 1.0))
               Cij_1[k, i, j] = max(0.0, min(Cij_1[k, i, j], 1.0))
               Cij_2[k, i, j] = max(0.0, min(Cij_2[k, i, j], 1.0))

               total_w = Cij_0[k, i, j] + Cij_1[k, i, j] + Cij_2[k, i, j]
               if total_w > 0:
                  Cij_0[k, i, j] /= total_w
                  Cij_1[k, i, j] /= total_w
                  Cij_2[k, i, j] /= total_w

   return Cij_x, Cij_0, Cij_1, Cij_2

# 投影角度の配列生成
def Projection_Angle(scan_start, scan_stop, scan_step):
   angle = np.linspace(scan_start, scan_stop, scan_step)
   return angle

angle = Projection_Angle(0, 60, 2)
x_size = 128
y_size = 128
print(angle)
cij_x, cij_0, cij_1, cij_2 = Make_Cij(y_size, x_size,angle)
np.save(r"Cij/cij_x_2.npy",cij_x)
np.save(r"Cij/cij_0_2.npy",cij_0)
np.save(r"Cij/cij_1_2.npy",cij_1)
np.save(r"Cij/cij_2_2.npy",cij_2)
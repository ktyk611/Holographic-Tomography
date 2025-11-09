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

# L (Linear operation)
def Linear_operation(P:np.ndarray, Q:np.ndarray):
    #nppad(p, ((left,right), (ue, shita))) 左右に1ずつパディング，上下はパディングしない
    P_virtual = np.pad(P, ((1, 1), (0, 0)))
    Q_virtual = np.pad(Q, ((0, 0), (1, 1)))
    P_term = P_virtual[1:, :] - P_virtual[:-1, :]
    Q_term = Q_virtual[:, 1:] - Q_virtual[:, :-1]
    L_pq = P_term + Q_term
    return L_pq

@numba.jit(nopython=True, cache=True) # Numbaデコレータを追加
def Linear_operation_numba(P:np.ndarray, Q:np.ndarray):
    # P と Q の形状を取得
    # P は (m-1, n), Q は (m, n-1)
    m_p, n_p = P.shape
    m_q, n_q = Q.shape

    # 最終的な出力 L_pq の形状は (m, n)
    # m = m_p + 1 = m_q
    # n = n_p = n_q + 1
    m, n = m_q, n_p

    # --- P_virtual (Pのパディング) を手動で作成 ---
    # np.pad(P, ((1, 1), (0, 0))) と同等
    # P_virtual は (m+1, n) の形状になる
    P_virtual = np.zeros((m + 1, n), dtype=P.dtype)
    P_virtual[1:m, :] = P # P_virtual[1:-1, :] と同じ

    # --- Q_virtual (Qのパディング) を手動で作成 ---
    # np.pad(Q, ((0, 0), (1, 1))) と同等
    # Q_virtual は (m, n+1) の形状になる
    Q_virtual = np.zeros((m, n + 1), dtype=Q.dtype)
    Q_virtual[:, 1:n] = Q # Q_virtual[:, 1:-1] と同じ

    # --- 元の計算を実行 ---
    # これらはNumbaがサポートするスライス操作
    P_term = P_virtual[1:, :] - P_virtual[:-1, :]
    Q_term = Q_virtual[:, 1:] - Q_virtual[:, :-1]

    L_pq = P_term + Q_term
    return L_pq

# L_Transposition
@numba.jit(nopython=True,cache=True)
def L_Transposition(X:np.ndarray):
    m,n = X.shape
    p, q = np.zeros((m-1,n), dtype=X.dtype), np.zeros((m,n-1), dtype=X.dtype)
    p = X[:-1, :] - X[1:, :]
    q = X[:, :-1] - X[:, 1:]
    return p, q

# Pc[] (C = B_l,u : l=0, u=1)
@numba.jit(nopython=True,cache=True)
def Projection_operation(X:np.ndarray):
    l, u = 0.0, 1.0
    X_clip = np.clip(X,l,u)
    return X_clip

# P_p 異方性(Anisotropic)Tvの場合
@numba.jit(nopython=True,cache=True)
def Projection_P_anisotropic(P, Q):
    # Remark 4.2 の P_P1 への射影
    denom_P = np.maximum(1.0, np.abs(P))
    denom_Q = np.maximum(1.0, np.abs(Q))
    return P / denom_P, Q / denom_Q

# p_p 等方性(Isotropic)の場合
@numba.jit(nopython=True, cache=True)
def Projection_P_isotropic(P, Q):
    R = np.copy(P)
    S = np.copy(Q)
    P_common = P[:, :-1]
    Q_common = Q[:-1, :]
    denom_sqrt = np.sqrt(np.abs(P_common)**2 + np.abs(Q_common)**2)
    denom = np.maximum(1.0, denom_sqrt)
    R[:, :-1] = P_common / denom
    S[:-1, :] = Q_common / denom
    P_border = P[:, -1]
    denom_P = np.maximum(1.0, np.abs(P_border))
    R[:, -1] = P_border / denom_P
    Q_border = Q[-1, :]
    denom_Q = np.maximum(1.0, np.abs(Q_border))
    S[-1, :] = Q_border / denom_Q
    return R, S

# FGP_Argorith
def FGP_Argorithm(img_noise:np.ndarray, lam:float, N_ite:int, TV_func: str):
    b = img_noise
    m,n = b.shape
    t = 1
    p, q = np.zeros((m-1,n)), np.zeros((m,n-1))
    r, s = np.zeros((m-1,n)), np.zeros((m,n-1))
    for k in range (1, N_ite+1):
        L = Linear_operation(r,s)
        Pc = Projection_operation(b - lam*L)
        r_n, s_n = L_Transposition(Pc)
        r_n /=  (8*lam)
        s_n /=  (8*lam)

        if TV_func == "anisotropic":
            p_n, q_n = Projection_P_anisotropic(r_n + r, s_n + s)
        elif TV_func == "isotropic":
            p_n, q_n = Projection_P_isotropic(r_n + r, s_n + s)
        else:
            # Numbaはエラーを投げられないので、デフォルトの動作（例：anisotropic）を定義
            p_n, q_n = Projection_P_isotropic(r_n + r, s_n + s)

        t_n = (1 + np.sqrt(1 + 4*t*t))/2
        r_n = p_n + (t-1)/t_n * (p_n - p)
        s_n = q_n + (t-1)/t_n * (q_n - q)

        #パラメータの更新
        r, s = r_n, s_n
        p, q = p_n, q_n
        t = t_n

    L_p_N = Linear_operation(p, q)
    img_denoised = Projection_operation(b - lam * L_p_N)
    return img_denoised

# FGP_Argorith Complex対応
@numba.jit(nopython=True, cache=True)
def FGP_Argorithm_complex(img_noise:np.ndarray, lam:float, N_ite:int, TV_func: str):

    b = img_noise
    m,n = b.shape
    t = 1.0 # t は実数

    # --- 修正点: b.dtype を使って複素数型で初期化 ---
    p, q = np.zeros((m-1,n), dtype=b.dtype), np.zeros((m,n-1), dtype=b.dtype)
    r, s = np.zeros((m-1,n), dtype=b.dtype), np.zeros((m,n-1), dtype=b.dtype)

    for k in range (1, N_ite+1):
        L = Linear_operation_numba(r,s)

        # --- 修正点: 実数への射影 (Projection_operation) を削除 ---
        Pc = b - lam*L

        r_n, s_n = L_Transposition(Pc)
        r_n /=  (8*lam)
        s_n /=  (8*lam)

        if TV_func == "anisotropic":
            p_n, q_n = Projection_P_anisotropic(r_n + r, s_n + s)
        elif TV_func == "isotropic":
            p_n, q_n = Projection_P_isotropic(r_n + r, s_n + s)
        else:
            # Numbaはエラーを投げられないので、デフォルトの動作（例：anisotropic）を定義
            p_n, q_n = Projection_P_isotropic(r_n + r, s_n + s)

        t_n = (1.0 + np.sqrt(1.0 + 4.0*t*t))/2.0
        r_n = p_n + (t-1.0)/t_n * (p_n - p)
        s_n = q_n + (t-1.0)/t_n * (q_n - q)

        #パラメータの更新
        r, s = r_n, s_n
        p, q = p_n, q_n
        t = t_n

    L_p_N = Linear_operation_numba(p, q)

    # --- 修正点: 実数への射影 (Projection_operation) を削除 ---
    img_denoised = b - lam * L_p_N
    return img_denoised

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

# PL(y)の計算
@numba.jit(nopython=True, cache=True)
def pL_y_radon(y, b, lam, L, N, Tv_func,Cij_x, Cij_0, Cij_1, Cij_2):
   grad_f_y = 2 * Back_Projection_Cij(Radon_Cij(y, Cij_x, Cij_0, Cij_1, Cij_2) - b, Cij_x, Cij_0, Cij_1, Cij_2)
   Y_denoise = y - (1.0 / L) * grad_f_y
   lam_new = (2.0 * lam) / L
   z = FGP_Argorithm_complex(Y_denoise, lam_new, N, Tv_func)
   return z

# g(x) = 2*lambda*TV(x) のTV項を計算
@numba.jit(nopython=True,cache=True)
def TV_anisotropic(x):
    p, q = L_Transposition(x)
    return np.sum(np.abs(p)) + np.sum(np.abs(q))

@numba.jit(nopython=True, cache=True)
def TV_isotropic(x: np.ndarray):
    p, q = L_Transposition(x)

    # --- 1. 共通部分 (論文の最初の総和項) ---
    # i=1..m-1, j=1..n-1 の領域
    p_common = p[:, :-1]
    q_common = q[:-1, :]

    # 複素数の場合を考慮し、L2ノルムを |p|^2 + |q|^2 で計算
    common_term_sum = np.sum(np.sqrt(np.abs(p_common)**2 + np.abs(q_common)**2))

    # --- 2. P の境界部分 (論文の2番目の総和項) ---
    # i=1..m-1, j=n の領域 (p の最後の列)
    p_border = p[:, -1]
    p_border_sum = np.sum(np.abs(p_border))

    # --- 3. Q の境界部分 (論文の3番目の総和項) ---
    # i=m, j=1..n-1 の領域 (q の最後の行)
    q_border = q[-1, :]
    q_border_sum = np.sum(np.abs(q_border))

    # 3つの項の合計
    return common_term_sum + p_border_sum + q_border_sum

# 目的関数 F(x) = f(x) + g(x) の値を計算
@numba.jit(nopython=True,cache=True)
def F_tomography(x, b, lam, TV_func: str, Cij_x, Cij_0, Cij_1, Cij_2):
    # f(x) = ||A(x) - b||^2
    Ax_b = Radon_Cij(x, Cij_x, Cij_0, Cij_1, Cij_2) - b

    # --- 修正点: 複素数のL2ノルム（絶対値の2乗和）に変更 ---
    f_x = np.sum(np.abs(Ax_b)**2)


    if TV_func == "anisotropic":
      val = TV_anisotropic(x)
    elif TV_func == "isotropic":
      val = TV_isotropic(x)
    else:
      # Numbaはエラーを投げられないので、デフォルトの動作（例：anisotropic）を定義
      val = TV_isotropic(x)

    # g(x) = 2 * lam * TV(x)
    g_x = 2.0 * lam * val
    return f_x + g_x

# MFISTA トモグラフィ画像再構成 (最適化版)
def MFISTA_tomography(b:np.ndarray, lam, N_outer, N_inner, Tv_func: str,L, Cij_x, Cij_0, Cij_1, Cij_2):
  m, n = b.shape
  L_f = L * m

  x_k = np.zeros((n,n), dtype=b.dtype)
  x_prev = np.zeros((n,n), dtype=b.dtype)
  y_k = np.zeros((n,n), dtype=b.dtype)
  t_k = 1.0

  F_history = np.zeros((N_outer))
  x_history = []
  # --- 変更点 1: F(x_0) をループ前に計算 ---
  F_x_prev = F_tomography(x_prev, b, lam, Tv_func, Cij_x, Cij_0, Cij_1, Cij_2)

  print(f"Starting MFISTA. F(x_0) = {F_x_prev:.4e}")

  for k in range(1, N_outer + 1):
    z_k = pL_y_radon(y_k, b, lam, L_f, N_inner, Tv_func,Cij_x, Cij_0, Cij_1, Cij_2)

    t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0

    F_z_k = F_tomography(z_k, b, lam, Tv_func, Cij_x, Cij_0, Cij_1, Cij_2)
    # --- 変更点 2: F_x_prev の再計算を削除 ---
    # F_x_prev = F_tomography(x_prev, b, lam, Cij_x, Cij_0, Cij_1, Cij_2) # <-- この行を削除

    if F_z_k <= F_x_prev:
      x_k = z_k
      F_x_k = F_z_k # F(z_k) の値を F(x_k) として保持
    else:
      x_k = x_prev
      F_x_k = F_x_prev # F(x_prev) の値を F(x_k) として保持

    F_history[k-1] = F_x_k
    x_history.append(x_k)
    y_next = x_k + (t_k / t_next) * (z_k - x_k) + ((t_k - 1.0) / t_next) * (x_k - x_prev)

    # パラメータ更新
    y_k = y_next
    t_k = t_next
    x_prev = x_k
    F_x_prev = F_x_k # --- 変更点 3: 次のループのために F(x_k) を F_x_prev にコピー ---

    if k % 10 == 0:
      print(f"Iteration {k}/{N_outer}, F(x) = {F_x_k:.4e}")

  return x_k, F_history#, x_history

# MSE(Mean Squared Error)差分２乗平均
def MSE(original_image, target_image):
   y_size, x_size = original_image.shape
   N = y_size * x_size

    # --- 修正点: 複素数のL2ノルム（絶対値の2乗和）に変更 ---
   mean_squared_error = np.sum(np.abs(original_image - target_image)**2)/N
   return mean_squared_error

# PSNR(Peak Signal to Noise Ratio)
def PSNR(mse):
   psnr = 10*np.log10(255**2 / mse)
   return psnr

# --- main ---
# 生成済みの係数行列の読み込み
# 180どまで20step
cij_x = np.load(r"Cij/cij_x_3.npy")
cij_0 = np.load(r"Cij/cij_0_3.npy")
cij_1 = np.load(r"Cij/cij_1_3.npy")
cij_2 = np.load(r"Cij/cij_2_3.npy")

#b = np.load(r"sinogram/sinogram.npy")
b_unwrap = np.load(r"sinogram/sinogram_unwrap.npy")
b_comp = np.load(r"sinogram/sinogram_complex.npy")
b = np.load(r"sinogram/sinogram.npy")
n_z, n_angle, n_plane = b_unwrap.shape
b_unwrap = b_unwrap / np.amax(b_unwrap)
b_comp = b_comp / np.amax(b_comp)
b = b / np.amax(b)
# img_ori = img_read(r"image/cat_1024.png")
# img_noise = img_read(r"image/cat_1024_noise.png")

lam1 = 0.00001
N_fgp = 5
N_mfista = 200
L = 1*10e4

layer,angle,x = b.shape
print(layer,angle,x)
print(f"lambda:{lam1},N_fgp:{N_fgp},N_mfista:{N_mfista},L:{L}")
# image_3d = np.zeros((layer,x,x),dtype="complex128")
# f_history = np.zeros((layer,N_mfista))
# for i in range(layer):
#    img, f =  MFISTA_tomography(b[i], lam1, N_mfista, N_fgp,"isotropic", cij_x, cij_0, cij_1, cij_2)
#    image_3d[i] = img
#    f_history[i] = f

# np.save("result/2025_1031/MFISTA/MFISTA_0,90/MFISTA_lam0.01.npy",image_3d)
# np.save("result/2025_1031/MFISTA//MFISTA_0,90/F_history_lam0.01.npy",f_history)
# f_his = np.load("result/2025_1031/MFISTA//MFISTA_0,90/F_history_lam0.01.npy")
# print(f_his.shape)
# plt.plot(f_his[60]);plt.xlabel("iteration(N)");plt.ylabel("F=|A(x)-b|+Tv(X)");plt.title(f"z:{0.15/128*60*100:.1f}cm")
# plt.show()
img_1, f1 = MFISTA_tomography(b[int(n_z/2)], lam1, N_mfista, N_fgp,"isotropic",L,cij_x, cij_0, cij_1, cij_2)
img_2, f2 = MFISTA_tomography(b_unwrap[int(n_z/2)], lam1, N_mfista, N_fgp, "isotropic",L,cij_x, cij_0, cij_1, cij_2)
img_3, f3 = MFISTA_tomography(b_comp[int(n_z/2)], lam1, N_mfista, N_fgp, "isotropic",L,cij_x, cij_0, cij_1, cij_2)

plt.subplot(231,title=f"Phase Wrap, isotropic \n intensity,{0.15/n_z*(n_z/2)*100:.1f}cm");plt.imshow(np.abs(img_1),"hsv");plt.colorbar(label='Phase')
plt.subplot(234,title=f"isotropic \n phase, {0.15/n_z*(n_z/2)*100:.1f}cm");plt.imshow(np.angle(img_1),"hsv");plt.colorbar(label='Phase')

plt.subplot(232,title=f"Phase Unwrap, isotropic \n phase, {0.15/n_z*(n_z/2)*100:.1f}cm");plt.imshow(np.abs(img_2),"hsv");plt.colorbar(label='Phase')
plt.subplot(235,title=f"isotropic \n phase, {0.15/n_z*(n_z/2)*100:.1f}cm");plt.imshow(np.angle(img_2),"hsv");plt.colorbar(label='Phase')

plt.subplot(233,title=f"Phase Unwrap, isotropic \n phase, {0.15/n_z*(n_z/2)*100:.1f}cm");plt.imshow(np.abs(img_3),"hsv");plt.colorbar(label='Amplitude')
plt.subplot(236,title=f"isotropic \n phase, {0.15/n_z*(n_z/2)*100:.1f}cm");plt.imshow(np.angle(img_3),"hsv");plt.colorbar(label='Phase')

plt.suptitle(f"lambda:{lam1}, N_fgp:{N_fgp}, N_mfista:{N_mfista} \n projection angle:0,90")
plt.show()

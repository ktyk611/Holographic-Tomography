import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import time

#画像読み込み
def img_read(fname):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    return img

#画像書き出し
def img_write(fname, img):
    img = img.astype(np.uint8)
    cv2.imwrite(fname, img)
    return 0

# 伝達関数 オリジナル
def H_tf(size:int,pitch,z,lamb):
    H = np.zeros((size,size),dtype="complex128")
    f_lim = 1/(lamb*np.sqrt((2*z/size/pitch)**2 +1))
    for i in range(size):
        f_y= (i/size - 1/2)/pitch
        for j in range(size):
            f_x = (j/size - 1/2)/pitch
            if ((np.abs(f_x) <= f_lim) and (np.abs(f_y) <= f_lim)):
                if ( 1/(lamb**2) > f_x**2 + f_y**2):
                    P = z * np.sqrt(1/(lamb**2) - f_x**2 - f_y**2)
                    H[j,i] = np.exp(1j*2*np.pi*P)
                else:
                    P = 0
                    H[j,i] = np.exp(1j*2*np.pi*P)
    return H

# # 伝達関数 ベクトル化
# def H_tf_vectorized(size: int, pitch, z, lamb):
#     # 1. 周波数の上限を計算
#     f_lim = 1 / (lamb * np.sqrt((2 * z / (size * pitch))**2 + 1))

#     # 2. 各ピクセルの周波数座標を計算するための1D配列を作成
#     f_coords = (np.arange(size) / size - 0.5) / pitch

#     # 3. 1D配列から2Dの周波数グリッドを作成
#     # indexing='ij' は、元のコードの H[j, i] のインデックス順に合わせるために重要
#     f_x, f_y = np.meshgrid(f_coords, f_coords, indexing='ij')

#     # 4. if文の条件をブールマスク（True/Falseの配列）に変換
#     # 最初のif文の条件
#     mask_lim = (np.abs(f_x) <= f_lim) & (np.abs(f_y) <= f_lim)
    
#     # 2番目のif文の条件（計算を効率化するため、二乗の和を先に計算）
#     f_squared = f_x**2 + f_y**2
#     mask_sqrt = (1 / (lamb**2)) > f_squared

#     # 5. マスクを組み合わせて、Pの値を計算
#     # Pの値を計算する必要があるのは、mask_limとmask_sqrtの両方がTrueの場所のみ
#     mask_calc_P = mask_lim & mask_sqrt
    
#     # Pをゼロで初期化
#     P = np.zeros((size, size))
#     # マスクがTrueの部分だけ、平方根の計算を実行
#     P[mask_calc_P] = z * np.sqrt(1 / (lamb**2) - f_squared[mask_calc_P])
    
#     # 6. Hの値を計算
#     # Hをゼロで初期化
#     H = np.zeros((size, size), dtype=np.complex128)
#     # mask_limがTrueの場所で、exp()を計算してHに代入
#     # mask_calc_PがFalseの場所ではP=0なので、自動的にexp(0)=1となる
#     H[mask_lim] = np.exp(1j * 2 * np.pi * P[mask_lim])

#     return H

# 凸レンズ伝達関数 オリジナル
def Lens(size:int ,pitch, lamb, focus):
    L = np.zeros((size,size),dtype="complex128")
    for i in range(size):
        for j in range(size):
            y = (i-size/2)*pitch
            x = (j-size/2)*pitch
            L[j,i] = np.exp(-1j*np.pi/lamb *(x*x + y*y)/focus)
    return L

# 凸レンズ伝達関数 ベクトル化
def Lens_vectorized(size:int, pitch, lamb, focus):
    coords = (np.arange(size) - size / 2) * pitch
    X, Y = np.meshgrid(coords, coords)
    Lens = np.exp(-1j*np.pi/lamb * (X**2 + Y**2)/focus)
    return Lens.T

# 伝達関数 ベクトル化
def H_tf_vectorized(size: int, pitch, z, lamb):
    f_lim = 1 / (lamb * np.sqrt((2 * z / (size * pitch))**2 + 1))

    f_coords = (np.arange(size) / size - 0.5) / pitch
    f_x, f_y = np.meshgrid(f_coords, f_coords, indexing='ij')
    
    mask_lim = (np.abs(f_x) <= f_lim) & (np.abs(f_y) <= f_lim)
    f_squared = f_x**2 + f_y**2
    mask_sqrt = (1 / (lamb**2)) > f_squared
    mask_calc_P = mask_lim & mask_sqrt
    P = np.zeros((size, size))
    P[mask_calc_P] = z * np.sqrt(1 / (lamb**2) - f_squared[mask_calc_P])
    H = np.zeros((size, size), dtype=np.complex128)
    H[mask_lim] = np.exp(1j * 2 * np.pi * P[mask_lim])

    return H

# 伝播用の関数
def Prop(image_compex:np.ndarray, size:int, pitch, lam):
    prop_func = H_tf_vectorized(size,pitch,lam)
    img_fft = np.fft.fftshift(np.fft.fft2(image_compex)) * prop_func
    img_prop = np.fft.ifft2(np.fft.fftshift(img_fft))
    return img_prop
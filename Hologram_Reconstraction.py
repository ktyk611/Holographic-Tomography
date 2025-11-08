import numpy as np
import matplotlib.pyplot as plt
import cv2
import numba

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

#参照光
def Ref(size:int, pitch, lamb):
    R = np.zeros((size,size), dtype="complex128")
    theata = 3/4 * np.arcsin(lamb/pitch/2)
    for i in range(size):
        y = (i - size/2)*pitch
        for j in range(size):
            x = (j - size/2)*pitch
            R[i,j] = np.exp(1j*2*np.pi/lamb * (x*np.sin(theata) + y*np.sin(theata)))
    return R

def Ref_vectorized(size: int, pitch, lamb, phase_coefficient):
    theta = phase_coefficient * np.arcsin(lamb / pitch / 2)
    coords_1d = (np.arange(size) - size / 2) * pitch
    X, Y = np.meshgrid(coords_1d, coords_1d)
    phase = (2 * np.pi / lamb) * (X + Y) * np.sin(theta)
    R = np.exp(1j * phase)
    
    return R

#画像読み込み
def img_read(fname):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    return img

#画像書き出し
def img_write(fname, img):
    img = img.astype(np.uint8)
    cv2.imwrite(fname, img)
    return 0

# 正規化(実数のみ)
def normalize(val):
    f = (val-np.amin(val))/(np.amax(val)-np.amin(val))*255
    return f

#クロップ
def crop(fname,num):
    img = fname[int(num/4):int(3*num/4),int(num/4):int(3*num/4)]
    return img

Hol_0 = img_read(r"Hologram/Hologram_0.png")
Hol_90 = img_read(r"Hologram/Hologram_90.png")
size ,qq = Hol_0.shape
lam = 532e-9
z_prop = -0.01
pitch = 3.45e-6

hol_0 = np.sqrt(Hol_0)
hol_90 = np.sqrt(Hol_90)

# --- 空間周波数分布 ---
hol_0_fft = np.fft.fftshift(np.fft.fft2(Hol_0))
hol_90_fft = np.fft.fftshift(np.fft.fft2(Hol_90))

# plt.subplot(121);plt.imshow(np.log(np.abs(hol_0_fft)+1),"gray")
# plt.subplot(122);plt.imshow(np.log(np.abs(hol_90_fft)+1),"gray")
# plt.show()

# --- 物体光成分の抽出 ---
hol_0_fft_crop = hol_0_fft[384:512,384:512]
hol_90_fft_crop = hol_90_fft[384:512,384:512]

# hol_0_fft_crop = hol_0_fft[0:128,0:128]
# hol_90_fft_crop = hol_90_fft[0:128,0:128]

hol_0_fft_pad = np.pad(hol_0_fft_crop,192)
hol_90_fft_pad = np.pad(hol_90_fft_crop,192)
# plt.subplot(121);plt.imshow(np.log(np.abs(hol_0_fft_pad)**2+1),"gray")
# plt.subplot(122);plt.imshow(np.log(np.abs(hol_90_fft_pad)**2+1),"gray")
# plt.show()

# --- 逆伝播 ---
prop_func = H_tf_vectorized(size,pitch, z_prop,lam)
hol_0_prop = hol_0_fft_pad * prop_func
hol_90_prop = hol_90_fft_pad * prop_func

hol_0_recon = np.fft.ifft2(np.fft.fftshift(hol_0_prop))
hol_90_recon = np.fft.ifft2(np.fft.fftshift(hol_90_prop))

# --- ホログラムから Sinogram作成
plt.subplot(221);plt.imshow(np.abs(hol_0_recon),"gray")
plt.subplot(222);plt.imshow(np.angle(hol_0_recon),"gray")
plt.subplot(223);plt.imshow(np.abs(hol_90_recon),"gray")
plt.subplot(224);plt.imshow(np.angle(hol_90_recon),"gray")
plt.show()
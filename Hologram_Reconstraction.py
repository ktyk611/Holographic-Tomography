import numpy as np
import matplotlib.pyplot as plt
import cv2
import numba
from skimage.restoration import unwrap_phase

# --- 記録したホログラムからの画像再構成とSinogram生成 ---

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

# 参照光 ベクトル化
def Ref_vectorized(size: int, pitch, lamb):
    theta = 3/4 * np.arcsin(lamb / pitch / 2)
    coords_1d = (np.arange(size) - size / 2) * pitch
    X, Y = np.meshgrid(coords_1d, coords_1d)
    phase = (2 * np.pi / lamb) * (X + Y) * np.sin(theta)
    R = np.exp(1j * phase)
    
    return R

# 参照光２ ベクトル化
def Ref_vectorized(size: int, pitch, lamb):
    theta = 3/4 * np.arcsin(lamb / pitch / 2)
    coords_1d = (np.arange(size) - size / 2) * pitch
    X, Y = np.meshgrid(coords_1d, coords_1d)
    phase = (2 * np.pi / lamb) * (X - Y) * np.sin(theta)
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

# 伝播用の関数
def Prop(image_compex:np.ndarray,z_prop, pitch, lam):
    size ,q = image_compex.shape
    img_pad = np.pad(image_compex,int(size/2))
    prop_func = H_tf_vectorized(int(2*size),pitch,z_prop,lam)
    img_fft = np.fft.fftshift(np.fft.fft2(img_pad)) * prop_func
    img_prop = np.fft.ifft2(np.fft.fftshift(img_fft))
    y_size, x_size = img_prop.shape
    img_prop_crop = img_prop[int(y_size/4):int(3*y_size/4),int(x_size/4):int(3*x_size/4)]
    return img_prop_crop

Hol_0 = img_read(r"Hologram/Hologram_0.png")
Hol_45 = img_read(r"Hologram/Hologram_45.png")
Hol_90 = img_read(r"Hologram/Hologram_90.png")
size ,qq = Hol_0.shape
lam = 532e-9
z_prop = 0.2
pitch = 20e-6

hol_0 = np.sqrt(Hol_0)
hol_45 = np.sqrt(Hol_45)
hol_90 = np.sqrt(Hol_90)

# --- 空間周波数分布 ---
hol_0_fft = np.fft.fftshift(np.fft.fft2(Hol_0))
hol_45_fft = np.fft.fftshift(np.fft.fft2(Hol_45))
hol_90_fft = np.fft.fftshift(np.fft.fft2(Hol_90))

# plt.subplot(131,title="hol_0_fft");plt.imshow(np.log(np.abs(hol_0_fft)+1),"gray")
# plt.subplot(132,title="hol_45_fft");plt.imshow(np.log(np.abs(hol_45_fft)+1),"gray")
# plt.subplot(133,title="hol_90_fft");plt.imshow(np.log(np.abs(hol_90_fft)+1),"gray")
# plt.show()

# --- 物体光成分の抽出 ---
hol_0_fft_crop = hol_0_fft[78:128,78:128]
hol_45_fft_crop = hol_45_fft[78:128,78:128]
hol_90_fft_crop = hol_90_fft[78:128,78:128]
# hol_0_fft_crop = hol_0_fft[96:128,96:128]
# hol_45_fft_crop = hol_45_fft[96:128,96:128]
# hol_90_fft_crop = hol_90_fft[96:128,96:128]

hol_0_fft_pad = np.pad(hol_0_fft_crop,39)
hol_45_fft_pad = np.pad(hol_45_fft_crop,39)
hol_90_fft_pad = np.pad(hol_90_fft_crop,39)
# 

# --- 逆伝播 ---
prop_func = H_tf_vectorized(size,pitch, z_prop,lam)
hol_0_prop = hol_0_fft_pad * prop_func
hol_45_prop = hol_45_fft_pad * prop_func
hol_90_prop = hol_90_fft_pad * prop_func

hol_0_recon = np.fft.ifft2(np.fft.ifftshift(hol_0_prop))
hol_45_recon = np.fft.ifft2(np.fft.ifftshift(hol_45_prop))
hol_90_recon = np.fft.ifft2(np.fft.ifftshift(hol_90_prop))

recon_0_phase = np.angle(hol_0_recon)
recon_45_phase = np.angle(hol_45_recon)
recon_90_phase = np.angle(hol_90_recon)
n1 = normalize(recon_0_phase)
n2 = normalize(recon_45_phase)
n3 = normalize(recon_90_phase)
# print(np.amin(normalize(recon_0_phase)))
# img_write("Hologram_0_recon_phase.png",normalize(recon_0_phase))
# img_write("Hologram_90_recon_phase.png",normalize(recon_90_phase))

plt.subplot(231,title="0");plt.imshow(n1,"viridis");plt.colorbar(label='Amplitude')
plt.subplot(232,title="45");plt.imshow(n2,"viridis");plt.colorbar(label='Amplitude')
plt.subplot(233,title="90");plt.imshow(n3,"viridis");plt.colorbar(label='Amplitude')
plt.subplot(234,title="0");plt.imshow(np.abs(hol_0_recon),"viridis")
plt.subplot(235,title="45");plt.imshow(np.abs(hol_45_recon),"viridis")
plt.subplot(236,title="90");plt.imshow(np.abs(hol_90_recon),"viridis")
plt.suptitle(f"Hologram Recon Image Phase")
plt.show()

phase_0 = n1
phase_45 = n2
phase_90 = n3

phase_0[0,63:64] = 0
phase_45[0,63:64] = 0
phase_90[0,63:64] = 0

plt.subplot(131);plt.imshow(phase_0,"viridis")
plt.subplot(132);plt.imshow(phase_45,"viridis")
plt.subplot(133);plt.imshow(phase_90,"viridis")
plt.suptitle(f"Hologram Recon Image Phase unwrap")
plt.show()

re_0_unwrap = unwrap_phase(phase_0,(0,63))
re_45_unwrap = unwrap_phase(phase_45,(0,63))
re_90_unqrap = unwrap_phase(phase_90,(0,63))

plt.subplot(131);plt.imshow(re_0_unwrap,"viridis")
plt.subplot(132);plt.imshow(re_45_unwrap,"viridis")
plt.subplot(133);plt.imshow(re_90_unqrap,"viridis")
plt.suptitle(f"Hologram Recon Image Phase unwrap")
plt.show()
# --- ホログラムから Sinogram作成
img_y, img_x = hol_0_recon.shape

sinogram_unwrap = np.zeros((img_y, 3,img_x),dtype="complex128")
for i in range (img_y):
    sinogram_unwrap[i,0,:] = re_0_unwrap[i]
    sinogram_unwrap[i,1,:] = re_45_unwrap[i]
    sinogram_unwrap[i,2,:] = re_90_unqrap[i]

sinogram = np.zeros((img_y, 3,img_x),dtype="complex128")
for i in range (img_y):
    sinogram[i,0,:] = n1[i]
    sinogram[i,1,:] = n2[i]
    sinogram[i,2,:] = n3[i]

sinogram_comp = np.zeros((img_y, 3,img_x),dtype="complex128")
for i in range (img_y):
    sinogram_comp[i,0,:] = hol_0_recon[i]
    sinogram_comp[i,1,:] = hol_45_recon[i]
    sinogram_comp[i,2,:] = hol_90_recon[i]

np.save(r"sinogram/sinogram_unwrap.npy",sinogram_unwrap)
np.save(r"sinogram/sinogram.npy",sinogram)
np.save(r"sinogram/sinogram_complex.npy",sinogram_comp)

# plt.subplot(221);plt.imshow(np.abs(hol_0_recon),"gray")
# plt.subplot(222);plt.imshow(np.angle(hol_0_recon),"hsv")
# plt.subplot(223);plt.imshow(np.abs(hol_90_recon),"gray")
# plt.subplot(224);plt.imshow(np.angle(hol_90_recon),"hsv")
# plt.show()
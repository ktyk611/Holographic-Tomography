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

# 投影画数N
N = 60

lam = 532e-9
z_prop = 0.2
pitch = 20e-6
Hol_size = 128

Hol = np.zeros((N,Hol_size,Hol_size))
hol = np.zeros((N,Hol_size,Hol_size))
hol_fft = np.zeros((N,Hol_size,Hol_size),dtype="complex128")
hol_fft_crop = np.zeros((N,int(Hol_size/4),int(Hol_size/4)),dtype="complex128")
hol_fft_pad = np.zeros((N,Hol_size,Hol_size),dtype="complex128")
hol_prop = np.zeros((N,Hol_size,Hol_size),dtype="complex128")
hol_recon = np.zeros((N,Hol_size,Hol_size),dtype="complex128")
recon_phase = np.zeros((N,Hol_size,Hol_size))
recon_phase_norm = np.zeros((N,Hol_size,Hol_size))
re_unwrap = np.zeros((N,Hol_size,Hol_size))

# --- 逆伝播用の関数 ---
prop_func = H_tf_vectorized(Hol_size,pitch, z_prop,lam)

# --- Hologram の逆伝播計算からの位相成分と位相アンラップ ---
for i in range (N):
    path = f"Hologram/Hologram_{i}.png"
    Hol[i] = img_read(path)
    hol[i] = np.sqrt(Hol[i])
    # --- 空間周波数分布 ---
    hol_fft[i] = np.fft.fftshift(np.fft.fft2(Hol[i]))
    # --- 物体光成分の抽出 ---
    hol_fft_crop[i] = hol_fft[i][96:128,96:128]
    # --- ゼロパディング ---
    hol_fft_pad[i] = np.pad(hol_fft_crop[i],48)
    # --- 逆伝播 ---   
    hol_prop[i] = hol_fft_pad[i] * prop_func
    hol_recon[i] = np.fft.ifft2(np.fft.ifftshift(hol_prop[i]))
    # --- 位相分布（）の正規化 ---
    recon_phase[i] = np.angle(hol_recon[i])
    recon_phase_norm[i] = normalize(recon_phase[i])
    # --- 位相アンラップ ---
    # unwrap_phase(対象画像,(アンラップ基準位置（y,x）))
    re_unwrap[i] = unwrap_phase(recon_phase_norm[i],(0,63))


# --- ホログラムから Sinogram作成
# img_z:投影角数
img_z, img_y, img_x = hol_recon.shape

# --- 位相アンラップした画像によるSinogram
sinogram_unwrap = np.zeros((img_y,img_z,img_x))
for i in range (img_y):
    for j in range (img_z):
        sinogram_unwrap[i,j,:] = re_unwrap[j,i,:]

# --- 位相アンラップ”しない”画像によるSinogram
sinogram_wrap = np.zeros((img_y,img_z,img_x))
for i in range (img_y):
    for j in range (img_z):
        sinogram_wrap[i,j,:] = recon_phase_norm[j,i,:]

# --- 複素振幅分布（生のホログラム）によるSinogram
sinogram_complex = np.zeros((img_y,img_z,img_x),dtype="complex128")
for i in range (img_y):
    for j in range (img_z):
        sinogram_complex[i,j,:] = hol_recon[j,i,:]



print(f"unwrap:{sinogram_unwrap.shape}")
print(f"wrap:{sinogram_wrap.shape}")
print(f"complex:{sinogram_complex.shape}")
np.save(r"sinogram/sinogram_unwrap.npy",sinogram_unwrap)
np.save(r"sinogram/sinogram_wrap.npy",sinogram_wrap)
np.save(r"sinogram/sinogram_complex.npy",sinogram_complex)

plt.subplot(131,title="hol_0_fft");plt.imshow(np.log(np.abs(hol_fft[0])+1),"gray")
plt.subplot(132,title="hol_45_fft");plt.imshow(np.log(np.abs(hol_fft[30])+1),"gray")
plt.subplot(133,title="hol_90_fft");plt.imshow(np.log(np.abs(hol_fft[59])+1),"gray")
plt.show()

plt.subplot(131,title="hol_0_fft");plt.imshow(np.log(np.abs(hol_fft_pad[0])+1),"gray")
plt.subplot(132,title="hol_45_fft");plt.imshow(np.log(np.abs(hol_fft_pad[30])+1),"gray")
plt.subplot(133,title="hol_90_fft");plt.imshow(np.log(np.abs(hol_fft_pad[59])+1),"gray")
plt.show()

plt.subplot(231,title="0");plt.imshow(recon_phase[0],"viridis");plt.colorbar(label='Amplitude')
plt.subplot(232,title="45");plt.imshow(recon_phase[30],"viridis");plt.colorbar(label='Amplitude')
plt.subplot(233,title="90");plt.imshow(recon_phase[59],"viridis");plt.colorbar(label='Amplitude')
plt.subplot(234,title="0");plt.imshow(np.abs(recon_phase[0]),"viridis");plt.colorbar(label='Amplitude')
plt.subplot(235,title="45");plt.imshow(np.abs(recon_phase[30]),"viridis");plt.colorbar(label='Amplitude')
plt.subplot(236,title="90");plt.imshow(np.abs(recon_phase[59]),"viridis");plt.colorbar(label='Amplitude')
plt.suptitle(f"Hologram Recon Image Phase")
plt.show()


plt.subplot(131);plt.imshow(re_unwrap[0],"viridis")
plt.subplot(132);plt.imshow(re_unwrap[30],"viridis")
plt.subplot(133);plt.imshow(re_unwrap[59],"viridis")
plt.suptitle(f"Hologram Recon Image Phase unwrap")
plt.show()

plt.subplot(131,title="unwrap");plt.imshow(sinogram_unwrap[30],"gray")
plt.subplot(132,title="wrap");plt.imshow(sinogram_wrap[30],"gray")
plt.subplot(133,title="complex");plt.imshow(np.angle(sinogram_complex[30]),"gray")
plt.show()
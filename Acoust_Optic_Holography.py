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
                # else:
                    # P = 0
                    # H[j,i] = np.exp(1j*2*np.pi*P)
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

def Ref_vectorized(size: int, pitch, lamb):
    theta = 3/4 * np.arcsin(lamb / pitch / 2)
    coords_1d = (np.arange(size) - size / 2) * pitch
    X, Y = np.meshgrid(coords_1d, coords_1d)
    phase = (2 * np.pi / lamb) * (X + Y) * np.sin(theta)
    R = np.exp(1j * phase)
    
    return R

# 参照光２ ベクトル化
def Ref2_vectorized(size: int, pitch, lamb):
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
def Prop(image_compex:np.ndarray, size:int, pitch, lam):
    prop_func = H_tf_vectorized(size,pitch,lam)
    img_fft = np.fft.fftshift(np.fft.fft2(image_compex)) * prop_func
    img_prop = np.fft.ifft2(np.fft.fftshift(img_fft))
    return img_prop

field = np.load(r"npy/acoustic_vortex_field_512.npy")
c = 3e8
lam = 532e-9
k = 1/lam
n_0 = 1.0003
p_0 = 101325 
ganmma = 1006
pitch = 20e-6
z_prop = 0.1

z, y, x = field.shape
sum_0 = np.zeros((y,x))
sum_90 = np.zeros((y,x))

for i in range (z):
    sum_0 += np.abs(field[:,i,:])
    sum_90 += np.abs(field[:,:,i])

# plt.subplot(121);plt.imshow(sum_0)
# plt.subplot(122);plt.imshow(sum_90)
# plt.show()

phi_p0 = k * (n_0 - 1)/(ganmma*p_0) * sum_0
phi_p90 = k * (n_0 - 1)/(ganmma*p_0) * sum_90

ob_0 = np.zeros((y,x),dtype="complex128") + 255
ob_90 = np.zeros((y,x),dtype="complex128") + 255
# plt.subplot(121);plt.imshow(np.abs(ob_0),"gray")
# plt.subplot(122);plt.imshow(np.angle(ob_90),"gray")
# plt.show()

ob_0_phase = ob_0 * np.exp(1j*phi_p0)
ob_90_phase = ob_90 * np.exp(1j*phi_p90)

ob_0_phase_pad = np.pad(ob_0_phase,int(y/2))
ob_90_phase_pad = np.pad(ob_90_phase,int(y/2))

# plt.subplot(121);plt.imshow(np.abs(ob_0_phase),"gray", vmin=np.amin(np.abs(ob_0_phase)), vmax=np.amax(np.abs(ob_0_phase)))
# plt.subplot(122);plt.imshow(np.angle(ob_90_phase),"gray")
# plt.show()
# img_write("intensity.png",normalize(np.abs(ob_0_phase_pad)))
# img_write("phase.png",normalize(np.angle(ob_0_phase_pad)))

# --- 伝播 ---
prop_func = H_tf_vectorized(2*y, pitch, z_prop, lam)
ob_0_prop = np.fft.fftshift(np.fft.fft2(ob_0_phase_pad)) * prop_func
ob_90_prop = np.fft.fftshift(np.fft.fft2(ob_90_phase_pad)) * prop_func

ob_0_proped = np.fft.ifft2(np.fft.fftshift(ob_0_prop))
ob_90_proped = np.fft.ifft2(np.fft.fftshift(ob_90_prop))

image_y, image_x = ob_0_proped.shape

ob_0_proped_crop = crop(ob_0_proped, int(image_y))
ob_90_proped_crop = crop(ob_90_proped, int(image_y))
# plt.subplot(121);plt.imshow(np.abs(ob_0_proped_crop),"gray")
# plt.subplot(122);plt.imshow(np.angle(ob_0_proped_crop),"gray")
# plt.show()
# --- 干渉 ---
Hol_0 = ob_0_proped_crop/np.sqrt(np.abs(np.amax(ob_0_proped_crop))) + Ref_vectorized(y, pitch, lam)
Hol_90 = ob_90_proped_crop/np.sqrt(np.abs(np.amax(ob_90_proped_crop))) + Ref2_vectorized(y, pitch, lam)

# --- 多重化 ---
# Hol = Hol_0 + Hol_90

img_write("Hologram_0.png",normalize(np.abs(Hol_0)**2))
img_write("Hologram_90.png",normalize(np.abs(Hol_90)**2))
Hol_0_fft = np.fft.fftshift(np.fft.fft2(np.abs(Hol_0)**2))
Hol_90_fft = np.fft.fftshift(np.fft.fft2(np.abs(Hol_90)**2))
# Hol_fft = np.fft.fftshift(np.fft.fft2(np.abs(Hol)**2))

plt.subplot(231);plt.imshow(np.abs(Hol_0)**2,"gray")
plt.subplot(232);plt.imshow(np.abs(Hol_90)**2,"gray")
plt.subplot(234);plt.imshow(np.log(np.abs(Hol_0_fft)+1),"gray")
plt.subplot(235);plt.imshow(np.log(np.abs(Hol_90_fft)+1),"gray")
# plt.subplot(233);plt.imshow(np.abs(Hol)**2,"gray")
# plt.subplot(236);plt.imshow(np.log(np.abs(Hol_fft)+1),"gray")
plt.show()

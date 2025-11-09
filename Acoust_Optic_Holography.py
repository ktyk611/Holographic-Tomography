import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import rotate
import numba

# --- 音響渦の複素振幅の空間分布からのホログラム生成 ---

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

def Ref_vectorized(size: int, pitch, lamb,shift_val):
    theta = shift_val * np.arcsin(lamb / pitch / 2)
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
def Prop(image_compex:np.ndarray,z_prop, pitch, lam):
    size ,q = image_compex.shape
    img_pad = np.pad(image_compex,int(size/2))
    prop_func = H_tf_vectorized(int(2*size),pitch,z_prop,lam)
    img_fft = np.fft.fftshift(np.fft.fft2(img_pad)) * prop_func
    img_prop = np.fft.ifft2(np.fft.ifftshift(img_fft))
    y_size, x_size = img_prop.shape
    img_prop_crop = img_prop[int(y_size/4):int(3*y_size/4),int(x_size/4):int(3*x_size/4)]
    return img_prop_crop

# 疑似係数行列による投影（ラドン変換）
@numba.jit(nopython=True,cache=True)
def Radon_Cij(Img:np.ndarray,Cij_x:np.ndarray, Cij_0:np.ndarray, Cij_1:np.ndarray, Cij_2:np.ndarray):
   Y_size, X_size = Img.shape
   num_angles, num_y, num_x = Cij_x.shape
   Sinogram = np.zeros((num_angles, X_size), dtype=Img.dtype)
   for k in range(num_angles):
      for i in range (Y_size):
         for j in range (X_size):
            x = int(Cij_x[k,i,j])
            if (x>=1 and x<X_size-1):
               Sinogram[k,x-1] += Cij_0[k,i,j] * np.abs(Img[i,j])
               Sinogram[k,x] += Cij_1[k,i,j] * np.abs(Img[i,j])
               Sinogram[k,x+1] += Cij_2[k,i,j] * np.abs(Img[i,j])
            #    Sinogram[k,x-1] += Cij_0[k,i,j] * Img[i,j]
            #    Sinogram[k,x] += Cij_1[k,i,j] * Img[i,j]
            #    Sinogram[k,x+1] += Cij_2[k,i,j] * Img[i,j]
   return Sinogram

# 投影角度の配列生成
def Projection_Angle(scan_start, scan_stop, scan_step):
   angle = np.linspace(scan_start, scan_stop, scan_step)
   print(f"angle \n {angle}")
   return angle

# --- main ---

# 生成済みの係数行列の読み込み
cij_x = np.load(r"Cij/cij_x_60.npy")
cij_0 = np.load(r"Cij/cij_0_60.npy")
cij_1 = np.load(r"Cij/cij_1_60.npy")
cij_2 = np.load(r"Cij/cij_2_60.npy")
field = np.load(r"npy/acoustic_vortex_field_30cm.npy")
c = 3e8
lam = 532e-9
k = 1/lam
n_0 = 1.0003
p_0 = 101325 
ganmma = 1.41
pitch = 20e-6
z_prop = 0.2

z, y, x = field.shape

angle = Projection_Angle(0,180,60)

# --- Scipyの回転をつかったラドン変換（？）
# sum = np.zeros((len(angle),y,x),dtype="complex128")
# center = (int(y/2),int(x/2))
# scale = 1
# for i in range (len(angle)):
#     im_rotate = rotate(field, angle[i], reshape=False, mode='constant', cval=0)
#     sum[i] = np.sum(np.abs(field),axis=1)

# --- 疑似係数行列を使ったラドン変換 ---
length = len(angle)
sino = np.zeros((y,length,x),dtype="complex128")
sum_n = np.zeros((length,y,x),dtype="complex128")
for i in range (y):
    sino[i] = Radon_Cij(field[i],cij_x, cij_0,cij_1,cij_2)
    for j in range (length):
        sum_n[j,i,:] = sino[i,j,:]

phi_p = k * (n_0 - 1)/(ganmma*p_0) * sum_n
print(phi_p.shape)

plt.subplot(131,title="0");plt.imshow(np.abs(phi_p[int(length/5)]),"viridis")
plt.subplot(132,title="45");plt.imshow(np.abs(phi_p[int(length/2)]),"viridis")
plt.subplot(133,title="90");plt.imshow(np.abs(phi_p[int(4*length/5)]),"viridis")
plt.suptitle(f"Acoust Optic effect :abs")

plt.show()

# --- 音響光学効果による位相変調 --- 
ob_phase = np.ones((length,y,x),dtype="complex128") * np.exp(1j*phi_p)

# --- 伝播，干渉 ---
shift = 3/4
Hol = np.zeros(((length,y,x)),dtype="complex128")
Hol_fft = np.zeros(((length,y,x)),dtype="complex128")
ob_phase_prop = np.zeros((length,y,x),dtype="complex128")
for i in range(length):
    # --- 伝播 ---
    ob_phase_prop[i] = Prop(ob_phase[i],z_prop,pitch,lam)
    # --- 干渉 ---
    Hol[i] = ob_phase_prop[i]/np.sqrt(np.abs(np.amax(ob_phase_prop[i])))+ Ref_vectorized(y, pitch, lam,shift)
    # --- ホログラム保存 ---
    path = f"Hologram/Hologram_{i}.png"
    img_write(path,normalize(np.abs(Hol[i])**2))
    # --- ホログラムのスペクトル（確認用，コメントアウトしてもいい） ---
    Hol_fft[i] = np.fft.fftshift(np.fft.fft2(np.abs(Hol[i])**2))

# --- 多重化 ---
# Hol = Hol_0 + Hol_90


plt.subplot(231,title=f"angle:{int(angle[int(length/5)])}");plt.imshow(np.abs(Hol[int(length/5)])**2,"gray")
plt.subplot(232,title=f"angle:{int(angle[int(length/2)])}");plt.imshow(np.abs(Hol[int(length/2)])**2,"gray")
plt.subplot(233,title=f"angle:{int(angle[int(4*length/5)])}");plt.imshow(np.abs(Hol[int(4*length/5)])**2,"gray")

plt.subplot(234,title=f"angle:{int(angle[int(length/5)])}");plt.imshow(np.log(np.abs(Hol_fft[int(length/5)])+1),"gray")
plt.subplot(235,title=f"angle:{int(angle[int(length/2)])}");plt.imshow(np.log(np.abs(Hol_fft[int(length/2)])+1),"gray")
plt.subplot(236,title=f"angle:{int(angle[int(4*length/5)])}");plt.imshow(np.log(np.abs(Hol_fft[int(length*4/5)])+1),"gray")
plt.suptitle(f"Upper row:Hologram, lower rows:spectram \n Special Freqency shift:{shift}")
plt.show()

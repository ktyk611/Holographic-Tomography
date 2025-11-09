import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import special


def calculate_A(Target_mode, Radius, Theta, Number_of_transducer, Radius_of_array, Wave_number):
    """(2)式：A_m(r, theta) を計算"""
    # sp.special.jv は第一種ベッセル関数 J_m
    A_val = Number_of_transducer * (1j)**Target_mode * (np.exp(-1j * Wave_number * Radius) \
    / Radius) * special.jv(Target_mode, Wave_number * Radius_of_array * np.sin(Theta))
    return A_val 

# 円形アレイから生成できる理想的な音響渦
def Ideal_Acoustic_Vortex(Target_mode, Plane_width, Resolution, z_observation,\
                          Number_of_transducer,Radius_of_array,Wave_number):
    x_coords = np.linspace(-Plane_width / 2, Plane_width / 2, Resolution)
    y_coords = np.linspace(-Plane_width / 2, Plane_width / 2, Resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    R = np.sqrt(X**2 + Y**2 + z_observation**2)
    Theta = np.atan2(np.sqrt(X**2 + Y**2), z_observation)
    Phi = np.atan2(Y, X)
    A_m_grid = calculate_A(Target_mode, R, Theta,Number_of_transducer,Radius_of_array,Wave_number)
    phase_term_grid = np.exp(1j * Target_mode * Phi)
    pure_Acoustic_vortex = A_m_grid * phase_term_grid
    return pure_Acoustic_vortex

# 単一円形アレイによる音場
# input_signal:1次元ndarray（音源の複素振幅の配列）を引数にした関数
def Ring_Array_Acoustic_field(Number_of_element, Input_signal, Freqency, \
                        Speed, Radius_of_Array, Plane_width, Resolution, z_observation):
    # 波数
    wave_length = 2*np.pi* Freqency / Speed
    # 素子の方位角
    phi_n_array = 2 *np.pi * np.arange(Number_of_element) / Number_of_element
    # アレイ素子の座標 (x_n, y_n, z_n)
    x_n = Radius_of_Array * np.cos(phi_n_array)
    y_n = Radius_of_Array * np.sin(phi_n_array)
    z_n = np.zeros(Number_of_element)
    # ステップ1: 観測平面のグリッド座標とアレイ素子の座標を準備
    x_coords = np.linspace(-Plane_width / 2, Plane_width / 2, Resolution)
    y_coords = np.linspace(-Plane_width / 2, Plane_width / 2, Resolution)
    # (resolution, resolution) のグリッド
    X, Y = np.meshgrid(x_coords, y_coords)
    # ステップ2: ブロードキャストを使い、全素子から全グリッド点までの距離を一括計算
    # 各座標配列の次元を (resolution, resolution, 1) や (1, 1, Ne) に変形する
    dx = X[..., np.newaxis] - x_n.reshape(1, 1, Number_of_element)
    dy = Y[..., np.newaxis] - y_n.reshape(1, 1, Number_of_element)
    dz = z_observation - z_n.reshape(1, 1, Number_of_element)
    # (resolution, resolution, Ne) の3D配列
    distances = np.sqrt(dx**2 + dy**2 + dz**2) 
    # ステップ3: 全点における伝播項と音圧を一括計算
    # ゼロ除算を避けるために微小値を追加
    propagator = np.exp(-1j * wave_length * distances) / (4 * np.pi * (distances + 1e-9))
    # a_solutionを(1, 1, Ne)に変形して掛ける
    # np.sumで素子の次元(axis=2)について合計し、2Dの音場データを作成
    Acoustic_field = np.sum(Input_signal.reshape(1, 1, Number_of_element) * propagator, axis=2)
    return Acoustic_field

target_mode = 1
plane_width = 0.05
z_prop = 0.05
resolution = 128
num_of_transducer = 16
r_of_array = 0.035
speed = 340
freq = 40e3
wave_num = freq / speed
delta = 0*np.pi
signal_pahse = np.linspace(0,2*np.pi,16) + delta
signal = np.exp(1j*signal_pahse)


field = np.zeros((resolution,resolution,resolution),dtype="complex128")
for i in range(resolution):
    z_obs = z_prop * i /resolution
    field[i] = Ring_Array_Acoustic_field(num_of_transducer,signal,freq,speed,r_of_array,plane_width,resolution,z_obs)
    if (i%25 == 0):
        print(f"iteration:{i}")

field = np.pad(field,int(resolution/2))#[int(resolution/2):int(resolution+resolution/2),:,:]
np.save(r"npy/acoustic_vortex_field_5cm_delay0.npy",field)

# field = np.load("acoustic_vortex_field.npy")
# angle = np.angle(field)
# np.save("acoustic_vortex_angle.npy",angle)
# angle = np.load("acoustic_vortex_angle.npy")
# print(angle[256].shape)
plt.subplot(231,title=f"prop:{z_prop*10/resolution:.3f}");plt.imshow(np.angle(field[10]),cmap='hsv')
plt.subplot(232,title=f"prop:{z_prop*resolution/2/resolution}");plt.imshow(np.angle(field[int(resolution/2)]),cmap='hsv')
plt.subplot(233,title=f"prop:{z_prop*(resolution-1)/resolution:.3f}");plt.imshow(np.angle(field[resolution-1]),cmap='hsv')
plt.subplot(234,title=f"");plt.imshow(np.abs(field[10]),cmap='viridis')
plt.subplot(235,title=f"");plt.imshow(np.abs(field[int(resolution/2)]),cmap='viridis')
plt.subplot(236,title=f"");plt.imshow(np.abs(field[resolution-1]),cmap='viridis')
plt.suptitle(f"plane_width:{plane_width},z_prop:{z_prop},delta:{delta:.3f}")
plt.show()
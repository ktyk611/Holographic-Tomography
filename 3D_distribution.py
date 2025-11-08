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

target_mode = 1
plane_width = 0.15
resolution = 512
num_of_transducer = 16
r_of_array = 0.035
speed = 340
freq = 40e3
wave_num = freq / speed
N = 512

field = np.zeros((N,resolution,resolution),dtype="complex128")
for i in range(N):
    z_obs = 0.1 * i /resolution
    field[i] = Ideal_Acoustic_Vortex(target_mode,plane_width,resolution,z_obs,num_of_transducer,r_of_array,wave_num)
    if (i%25 == 0):
        print(f"iteration:{i}")

np.save("acoustic_vortex_field.npy",field)

# field = np.load("acoustic_vortex_field.npy")
# angle = np.angle(field)
# np.save("acoustic_vortex_angle.npy",angle)
# angle = np.load("acoustic_vortex_angle.npy")
# print(angle[256].shape)
# plt.subplot(231);plt.imshow(angle[:,:,0],cmap='hsv')
# plt.subplot(232);plt.imshow(angle[:,:,63],cmap='hsv')
# plt.subplot(233);plt.imshow(angle[:,:,127],cmap='hsv')
# plt.subplot(234);plt.imshow(angle[:,:,255],cmap='hsv')
# plt.subplot(235);plt.imshow(angle[:,:,382],cmap='hsv')
# plt.subplot(236);plt.imshow(angle[:,:,511],cmap='hsv')
# plt.show()
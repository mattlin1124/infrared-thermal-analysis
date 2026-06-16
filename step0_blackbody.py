import numpy as np
import matplotlib.pyplot as plt
from file_open import thermal_file_open

# =========================
# 0. Settings
# =========================
thermal_filepath = r"D:\mattsdata\生醫影像\infrared_hw\min_yu\1211_2_temp4envi"
output_path = r"D:\mattsdata\生醫影像\infrared_hw\thermal_bb_corrected.npy"
rows, cols, bands = 240, 320, 3000
dtype = np.float64  # MATLAB 'double'
offset = 0
byteorder = '<'  # little-endian
interleave = 'bsq'
T_BB_TRUE = 25.0   # ← 用你同學的設定（請確認實驗記錄）

# 黑體 ROI（比單一 pixel 穩定）
bb_y0, bb_y1 = 159, 174
bb_x0, bb_x1 = 71, 83

# =========================
# 1. Load raw data
# =========================
T_raw = thermal_file_open(thermal_filepath).astype(np.float32)
H, W, T = T_raw.shape
print("Loaded thermal data:", T_raw.shape)

x = 80 #黑體位置
y = 165
profile = T_raw[y, x, :]
H, W, T = T_raw.shape


Black = 25
Temperature_bias = []
for i in range(T):
    Temperature_bias.append(profile[i]-Black)
Temperature_bias_array = np.array(Temperature_bias, dtype=np.float32)
After_temperature_correction = T_raw - Temperature_bias_array[None,None,:]

MV_x = 0
MV_y = 0
profile_MV = After_temperature_correction[MV_y, MV_x, :] 
LCV_x = 287#176
LCV_y = 156#160
profile_LSV = After_temperature_correction[LCV_y, LCV_x, :] 
TWVV_x = 219
TWVV_y = 124
profile_TWVV = After_temperature_correction[TWVV_y, TWVV_x, :]

# 畫 profile 圖
plt.figure()
plt.plot(np.arange(1, bands+1), profile_LSV)
plt.xlabel('Frame')
plt.ylabel('Temperature')
plt.title(f'Pixel profile at (row={LCV_y}, col={LCV_x})')
plt.grid(True)
plt.show()

# =======================
# 儲存成 NPY 檔案（給後續影像對位使用）
# =======================

output_npy_path = r"D:\mattsdata\生醫影像\infrared_hw\thermal_step0.npy"

np.save(output_npy_path, After_temperature_correction.astype(np.float32))

print("STEP0 完成，已儲存 NPY 檔案：")
print(output_npy_path)
print("資料 shape:", After_temperature_correction.shape)
print("dtype:", After_temperature_correction.dtype)

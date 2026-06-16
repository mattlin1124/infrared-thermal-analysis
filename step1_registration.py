# step1_registration.py
# ==========================================================
# Step 1: Image Registration (translation-only)
# Input : thermal_bb_corrected.npy  (after blackbody calibration)
# Output: thermal_aligned.npy
# Method: Phase correlation
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# ------------------------------------------------
# 0. Paths
# ------------------------------------------------
input_path  = r"D:\mattsdata\生醫影像\infrared_hw\thermal_step0.npy"
output_path = r"D:\mattsdata\生醫影像\infrared_hw\thermal_aligned.npy"

if not os.path.isfile(input_path):
    raise FileNotFoundError("找不到 thermal_bb_corrected.npy，請先完成 Step 0")

# ------------------------------------------------
# 1. Load calibrated thermal data
# ------------------------------------------------
T = np.load(input_path).astype(np.float32)

# 保證 shape = (rows, cols, bands)
if T.ndim != 3:
    raise ValueError(f"資料維度錯誤: {T.shape}")

# 若是 (bands, rows, cols) 則轉置
if T.shape[0] == 3000:
    T = np.transpose(T, (1, 2, 0))

rows, cols, bands = T.shape
print("Loaded calibrated data:", T.shape)

# ------------------------------------------------
# 2. Choose reference frame
# ------------------------------------------------
ref_idx = bands // 2  # 中間 frame 通常最穩
ref = T[:, :, ref_idx]

print("Reference frame index:", ref_idx)

# ------------------------------------------------
# 3. Phase correlation to estimate shifts
# ------------------------------------------------
shifts = np.zeros((bands, 2), dtype=np.float32)  # (dx, dy)

for i in range(bands):
    shift, _ = cv2.phaseCorrelate(ref, T[:, :, i])
    dx, dy = shift
    shifts[i] = [dx, dy]

print("dx range:", shifts[:,0].min(), "~", shifts[:,0].max())
print("dy range:", shifts[:,1].min(), "~", shifts[:,1].max())

# ------------------------------------------------
# 4. Apply translation to each frame
# ------------------------------------------------
T_aligned = np.zeros_like(T)

for i in range(bands):
    dx, dy = shifts[i]
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    T_aligned[:, :, i] = cv2.warpAffine(
        T[:, :, i],
        M,
        (cols, rows),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

# ------------------------------------------------
# 5. Save aligned data
# ------------------------------------------------
np.save(output_path, T_aligned.astype(np.float32))
print("Saved aligned data to:")
print(output_path)

# ------------------------------------------------
# 6. Validation plots
# ------------------------------------------------
# (a) dx / dy over time
plt.figure(figsize=(10,4))
plt.plot(shifts[:,0], label="dx")
plt.plot(shifts[:,1], label="dy")
plt.xlabel("Frame")
plt.ylabel("Shift (pixels)")
plt.title("Estimated translation shifts")
plt.legend()
plt.grid(True)
plt.show()

# (b) Before vs After difference (single frame)
test_idx = ref_idx + 10 if ref_idx + 10 < bands else ref_idx - 10

diff_before = T[:, :, test_idx] - ref
diff_after  = T_aligned[:, :, test_idx] - ref

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(diff_before, cmap="gray")
plt.title("Before registration")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(diff_after, cmap="gray")
plt.title("After registration")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n=== Step 1 complete ===")

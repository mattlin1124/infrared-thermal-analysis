import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# ======================================================
# Step 3: Spatial classification with manual ROI
# ======================================================

# -------------------------
# 0) Paths
# -------------------------
aligned_path = r"D:\mattsdata\生醫影像\infrared_hw\thermal_aligned.npy"
out_dir = r"D:\mattsdata\生醫影像\infrared_hw\step3_outputs"
os.makedirs(out_dir, exist_ok=True)

if not os.path.isfile(aligned_path):
    raise FileNotFoundError("找不到 thermal_aligned.npy，請確認 Step1 的輸出檔名與路徑")

T = np.load(aligned_path).astype(np.float32)

# 若檔案是 (frames, rows, cols)，轉成 (rows, cols, frames)
if T.shape[0] == 3000:
    T = np.transpose(T, (1, 2, 0))

rows, cols, frames = T.shape
print("Loaded thermal data:", T.shape)

# -------------------------
# 1) Define time windows (2 Hz, BL 10min, OCC 5min, PO 10min)
# -------------------------
fs = 2.0

BL  = (0, int(10 * 60 * fs))                 # 0 ~ 1200
OCC = (BL[1], BL[1] + int(5 * 60 * fs))      # 1200 ~ 1800
PO  = (OCC[1], min(frames, OCC[1] + int(10 * 60 * fs)))  # 1800 ~

# Hyperemia window H（稍偏中段，兼顧 LCV + MV）
H = (PO[0] + int(1 * 60 * fs), min(PO[0] + int(5 * 60 * fs), PO[1]))
h0, h1 = H
print("Using H window:", H)

# -------------------------
# 2) I^(max-min)H
# -------------------------
I_maxH = T[:, :, h0:h1].max(axis=2)
I_minH = T[:, :, h0:h1].min(axis=2)
I_mmH  = I_maxH - I_minH

# -------------------------
# 3) Background suppression (Gaussian blur)
# -------------------------
# sigma 控制「背景尺度」：
#   小 → 保留細節多（MV 多）
#   大 → 背景更平滑（MV 乾淨）
sigma = 2.5   # 建議 2.0 ~ 3.5 之間慢慢試

S = cv2.GaussianBlur(
    I_mmH.astype(np.float32),
    (0, 0),          # kernel size 由 sigma 決定（標準做法）
    sigmaX=sigma,
    sigmaY=sigma
)

I_E = (I_mmH - S).astype(np.float32)

# -------------------------
# 4) Forearm mask (baseline-based)
# -------------------------
bl0, bl1 = BL
mean_BL = T[:, :, bl0:bl1].mean(axis=2)

thr_forearm = np.percentile(mean_BL, 35)
forearm_mask = mean_BL > thr_forearm

# -------------------------
# 5) Manual polygon ROI selection (coarse boundary)
# -------------------------
print("請沿著前臂邊界『大略點選多個點』，完成後按 Enter")

plt.figure(figsize=(6, 4))
plt.imshow(mean_BL, cmap="gray")
plt.title("Click multiple points around forearm boundary, then press Enter")
plt.axis("off")

# 允許點很多點（直到按 Enter）
pts = plt.ginput(n=-1, timeout=0)
plt.close()

if len(pts) < 3:
    raise RuntimeError("ROI needs at least 3 points")

# 轉成整數座標
poly = np.array([(int(x), int(y)) for x, y in pts])

# 建立 polygon ROI mask
roi_mask = np.zeros((rows, cols), dtype=np.uint8)
cv2.fillPoly(roi_mask, [poly], 1)
roi_mask = roi_mask.astype(bool)

# 最終分析範圍：forearm ∩ polygon ROI
analysis_mask = forearm_mask & roi_mask

print(f"Polygon ROI created with {len(poly)} points")


# -------------------------
# 6) Vessel candidates (threshold in ROI)
# -------------------------
vals = I_E[analysis_mask]
thr_v = np.percentile(vals, 85)

vessel_bin = (I_E > thr_v) & analysis_mask

# -------------------------
# 7) Connected components (BFS)
# -------------------------
visited = np.zeros_like(vessel_bin, dtype=bool)
labels = np.zeros_like(vessel_bin, dtype=np.int32)
comp_sizes = []
label_id = 0

neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

for y in range(rows):
    for x in range(cols):
        if vessel_bin[y, x] and not visited[y, x]:
            label_id += 1
            q = [(y, x)]
            visited[y, x] = True
            labels[y, x] = label_id
            cnt = 1
            while q:
                cy, cx = q.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        if vessel_bin[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            labels[ny, nx] = label_id
                            q.append((ny, nx))
                            cnt += 1
            comp_sizes.append(cnt)

if label_id == 0:
    raise RuntimeError("No vessel components detected")

# -------------------------
# 8) MV / LCV classification
# -------------------------
sizes = np.array(comp_sizes)
lcv_size_thr = np.percentile(sizes, 95)

mv_mask = np.zeros_like(vessel_bin, dtype=bool)
lcv_mask = np.zeros_like(vessel_bin, dtype=bool)

for cid in range(1, label_id + 1):
    comp = (labels == cid)
    if sizes[cid - 1] >= lcv_size_thr:
        lcv_mask |= comp
    else:
        mv_mask |= comp

mv_mask &= ~lcv_mask
twvv_mask = analysis_mask & (~(mv_mask | lcv_mask))

# -------------------------
# 9) Save outputs
# -------------------------
np.save(os.path.join(out_dir, "I_E.npy"), I_E)
np.save(os.path.join(out_dir, "mv_mask.npy"), mv_mask.astype(np.uint8))
np.save(os.path.join(out_dir, "lcv_mask.npy"), lcv_mask.astype(np.uint8))
np.save(os.path.join(out_dir, "twvv_mask.npy"), twvv_mask.astype(np.uint8))
np.save(os.path.join(out_dir, "analysis_mask.npy"), analysis_mask.astype(np.uint8))

plt.imsave(os.path.join(out_dir, "mv_mask.png"), mv_mask.astype(np.uint8)*255, cmap="gray")
plt.imsave(os.path.join(out_dir, "lcv_mask.png"), lcv_mask.astype(np.uint8)*255, cmap="gray")
plt.imsave(os.path.join(out_dir, "twvv_mask.png"), twvv_mask.astype(np.uint8)*255, cmap="gray")

# -------------------------
# 10) Summary visualization (like PPT)
# -------------------------
plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(I_mmH, cmap="gray")
plt.title("I^(max-min)_H")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(I_E, cmap="gray")
plt.title("Enhanced I_E (Mean blur)")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(vessel_bin, cmap="gray")
plt.title("Vessel candidates")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(mv_mask, cmap="gray")
plt.title("MV")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(lcv_mask, cmap="gray")
plt.title("LCV")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(twvv_mask, cmap="gray")
plt.title("TWVV")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Finished Step 3 with ROI-based spatial classification")
print(f"H={H}, size={sizes}, thr_v={thr_v:.3f}, lcv_size_thr={lcv_size_thr:.1f}")

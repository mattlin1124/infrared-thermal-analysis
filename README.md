# 紅外線熱影像上臂阻斷分析作業

本 repository 收錄紅外線熱影像觀察作業的主要分析程式。
本作業以單一受試者之前臂紅外線熱影像資料為例，透過上臂阻斷法觀察前臂皮膚表面溫度在不同實驗階段中的變化，並分析不同血管相關區域的溫度振盪特性。

整體流程包含熱影像資料讀取、黑體溫度校正、影像對位、MV / LCV / TWVV 區域分類、溫度時間序列擷取、Morlet 小波時頻分析，以及不同生理頻段下的 band energy 比較。

> 本專案主要作為課程作業、研究學習與成果展示用途。
> 原始熱影像資料因檔案大小與資料隱私考量，未上傳至此 repository。

---

## 專案簡介

紅外線熱影像提供一種非侵入式方式，用來觀察皮膚表面溫度隨時間的變化。這些微小的溫度振盪可能與底層血管調控機制有關。

本作業採用上臂阻斷法進行量測，透過暫時阻斷上臂血流並觀察解除阻斷後的前臂溫度變化，分析不同血管區域在 baseline、occlusion 與 post-occlusion 階段中的溫度反應與頻域特徵。

實驗資料分為三個主要階段：

```text
BL  ：Baseline，阻斷前基線期，約 10 分鐘
OCC ：Occlusion，上臂血流阻斷期，約 5 分鐘
PO  ：Post-occlusion，解除阻斷後恢復期，約 10 分鐘
```

本作業將前臂熱影像中的分析區域概念性分為三類：

* **MV**：Microvascular region，微血管區域
* **LCV**：Larger cutaneous vessel region，較大血管區域
* **TWVV**：Tissue without visible vessels，非明顯血管區域

針對每一類區域，本作業擷取其平均溫度時間序列，並透過高通濾波與 Morlet continuous wavelet transform（CWT）進行時頻分析，最後計算不同生理頻段中的能量分布，用以比較三類區域的頻域特徵差異。

---

## 專案結構

```text
.
├── README.md
├── requirements.txt
├── file_open.py
├── step0_blackbody.py
├── step1_registration.py
├── step3_region_classification.py
├── step5_frequency_analysis.py
└── step6_band_energy_stats.py
```

---

## 分析流程

### Step 0：熱影像讀取與黑體校正

相關檔案：

```text
file_open.py
step0_blackbody.py
```

`file_open.py` 提供原始熱影像資料讀取函式，主要用於讀取 BSQ 格式的熱影像資料。

`step0_blackbody.py` 進行黑體溫度校正。程式會根據黑體參考位置的溫度變化，估計每一個 frame 的溫度偏移量，並將該偏移量從整段熱影像資料中扣除，以降低量測過程中的溫度漂移影響。

預期輸出：

```text
thermal_step0.npy
```

---

### Step 1：影像對位

相關檔案：

```text
step1_registration.py
```

此步驟針對校正後的熱影像序列進行影像對位。
由於受試者手部可能有輕微移動，程式使用 phase correlation 估計各 frame 與 reference frame 之間的平移量，並透過影像平移校正降低位移對後續 ROI 與溫度時間序列分析的影響。

預期輸出：

```text
thermal_aligned.npy
```

---

### Step 3：MV / LCV / TWVV 區域分類

相關檔案：

```text
step3_region_classification.py
```

此步驟負責定義三種主要分析區域：

* MV：微血管區域
* LCV：較大血管區域
* TWVV：非明顯血管區域

本作業並非追求精準的像素級血管邊界分割，而是採用 region-based 的分析方式。程式會根據熱影像中的血管分布特徵、baseline 期間的前臂區域，以及使用者手動選取的前臂 ROI，產生三類區域 mask，作為後續溫度時間序列分析的基礎。

此外，此步驟會根據實驗時間設定切分：

```text
BL  ：0–10 min
OCC ：10–15 min
PO  ：15–25 min
```

其中 post-occlusion 階段可用來觀察解除上臂阻斷後的溫度變化與反應性充血相關特徵。

預期輸出：

```text
step3_outputs/
├── mv_mask.npy
├── lcv_mask.npy
├── twvv_mask.npy
├── analysis_mask.npy
├── mv_mask.png
├── lcv_mask.png
└── twvv_mask.png
```

---

### Step 5：Morlet 小波時頻分析

相關檔案：

```text
step5_frequency_analysis.py
```

此步驟為本作業的主要分析核心。
程式會根據 Step 3 產生的 MV、LCV、TWVV masks，擷取各區域的平均溫度時間序列，也就是 temperature signal region（TSR）。

接著進行：

* baseline normalization
* high-pass filtering
* Morlet continuous wavelet transform（CWT）
* scalogram 繪製
* time-averaged wavelet power spectrum 計算
* band energy 計算

此步驟用於觀察不同血管區域在時間與頻率上的能量分布差異，並分析上臂阻斷與解除阻斷後，不同血管尺度區域所呈現的溫度振盪特徵。

預期輸出：

```text
step5_outputs/
├── mv_highpass.npy
├── lcv_highpass.npy
├── twvv_highpass.npy
├── scalogram figures
├── time-averaged power spectrum figure
└── band energy results
```

---

### Step 6：Band Energy 統計圖

相關檔案：

```text
step6_band_energy_stats.py
```

此步驟會根據 Step 5 的 high-pass 訊號與小波分析結果，計算不同生理頻段中的 band energy，並產生 MV、LCV、TWVV 三類區域的比較圖。

圖中包含：

* 不同血管區域的 band energy bar chart
* 誤差棒
* 以時間視窗為基礎的探索性統計分析

需要注意的是，本作業僅使用單一受試者資料，因此統計檢定僅反映單次量測期間內的變化情形，並不代表跨受試者的群體推論結果。

預期輸出：

```text
step6_outputs/
├── band_energy_stats.png
├── band_energy_windows.csv
└── band_energy_pvalues.csv
```

---

## 生理頻段設定

本作業主要分析下列低頻生理相關頻段：

```text
0.005–0.0095 Hz
0.0095–0.02 Hz
0.02–0.06 Hz
0.06–0.2 Hz
```

這些頻段用於比較 MV、LCV 與 TWVV 三類區域在不同血管調控相關頻率範圍中的能量差異。

---

## 安裝方式

建議使用 Python 3.9 以上版本。

安裝所需套件：

```bash
pip install -r requirements.txt
```

`requirements.txt` 內容建議如下：

```text
numpy
matplotlib
opencv-python
PyWavelets
scipy
```

---

## 執行方式

執行前，請先準備原始熱影像資料，並依照自己的資料位置修改各程式中的 input / output 路徑。

建議執行順序如下：

```bash
python step0_blackbody.py
python step1_registration.py
python step3_region_classification.py
python step5_frequency_analysis.py
python step6_band_energy_stats.py
```

---

## 注意事項

* 原始熱影像資料未包含於此 repository。
* 使用者需自行準備熱影像資料，並修改程式中的檔案路徑。
* 部分程式需要透過 matplotlib 互動式視窗手動選取 ROI，請在支援 GUI 的環境中執行。
* 本 repository 主要展示分析流程與作業成果，並非完整自動化 pipeline。
* 本作業以單一受試者資料進行分析，因此結果主要作為方法展示與探索性觀察。
* 統計結果主要反映單一量測資料中不同時間視窗的變異，不應解讀為跨受試者或臨床層級的推論。

---

## 作業流程總結

本作業完整流程如下：

```text
原始熱影像資料
→ 黑體溫度校正
→ 影像對位
→ 上臂阻斷實驗階段切分（BL / OCC / PO）
→ MV / LCV / TWVV 區域分類
→ 溫度時間序列擷取
→ High-pass filtering
→ Morlet 小波轉換
→ 時頻分析
→ Band energy 比較
```

本專案展示如何從上臂阻斷法的紅外線熱影像資料中，擷取不同血管相關區域的溫度時間序列，並透過時頻分析觀察其在不同生理頻段中的能量分布差異。

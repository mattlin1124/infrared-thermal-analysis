# 紅外線熱影像分析作業

本 repository 收錄紅外線熱影像觀察作業的主要分析程式。  
本作業以單一受試者之前臂紅外線熱影像資料為例，分析不同血管相關區域的溫度振盪特性，包含微血管區域（MV）、較大血管區域（LCV）以及非明顯血管區域（TWVV）。

整體流程包含熱影像資料讀取、黑體溫度校正、影像對位、區域分類、溫度時間序列擷取、Morlet 小波時頻分析，以及不同生理頻段下的 band energy 比較。

> 本專案主要作為課程作業、研究學習與成果展示用途。  
> 原始熱影像資料因檔案大小與資料隱私考量，未上傳至此 repository。

---

## 專案簡介

紅外線熱影像提供一種非侵入式方式，用來觀察皮膚表面溫度隨時間的變化。這些微小的溫度振盪可能與底層血管調控機制有關。

本作業將前臂熱影像中的分析區域概念性分為三類：

- **MV**：Microvascular region，微血管區域
- **LCV**：Larger cutaneous vessel region，較大血管區域
- **TWVV**：Tissue without visible vessels，非明顯血管區域

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
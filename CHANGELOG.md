# 變更紀錄

本檔案記錄 `acoustic-feature-lab` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

### 新增

- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`。
- 流程圖也有英文版（`docs/workflow.html`、`docs/images/workflow.png`），中文版是 `docs/workflow.zh-TW.html` 與 `docs/images/workflow.zh-TW.png`。

---

## [0.1.0] - 2026-10-09

### 新增

- 首次釋出。
- 可設定的倒頻譜前端：框長與位移依實際取樣率換算、對稱 Hamming 窗、64 Hz 起的 23 個梅爾濾波器、HTK 縮放的 DCT 與 `c0`、正弦倒頻譜提升、可選 `c0`／對數能量／不加能量項，以及 13／26／39 維的 Δ／ΔΔ 版面，並以 `feature_layout` 追溯每一欄。
- 語句層級彙整（平均、標準差、極值、百分位數、偏態、峰態），可先做 CMN 或 CMVN，並拒絕會讓統計量變成常數的組合。
- 特徵設定的 ablation 框架：倒頻譜數 × 濾波器數 × 動態階數 × 能量項 × 框長位移 × 提升 × 彙整統計量 × 降維 × 估計器，所有設定在相同的重複分組 k 折上評估，並以 Nadeau–Bengio 修正 t 檢定與 Wilcoxon 符號等級檢定比較基準設定，Holm 校正。
- 連續嚴重度迴歸（線性、Ridge、SVR、隨機森林、MLP）與二類偵測（Logistic、SVC、隨機森林、MLP），標準化與 PCA／LDA 只在訓練折擬合，可選巢狀交叉驗證調參；PyTorch 感知器放在 `dl` extra。
- 最小平方推論：係數標準誤、t、p、信賴區間、R²、調整後 R²、整體 F 檢定、AIC、BIC；殘差自由度為 0 時輸出 NaN 並標記為不可估計。
- 迴歸診斷：Shapiro–Wilk、Jarque–Bera、Breusch–Pagan、Durbin–Watson、leverage、Cook's distance、VIF 與條件數。
- 偏 F 檢定逐步選擇、正交多項式基底的階數掃描（樣本內指標與 LOOCV／k 折 RMSE 並列）、與閉式解比對的梯度下降。
- 特徵與嚴重度的相關分析（Pearson 的 Fisher z 信賴區間、Spearman、Holm／BH 校正），以及預測值與評分的一致性分析（Lin's CCC、Bland–Altman）。
- 批次擷取資料夾樹的音框層級特徵，輸出 npz、HTK（大端序、100 ns 單位、依實際版面計算 `parmKind`）或文字矩陣，每個檔案都有狀態紀錄。
- 可控嚴重度的合成母音（Rosenberg 聲門脈衝、jitter、shimmer、共振峰濾波與雜訊）與合成迴歸資料，供測試與範例使用。
- typer CLI：`synthesize`、`index`、`prepare`、`evaluate`、`train`、`predict`、`extract`、`ablate`、`correlate`、`regress`、`stepwise`、`order-sweep`；除了只寫出 `dataset.csv` 的 `index`，每個指令都在輸出資料夾寫出 `config.yaml` 與 `manifest.json`。

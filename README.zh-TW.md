# Index-TTS (客製化分支)

**語言**: [English](README.md) | [繁體中文](#)

> [!NOTE]
> **這是官方 Index-TTS 專案的客製化分支。**
>
> 如需查看原始 README 和官方文件，請訪問：
> **https://github.com/index-tts/index-tts**

---

## 🎯 主要重點：使用 `test_streaming.py` 進行串流 TTS

本分支主要專注於**即時串流 TTS 功能**，用於與 LiveKit 及其他即時應用程式整合。核心演示在 `test_streaming.py` 中，展示了具有速度控制和並行音訊生成/播放的進階串流功能。

### 🚀 串流快速開始

```bash
# 基本串流測試
uv run test_streaming.py

# 使用自訂參考音檔
uv run test_streaming.py --ref_audio examples/Joneshong.wav

# 使用速度控制（播放速度）
uv run test_streaming.py --speed 1.3

# 使用參考音檔預處理（在 TTS 前加速參考音檔）
uv run test_streaming.py --pre_speed_ref 1.3

# 包含所有選項的完整範例
uv run test_streaming.py \
  --version v2 \
  --method token \
  --ref_audio examples/Joneshong.wav \
  --pre_speed_ref 1.3 \
  --speed 1.0 \
  --warmup \
  --text "你的測試文本"
```

### 📖 `test_streaming.py` 功能

#### 1. **雙重速度控制策略**
- **`--pre_speed_ref`**：在 TTS 生成前預處理參考音檔
  - 使用時間拉伸加速參考音檔
  - TTS 模型模仿較快的語速模式
  - 使用 `pyrubberband` 進行高品質時間拉伸

- **`--speed`**：後處理播放速度
  - 在播放期間對生成的音訊應用 DSP 時間拉伸
  - 獨立於 TTS 生成
  - 適用於即時播放控制

#### 2. **串流架構**
- **並行生成與播放**：音訊片段同時生成和播放
- **佇列式管線**：高效的音訊片段管理
- **低延遲**：首次 token 延遲追蹤
- **即時統計**：生成倍率、RTF（即時因子）、並行效率

#### 3. **智慧文本分割**
- **Token-based**（僅 v2）：由模型自動分割
- **Word-based**：基於標點符號的手動分割
- 可配置的片段長度以優化串流

#### 4. **全面統計資訊**
腳本提供詳細的性能指標：
- 首次 token 延遲
- 音訊生成倍率（音訊/處理速度）
- 整體 RTF（即時因子）
- 並行效率（生成與播放重疊的程度）
- 參考音檔分析（格式、位元率、時長）

---

## 📊 基準測試

### 🚀 快速開始 - 一鍵執行完整測試

```bash
# Windows（推薦使用 Python 版本）
uv run run_tests_launcher.py

# 或使用批次檔
run_tests.bat
```

### 📈 測試套件說明

完整的基準測試包含 **3 個測試套件共 9 個測試案例**，自動生成多種格式的分析報告：

#### **Test Suite 1：參考音檔比較**（2 個測試）
比較不同參考音檔（voice_06.wav vs voice_07.wav）在相同參數下的表現
- 版本：v2
- 分詞方式：token
- 包含 warmup

#### **Test Suite 2：變速策略比較**（4 個測試）⭐ 含音檔輸出
比較四種變速策略的效果差異：
1. **No Speed** - 無變速基準線
2. **Pre-Speed 1.2x** - 預處理加速（加速參考音檔）
3. **Post-Speed 1.2x** - 後處理加速（DSP 時間拉伸）
4. **Hybrid 1.2x** - 混合加速（同時使用兩種）

**輸出**：4 個 WAV 音檔供人工音質評估

#### **Test Suite 3：版本與模式比較**（3 個測試）
比較不同版本和分詞模式：
- v1 streaming
- v2 streaming（token-based）
- v2 streaming（word-based）

### 📁 測試輸出

執行完成後，在 `test_results/` 目錄生成以下檔案：

```
test_results/
├── test_results_YYYYMMDD_HHMMSS.csv           # 📊 CSV 統計表格
├── test_results_YYYYMMDD_HHMMSS.json          # 📝 完整測試日誌
├── performance_comparison_YYYYMMDD_HHMMSS.png # 📈 性能比較圖表
├── efficiency_analysis_YYYYMMDD_HHMMSS.png    # 📈 效率分析圖表
├── summary_report_YYYYMMDD_HHMMSS.txt         # 📄 摘要報告
└── audio_samples/                              # 🎵 音檔樣本（Suite 2）
    ├── voice_07_no_speed.wav
    ├── voice_07_pre_speed_1.2x.wav
    ├── voice_07_post_speed_1.2x.wav
    └── voice_07_hybrid_speed_1.2x.wav
```

### 📊 測試數據指標

#### **性能比較圖表**（performance_comparison）
![Performance Comparison](test_results/performance_comparison_example.png)

1. **TTFB（Time To First Byte）** - 首次響應時間
   - 測量從開始到第一個音訊片段的延遲
   - ✅ 優秀：<3s | ⚠️ 可接受：<5s | ❌ 需優化：>5s

2. **Total Generation Time** - 總生成時間
   - 完整音訊生成所需的總時間
   - 越短越好，影響整體效率

3. **Average Generation Rate** - 平均生成倍率
   - 音訊長度 / 生成時間的比值
   - ✅ 優秀：>2.0x | ✅ 良好：>1.0x | ❌ 不足：<1.0x
   - **必須 >1.0x 才能實時串流**

#### **效率分析圖表**（efficiency_analysis）
![Efficiency Analysis](test_results/efficiency_analysis_example.png)

1. **Overall RTF（Real-Time Factor）** - 整體即時因子
   - 總耗時 / 音訊長度
   - ✅ 優秀：<0.5 | ✅ 良好：<1.0 | ⚠️ 可接受：<1.5

2. **Parallel Efficiency** - 並行效率
   - 生成與播放重疊的百分比
   - ✅ 優秀：>80% | ✅ 良好：>60% | ⚠️ 需改善：<60%
   - 高並行效率表示串流效果好

3. **Memory Usage** - 記憶體使用
   - 峰值記憶體使用量（MB）
   - 監控資源消耗情況

### 📋 CSV 數據欄位說明

生成的 CSV 檔案包含以下欄位：

| 欄位 | 說明 | 單位 |
|------|------|------|
| `test_name` | 測試案例名稱 | - |
| `ttfb` | 首次響應時間 | 秒（s）|
| `total_time` | 總生成時間 | 秒（s）|
| `avg_gen_rate` | 平均生成倍率 | 倍速（x）|
| `max_gen_rate` | 最大生成倍率 | 倍速（x）|
| `min_gen_rate` | 最小生成倍率 | 倍速（x）|
| `overall_rtf` | 整體 RTF | - |
| `parallel_efficiency` | 並行效率 | 百分比（%）|
| `total_chunks` | 總音訊片段數 | 個 |
| `total_audio_duration` | 總音訊長度 | 秒（s）|
| `peak_memory_mb` | 峰值記憶體 | MB |
| `avg_memory_mb` | 平均記憶體 | MB |

### 🔧 依賴套件

#### 必要套件
```bash
uv pip install pyrubberband librosa opencc-python-reimplemented sounddevice soundfile torch numpy
```

#### 可選套件（啟用完整功能）
```bash
# 視覺化圖表生成
uv pip install matplotlib

# 記憶體監控
uv pip install psutil
```

**注意**：沒有可選套件也能執行測試，只是會缺少對應的圖表或記憶體數據。

### 📖 詳細文檔

更多測試相關資訊請參考：
- **快速開始**：[QUICKSTART.md](QUICKSTART.md)
- **完整測試指南**：[TEST_GUIDE.md](TEST_GUIDE.md)
- **測試系統說明**：[README_TESTING.md](README_TESTING.md)
- **技術實作細節**：[TESTING_IMPLEMENTATION.md](TESTING_IMPLEMENTATION.md)

### ⏱️ 預估執行時間

- **完整測試**：30-60 分鐘（視硬體性能而定）
- **單一測試**：3-5 分鐘
- **GPU 加速**：顯著提升速度（建議使用 CUDA）

---

## 🔧 使用 `runtime_setup.py` 設定環境

本分支包含自訂的環境初始化系統：

```python
import runtime_setup

# 初始化環境（處理路徑、快取、CUDA 等）
env_paths = runtime_setup.initialize(__file__)
INDEX_TTS_DIR = env_paths["INDEX_TTS_DIR"]
```

**功能說明：**
- 設定 HuggingFace 快取目錄
- 配置 DeepSpeed 環境變數
- 將 FFMPEG 加入 PATH
- 配置 Torch 擴充功能目錄
- 處理 BigVGAN CUDA 外掛路徑

---

## 📦 安裝

### 前置要求
1. 安裝 [uv 套件管理器](https://docs.astral.sh/uv/getting-started/installation/)
2. 複製此儲存庫：
```bash
git clone https://github.com/JonesHong/index-tts.git
cd index-tts
```

### 安裝依賴套件
```bash
# 安裝所有依賴
uv sync --all-extras

# 或不含 DeepSpeed（Windows 使用者）
uv sync --extra webui
```

### 下載模型

**v2（推薦）：**
```bash
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints_v2
```

**v1.5：**
```bash
hf download IndexTeam/IndexTTS-1.5 --local-dir=checkpoints_v1.5
```

### 串流功能的額外依賴
```bash
# 安裝串流功能所需套件
uv pip install pyrubberband sounddevice soundfile opencc-python-reimplemented
```

---

## 🔬 其他測試腳本

### `test_infer.py`
基本推論測試，不含串流功能。

### `prepare_speed_ref_final.py`
用於預處理參考音檔速度調整的工具程式。

---

## 🛠️ 本分支的主要修改

### 1. **串流支援**
- 新增 `indextts/infer_streaming_patch.py` - IndexTTS v1 的串流補丁
- 修改 `indextts/infer_v2.py` - v2 的增強串流支援
- 實作基於片段的音訊生成

### 2. **速度控制**
- 雙階段速度控制（預處理 + 後處理）
- 使用 `pyrubberband` 進行高品質時間拉伸

### 3. **環境管理**
- `runtime_setup.py` - 集中式環境初始化
- 自動快取目錄管理
- Windows 相容的 DeepSpeed 配置

### 4. **模型修改**
- `indextts/gpt/model_v2.py` - 串流的自訂修改
- `indextts/s2mel/modules/commons.py` - 增強相容性
- `indextts/s2mel/modules/diffusion_transformer.py` - 性能優化

### 5. **測試與工具**
- 全面的串流測試套件
- 性能基準測試工具
- 音訊預處理工具

---

## 📊 性能提示

1. **使用 FP16 加快推論速度：**
   ```python
   tts = IndexTTS2(use_fp16=True, ...)
   ```

2. **啟用 warmup 以獲得一致的延遲：**
   ```bash
   uv run test_streaming.py --warmup
   ```

3. **調整串流的片段長度：**
   - 較小的片段 = 較低延遲，較多開銷
   - 較大的片段 = 較高延遲，較佳效率

4. **速度控制建議：**
   - 自然語速：`--pre_speed_ref 1.0 --speed 1.0`
   - 加快播放：`--speed 1.2` 到 `1.5`
   - TTS 學習較快語速：`--pre_speed_ref 1.2` 到 `1.3`

---

## 🐛 已知問題與修復

### 問題：Windows 上出現 `DLL load failed`
**解決方案：**確保已安裝 CUDA Toolkit 12.8+ 且 `torch/lib` 在 PATH 中（由 `runtime_setup.py` 處理）

### 問題：串流性能緩慢
**解決方案：**
- 啟用 `--warmup` 旗標
- v2 使用 `--method token`（自動分割）
- 減少片段長度
- 使用 FP16 模式

---

## 📝 授權

本分支維持與原專案相同的授權。詳見 [LICENSE](LICENSE)。

---

## 🤝 貢獻

這是用於 LiveKit 整合的個人分支。若要對主專案做出貢獻，請訪問[官方儲存庫](https://github.com/index-tts/index-tts)。

---

## 📧 聯絡

有關本分支的問題：
- GitHub Issues：https://github.com/JonesHong/index-tts/issues

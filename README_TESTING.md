# IndexTTS 串流測試系統

全面的 IndexTTS 性能測試工具集 - 支持多維度比較分析和結果視覺化

## 🚀 快速開始

```bash
# 方法 1: 使用 Bash 腳本 (推薦)
./run_tests.sh

# 方法 2: 直接執行 Python
python run_comprehensive_tests.py

# 方法 3: 單獨測試
python test_streaming_with_output.py \
  --version v2 \
  --ref_audio examples/voice_07.wav \
  --output my_test.wav \
  --warmup
```

## 📦 系統組成

### 核心腳本

| 文件 | 功能 | 說明 |
|------|------|------|
| `test_streaming_with_output.py` | 單一測試執行 | 支持音檔輸出的增強版測試腳本 |
| `run_comprehensive_tests.py` | 全面測試運行器 | 自動執行 9 個測試並生成報告 |
| `run_tests.sh` | 一鍵執行腳本 | Bash 腳本，自動檢查環境和依賴 |

### 文檔

| 文件 | 內容 | 適合對象 |
|------|------|----------|
| `README_TESTING.md` | 系統概覽 (本文件) | 所有人 |
| `QUICKSTART.md` | 快速參考卡片 | 快速上手 |
| `TEST_GUIDE.md` | 完整測試指南 | 深入了解 |
| `TESTING_IMPLEMENTATION.md` | 實施總結 | 技術細節 |

## 🎯 測試套件

### Test Suite 1: Voice Comparison
- **測試數量**: 2
- **比較對象**: voice_06.wav vs voice_07.wav
- **參數**: 相同 (v2, token, warmup)
- **輸出音檔**: ❌

### Test Suite 2: Speed Strategy Comparison
- **測試數量**: 4
- **比較策略**:
  - No speed (baseline)
  - Pre-speed 1.2x (參考音檔加速)
  - Post-speed 1.2x (播放加速)
  - Hybrid 1.2x (混合加速)
- **輸出音檔**: ✅ 4 個 WAV 文件

### Test Suite 3: Version & Mode Comparison
- **測試數量**: 3
- **比較版本**: V1 vs V2 (token) vs V2 (word)
- **輸出音檔**: ❌

**總計**: 9 個測試，約 30-60 分鐘完成

## 📊 測試維度

| 維度 | 指標 | 目標值 | 重要性 |
|------|------|--------|--------|
| 首次響應 | TTFB | <3s | ⭐⭐⭐⭐⭐ |
| 生成速度 | Gen Rate | >1.0x | ⭐⭐⭐⭐⭐ |
| 整體效率 | RTF | <1.0 | ⭐⭐⭐⭐ |
| 串流效果 | 並行效率 | >60% | ⭐⭐⭐ |
| 資源使用 | 記憶體 | <500MB | ⭐⭐⭐ |
| 音質評估 | 人工聆聽 | - | ⭐⭐⭐⭐⭐ |

## 📁 輸出結果

執行測試後，在 `test_results/` 目錄生成:

```
test_results/
├── test_results_20250127_143022.csv           # 📊 CSV 數據表格
├── test_results_20250127_143022.json          # 📝 JSON 詳細日誌
├── performance_comparison_20250127_143022.png # 📈 性能比較圖
├── efficiency_analysis_20250127_143022.png    # 📈 效率分析圖
├── summary_report_20250127_143022.txt         # 📄 文字摘要
└── audio_samples/                              # 🎵 音檔樣本
    ├── voice_07_no_speed.wav
    ├── voice_07_pre_speed_1.2x.wav
    ├── voice_07_post_speed_1.2x.wav
    └── voice_07_hybrid_speed_1.2x.wav
```

## 💻 安裝依賴

### 必要依賴

```bash
pip install pyrubberband librosa opencc-python-reimplemented sounddevice soundfile torch
```

### 可選依賴

```bash
# 視覺化圖表生成
pip install matplotlib

# 記憶體監控
pip install psutil
```

**注意**: 沒有可選依賴也能執行，只是缺少對應功能

## 📈 使用場景

### 場景 1: 性能基準測試
```bash
# 執行所有測試，建立性能基準
./run_tests.sh

# 記錄硬體資訊
echo "GPU: RTX 3090, CUDA 11.8" > baseline_hardware.txt

# 保存結果
cp test_results/test_results_*.csv baseline_results.csv
```

### 場景 2: 音質評估
```bash
# 重點執行 Suite 2
# (編輯 run_comprehensive_tests.py，註解其他套件)

# 人工聆聽 4 個音檔
open test_results/audio_samples/*.wav
```

### 場景 3: 版本升級驗證
```bash
# 升級前測試
./run_tests.sh
mv test_results before_upgrade/

# 升級後測試
./run_tests.sh
mv test_results after_upgrade/

# 比較結果
diff before_upgrade/summary*.txt after_upgrade/summary*.txt
```

### 場景 4: 自定義測試
```bash
# 單獨測試特定配置
python test_streaming_with_output.py \
  --version v2 \
  --method token \
  --ref_audio examples/my_custom_voice.wav \
  --text "我的自定義測試文本" \
  --pre_speed_ref 1.3 \
  --speed 1.1 \
  --warmup \
  --output my_custom_test.wav
```

## 🔧 進階配置

### 修改測試文本

編輯 `run_comprehensive_tests.py`:

```python
DEFAULT_TEXT = (
    "你的自定義測試文本。"
    "可以包含多段落。"
)
```

### 調整變速倍率

編輯 `run_comprehensive_tests.py` 中的 `TEST_SUITE_2`:

```python
"--pre_speed_ref", "1.3",  # 改為 1.3x
"--speed", "1.3",
```

### 添加新的參考音檔

```python
# 在 TEST_SUITE 中添加
{
    "name": "my_voice_test",
    "args": [
        "--ref_audio", str(REF_AUDIO_DIR / "my_voice.wav"),
        ...
    ]
}
```

### 修改超時時間

編輯 `run_comprehensive_tests.py`:

```python
proc = subprocess.run(
    cmd,
    timeout=1200  # 改為 20 分鐘
)
```

## 🐛 故障排除

### CUDA 不可用

```bash
# 檢查 GPU
nvidia-smi

# 檢查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**解決**: 安裝支持 CUDA 的 PyTorch 版本

### 找不到參考音檔

```bash
# 檢查音檔
ls examples/voice_*.wav
```

**解決**: 確保 `examples/` 目錄下有所需音檔

### matplotlib 警告

```
⚠️  matplotlib 未安裝，將跳過圖表生成
```

**影響**: 沒有 PNG 圖表，但其他功能正常

**解決**: `pip install matplotlib` (可選)

### psutil 警告

```
⚠️  psutil 未安裝，將跳過記憶體監控
```

**影響**: 記憶體數據為空，但其他功能正常

**解決**: `pip install psutil` (可選)

### 測試超時

```
❌ 測試超時 (>600s)
```

**可能原因**:
- CPU 模式運行 (非常慢)
- 系統資源不足
- 文本過長

**解決**:
1. 使用 GPU
2. 減少測試文本
3. 增加超時時間 (見進階配置)

## 📚 文檔導航

**新手入門**:
1. 閱讀本文件 (`README_TESTING.md`)
2. 查看快速參考 (`QUICKSTART.md`)
3. 執行測試 (`./run_tests.sh`)
4. 分析結果

**深入了解**:
1. 完整測試指南 (`TEST_GUIDE.md`)
2. 實施技術細節 (`TESTING_IMPLEMENTATION.md`)

**問題解決**:
1. 故障排除 (本文件)
2. 完整指南的故障排除章節 (`TEST_GUIDE.md`)

## 💡 最佳實踐

### 1. 首次執行

```bash
# 檢查環境
./run_tests.sh  # 會自動檢查依賴

# 如果有問題，手動檢查
python --version  # 確認 Python 3.7+
nvidia-smi       # 確認 GPU
ls examples/     # 確認音檔
```

### 2. 建立基準

```bash
# 首次完整測試
./run_tests.sh

# 保存基準結果
mkdir baseline
cp test_results/* baseline/

# 記錄環境
echo "Date: $(date)" > baseline/environment.txt
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> baseline/environment.txt
echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" >> baseline/environment.txt
```

### 3. 定期測試

```bash
# 週期性執行 (如每週)
./run_tests.sh

# 與基準比較
python -c "
import pandas as pd
baseline = pd.read_csv('baseline/test_results_*.csv')
current = pd.read_csv('test_results/test_results_*.csv')
print((current['avg_gen_rate'] / baseline['avg_gen_rate'] - 1) * 100)
"
```

### 4. 結果管理

```bash
# 重要結果重命名保存
cp test_results/test_results_*.csv important_results/v1.0_baseline.csv

# 使用 git 追蹤關鍵結果
git add important_results/*.csv
git commit -m "Add performance baseline v1.0"
```

### 5. 團隊協作

```bash
# 分享測試配置
git add run_comprehensive_tests.py
git commit -m "Update test configuration"

# 分享結果摘要 (不要提交大文件)
git add baseline/summary_*.txt
git add baseline/environment.txt
```

## 🎯 關鍵指標解讀

### TTFB (首次響應時間)

```
✅ 優秀: <3s
✅ 良好: <5s
⚠️  可接受: <8s
❌ 需優化: >8s
```

**影響**: 使用者感知延遲，越低越好

### 生成倍率 (Generation Rate)

```
✅ 優秀: >2.0x (2倍實時速度)
✅ 良好: >1.0x (超過實時)
⚠️  邊緣: 0.8-1.0x (接近實時)
❌ 不足: <0.8x (無法實時)
```

**影響**: 能否順暢串流，必須 >1.0x

### 整體 RTF (Real-Time Factor)

```
✅ 優秀: <0.5
✅ 良好: <1.0 (總耗時小於音訊長度)
⚠️  可接受: 1.0-1.5
❌ 需優化: >1.5
```

**影響**: 整體效率，越低越好

### 並行效率 (Parallel Efficiency)

```
✅ 優秀: >80% (高度重疊)
✅ 良好: >60% (良好重疊)
⚠️  可接受: 40-60%
❌ 需優化: <40% (串流效果差)
```

**影響**: 串流順暢度

## 📞 支援

### 常見問題

查看 `TEST_GUIDE.md` 的故障排除章節

### 技術細節

查看 `TESTING_IMPLEMENTATION.md`

### 快速參考

查看 `QUICKSTART.md`

---

**版本**: 1.0.0
**最後更新**: 2025-01-27
**維護者**: IndexTTS Testing Team
**授權**: MIT

**快速連結**:
- [快速開始](QUICKSTART.md)
- [完整指南](TEST_GUIDE.md)
- [實施細節](TESTING_IMPLEMENTATION.md)

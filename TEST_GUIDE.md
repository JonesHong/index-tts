# IndexTTS 串流測試指南

全面的 IndexTTS 性能測試工具集，支持多維度比較分析和結果視覺化。

## 📋 測試概覽

### 測試套件

**Test Suite 1: Voice Comparison (音色比較)**
- 比較 `voice_06.wav` vs `voice_07.wav` 的基準性能
- 相同參數 (v2, token mode, 預熱)
- 評估不同參考音檔對性能的影響

**Test Suite 2: Speed Strategy Comparison (變速策略比較)**
- 使用 `voice_07.wav` 作為參考音檔
- 比較 4 種策略:
  1. 無變速 (baseline)
  2. 預處理加速 1.2x (--pre_speed_ref 1.2)
  3. 後處理加速 1.2x (--speed 1.2)
  4. 混合加速 1.2x (pre + post)
- **包含音檔輸出**: 每個測試生成 WAV 文件供人工評估

**Test Suite 3: Version & Mode Comparison (版本與模式比較)**
- 使用 `voice_07.wav` 作為參考音檔
- 比較:
  1. V1 串流模式
  2. V2 串流模式 (token-based)
  3. V2 串流模式 (word-based)

### 測試維度

測試腳本會自動收集以下指標:

1. **首次響應時間 (TTFB)** - 從開始到第一個音訊片段的時間
2. **總生成時間** - 完整測試的總耗時
3. **生成倍率** - 平均/最大/最小 (audio_duration / generation_time)
4. **整體實時率 (RTF)** - 總生成時間 / 總音訊長度 (越低越好)
5. **並行效率** - 生成與播放重疊的百分比
6. **記憶體使用** - 測試期間的記憶體消耗 (需要 psutil)
7. **參考音檔資訊** - 檔案大小、時長、採樣率等
8. **模型載入與預熱時間**

## 🚀 快速開始

### 前置需求

```bash
# 基本依賴 (已在 test_streaming.py 中使用)
pip install pyrubberband librosa opencc-python-reimplemented sounddevice soundfile

# 可選依賴 (用於視覺化和記憶體監控)
pip install matplotlib psutil
```

### 執行完整測試

```bash
cd /mnt/c/work/livekit_node/libs/index-tts

# 執行所有測試套件 (大約 30-60 分鐘)
python run_comprehensive_tests.py
```

### 檢查結果

測試完成後，在 `test_results/` 目錄下會生成:

```
test_results/
├── test_results_20250127_143022.csv           # CSV 數據表格
├── test_results_20250127_143022.json          # JSON 詳細日誌
├── performance_comparison_20250127_143022.png # 性能比較圖表
├── efficiency_analysis_20250127_143022.png    # 效率分析圖表
├── summary_report_20250127_143022.txt         # 文字摘要報告
└── audio_samples/                              # 生成的音檔樣本
    ├── voice_07_no_speed.wav
    ├── voice_07_pre_speed_1.2x.wav
    ├── voice_07_post_speed_1.2x.wav
    └── voice_07_hybrid_speed_1.2x.wav
```

## 📊 輸出說明

### CSV 數據表格

包含所有測試的結構化數據，可用於 Excel/Pandas 分析:

```csv
test_id,name,description,timestamp,execution_time,memory_usage_mb,error,ttfb,total_time,avg_gen_rate,...
Suite1_Voice_Comparison_01,voice_06_baseline,"Voice 06 - Baseline",2025-01-27T14:30:22,45.23,123.4,,2.15,43.87,1.85,...
```

### JSON 詳細日誌

包含完整的測試輸出和元數據:

```json
{
  "Suite1_Voice_Comparison": [
    {
      "test_id": "Suite1_Voice_Comparison_01",
      "name": "voice_06_baseline",
      "description": "Voice 06 - Baseline (Default Parameters)",
      "timestamp": "2025-01-27T14:30:22",
      "metrics": {
        "ttfb": 2.15,
        "total_time": 43.87,
        "avg_gen_rate": 1.85,
        ...
      },
      "raw_output": "...",
      "memory_usage_mb": 123.4
    }
  ]
}
```

### 視覺化圖表

**圖表 1: `performance_comparison_*.png`**
- 3 個子圖並排比較
- TTFB (越低越好)
- 總生成時間 (越低越好)
- 平均生成倍率 (越高越好，>1.0x 表示比實時快)

**圖表 2: `efficiency_analysis_*.png`**
- 3 個子圖並排比較
- 整體 RTF (越低越好，<1.0 表示實時以內)
- 並行效率百分比 (越高越好)
- 記憶體使用 (MB)

### 音檔樣本

Test Suite 2 會生成 4 個 WAV 文件供人工音質評估:

- **no_speed**: 原始速度生成
- **pre_speed_1.2x**: TTS 模仿加速後的語速
- **post_speed_1.2x**: DSP 時間拉伸後的音檔
- **hybrid_1.2x**: 同時應用兩種加速

## 🔧 進階用法

### 單獨測試腳本

如果只想測試單一配置並保存音檔:

```bash
# 使用 test_streaming_with_output.py
python test_streaming_with_output.py \
  --version v2 \
  --method token \
  --ref_audio examples/voice_07.wav \
  --text "你的測試文本" \
  --pre_speed_ref 1.2 \
  --speed 1.2 \
  --warmup \
  --output my_test_output.wav
```

### 自定義測試配置

編輯 `run_comprehensive_tests.py` 中的測試套件配置:

```python
# 添加新的測試案例
TEST_SUITE_CUSTOM = [
    {
        "name": "my_custom_test",
        "description": "My Custom Test Configuration",
        "args": [
            "--version", "v2",
            "--method", "word",  # 改用 word-based
            "--ref_audio", str(REF_AUDIO_DIR / "my_voice.wav"),
            "--text", "自定義測試文本",
            "--pre_speed_ref", "1.5",
            "--warmup",
            "--output", str(AUDIO_OUTPUT_DIR / "my_custom.wav")
        ]
    }
]
```

然後在 `main()` 中添加執行:

```python
# 在 main() 函數中
custom_results = run_test_suite("Suite_Custom", TEST_SUITE_CUSTOM)
all_results["Suite_Custom"] = custom_results
```

### 修改測試文本

在 `run_comprehensive_tests.py` 頂部修改 `DEFAULT_TEXT`:

```python
DEFAULT_TEXT = (
    "你的自定義測試文本。"
    "可以是多段落。"
)
```

### 調整變速倍率

修改 Test Suite 2 中的 `--pre_speed_ref` 和 `--speed` 值:

```python
"--pre_speed_ref", "1.3",  # 從 1.2 改為 1.3
"--speed", "1.3",
```

## 📈 結果分析建議

### 性能分析

1. **TTFB (首次響應時間)**
   - 關鍵指標: 使用者感知延遲
   - 目標: <3s 為良好, <5s 為可接受
   - 影響因素: 模型載入、預熱、參考音檔處理

2. **生成倍率 (Generation Rate)**
   - 關鍵指標: 是否能實時生成
   - 目標: >1.0x (比播放速度快)
   - 影響因素: GPU 性能、模型版本、文本長度

3. **整體 RTF (Real-Time Factor)**
   - 關鍵指標: 整體效率
   - 目標: <1.0 (總生成時間小於音訊長度)
   - 計算: 總生成時間 / 總音訊長度

4. **並行效率**
   - 關鍵指標: 串流效果
   - 目標: >80% (大部分時間在生成時同時播放)
   - 影響: 串流切分策略、chunk 大小

### 變速策略分析

比較 Test Suite 2 的結果:

- **音質**: 人工聆聽 4 個 WAV 文件，評估清晰度和自然度
- **性能**: 比較 TTFB 和總生成時間
- **效果**: 預加速影響 TTS 模仿速度，後處理加速影響播放速度

建議評估標準:

| 策略 | 音質 | TTFB | 總耗時 | 推薦場景 |
|------|------|------|--------|----------|
| No Speed | 最佳 | 基準 | 基準 | 質量優先 |
| Pre-Speed | 好 | 相似 | 相似 | 自然加速 |
| Post-Speed | 中 | 相似 | 稍慢 | 快速輸出 |
| Hybrid | 中 | 相似 | 稍慢 | 極速場景 |

### 版本比較分析

比較 Test Suite 3 的結果:

- **V1 vs V2**: 性能差異、音質差異
- **Token vs Word**: 切分策略對串流的影響

## 🐛 故障排除

### matplotlib 未安裝

```
⚠️  matplotlib 未安裝，將跳過圖表生成
```

**解決方案**:
```bash
pip install matplotlib
```

圖表生成會被跳過，但 CSV/JSON 仍會正常生成。

### psutil 未安裝

```
⚠️  psutil 未安裝，將跳過記憶體監控
```

**解決方案**:
```bash
pip install psutil
```

記憶體使用數據會顯示為空，但不影響其他指標。

### 找不到參考音檔

```
❌ 錯誤: 找不到參考音檔: voice_06.wav, voice_07.wav
```

**解決方案**:
確保 `examples/` 目錄下有對應的音檔文件:
```bash
ls examples/voice_*.wav
```

### 測試超時

```
❌ 測試超時 (>600s)
```

**原因**: 單個測試超過 10 分鐘

**解決方案**:
1. 檢查 GPU 是否可用 (`nvidia-smi`)
2. 減少測試文本長度
3. 調整 `run_comprehensive_tests.py` 中的 `timeout` 值

### CUDA 不可用

```
⚠️ CUDA 不可用, 將使用 CPU (速度會受影響)
```

**影響**: 測試會慢很多，但仍能完成

**解決方案**:
- 確認 CUDA 安裝: `nvidia-smi`
- 確認 PyTorch CUDA 版本: `python -c "import torch; print(torch.cuda.is_available())"`

## 📝 注意事項

1. **測試時間**: 完整測試約需 30-60 分鐘，取決於硬體性能
2. **磁碟空間**: 音檔樣本約需 50-100 MB 空間
3. **並行執行**: 測試之間有 5 秒延遲以釋放資源
4. **輸出覆蓋**: 每次執行會生成新的時間戳記文件，不會覆蓋舊結果

## 📚 相關文檔

- `test_streaming.py` - 原始測試腳本 (無音檔輸出)
- `test_streaming_with_output.py` - 帶音檔輸出的測試腳本
- `run_comprehensive_tests.py` - 全面測試運行器
- `LIVEKIT_FASTER_WHISPER_STREAMING_SPEC.md` - LiveKit 整合規格

## 💡 最佳實踐

1. **基準測試**: 先跑 Test Suite 1 確定基準性能
2. **音質評估**: Test Suite 2 生成的音檔務必人工聆聽
3. **結果保存**: 重要的測試結果建議重命名保存，避免被新測試覆蓋
4. **版本管理**: 在 git commit 中記錄測試配置和結果摘要
5. **硬體標註**: 在報告中註明硬體配置 (GPU型號、VRAM、CUDA版本)

## 🔗 支援

如遇問題請檢查:
1. 所有依賴是否正確安裝
2. 參考音檔路徑是否正確
3. 模型檔案是否完整 (`checkpoints_v1.5/`, `checkpoints_v2/`)
4. CUDA 是否可用並正確配置

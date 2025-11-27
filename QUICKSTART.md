# IndexTTS 測試快速參考

## 🚀 一鍵執行

```bash
cd /mnt/c/work/livekit_node/libs/index-tts
./run_tests.sh
```

**或直接執行 Python:**

```bash
python run_comprehensive_tests.py
```

## 📦 安裝依賴

### 必要依賴

```bash
pip install pyrubberband librosa opencc-python-reimplemented sounddevice soundfile torch
```

### 可選依賴 (視覺化和記憶體監控)

```bash
pip install matplotlib psutil
```

## 📊 測試套件

| 套件 | 描述 | 測試數 | 音檔輸出 |
|------|------|--------|----------|
| Suite 1 | Voice 06 vs 07 比較 | 2 | ❌ |
| Suite 2 | 變速策略比較 | 4 | ✅ |
| Suite 3 | 版本與模式比較 | 3 | ❌ |

**總計: 9 個測試**

## 📁 輸出文件

```
test_results/
├── test_results_YYYYMMDD_HHMMSS.csv           # 數據表格
├── test_results_YYYYMMDD_HHMMSS.json          # 詳細日誌
├── performance_comparison_YYYYMMDD_HHMMSS.png # 圖表 1
├── efficiency_analysis_YYYYMMDD_HHMMSS.png    # 圖表 2
├── summary_report_YYYYMMDD_HHMMSS.txt         # 摘要
└── audio_samples/                              # 音檔
    ├── voice_07_no_speed.wav
    ├── voice_07_pre_speed_1.2x.wav
    ├── voice_07_post_speed_1.2x.wav
    └── voice_07_hybrid_speed_1.2x.wav
```

## 🎯 關鍵指標

### TTFB (首次響應時間)
- **優秀**: <3s
- **良好**: <5s
- **需優化**: >5s

### 生成倍率 (Generation Rate)
- **優秀**: >2.0x
- **良好**: >1.0x (實時以上)
- **需優化**: <1.0x (無法實時)

### 整體 RTF (Real-Time Factor)
- **優秀**: <0.5
- **良好**: <1.0 (實時以內)
- **需優化**: >1.0 (超過實時)

### 並行效率 (Parallel Efficiency)
- **優秀**: >80%
- **良好**: >60%
- **需優化**: <60%

## 🔧 單獨測試

```bash
# 基本測試
python test_streaming_with_output.py \
  --version v2 \
  --method token \
  --ref_audio examples/voice_07.wav \
  --warmup

# 帶變速測試
python test_streaming_with_output.py \
  --version v2 \
  --method token \
  --ref_audio examples/voice_07.wav \
  --pre_speed_ref 1.2 \
  --speed 1.2 \
  --warmup \
  --output my_output.wav

# 完整參數
python test_streaming_with_output.py \
  --version v2 \
  --method token \
  --ref_audio examples/voice_07.wav \
  --text "你的測試文本" \
  --pre_speed_ref 1.2 \
  --speed 1.2 \
  --warmup \
  --output my_output.wav
```

## 📝 變速參數說明

### --pre_speed_ref (預處理加速)
- 加速參考音檔後再送給 TTS
- TTS 會模仿加速後的語速
- 影響生成音訊的語速特徵

### --speed (後處理加速)
- 生成後用 DSP 時間拉伸
- 不影響 TTS 生成過程
- 可能影響音質

### 混合使用
- 同時使用兩個參數
- 獲得最大加速效果
- 音質可能受影響最大

## ⚡ 故障排除

### CUDA 不可用
```bash
# 檢查 CUDA
nvidia-smi

# 檢查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 找不到參考音檔
```bash
# 檢查音檔
ls examples/voice_*.wav

# 應該看到:
# voice_03.wav  voice_06.wav  voice_07.wav  voice_11.wav ...
```

### matplotlib 警告
```bash
# 安裝視覺化依賴
pip install matplotlib

# 不安裝也可以，只是沒有圖表
```

### psutil 警告
```bash
# 安裝記憶體監控
pip install psutil

# 不安裝也可以，只是沒有記憶體數據
```

## 📚 更多資訊

- **完整指南**: [TEST_GUIDE.md](TEST_GUIDE.md)
- **原始測試腳本**: `test_streaming.py`
- **帶輸出版本**: `test_streaming_with_output.py`
- **測試運行器**: `run_comprehensive_tests.py`

## 💡 提示

1. **首次運行**: 建議先跑一個快速測試確認環境正常
2. **完整測試**: 預留 30-60 分鐘時間
3. **結果保存**: 重要結果建議重命名保存
4. **音質評估**: Suite 2 的音檔務必人工聆聽比較
5. **硬體記錄**: 在報告中註明 GPU 型號和 CUDA 版本

## 🎯 使用場景

### 場景 1: 快速性能檢查
```bash
# 只跑 Suite 1 (修改 run_comprehensive_tests.py)
# 註解掉 Suite 2 和 Suite 3 的執行
```

### 場景 2: 音質對比
```bash
# 只跑 Suite 2，專注音檔品質評估
# 人工聆聽 4 個生成的 WAV
```

### 場景 3: 版本升級驗證
```bash
# 跑 Suite 3，比較 v1 vs v2 差異
```

### 場景 4: 自定義測試
```bash
# 編輯測試配置
# 修改參考音檔、文本、參數
# 執行單獨測試
```

## 📊 結果查看順序

1. **摘要報告** (`summary_report_*.txt`)
   - 快速了解所有測試結果

2. **視覺化圖表** (`*.png`)
   - 直觀比較性能差異

3. **CSV 數據** (`*.csv`)
   - Excel/Numbers 打開做深入分析

4. **音檔樣本** (`audio_samples/*.wav`)
   - 人工評估音質差異

5. **JSON 日誌** (`*.json`)
   - 需要時查看完整輸出

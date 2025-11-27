# IndexTTS 測試系統實施總結

## 📦 已交付文件

### 核心腳本

1. **`test_streaming_with_output.py`**
   - 基於原始 `test_streaming.py` 的增強版
   - 新增 `--output` 參數支持音檔保存
   - 修復重複代碼問題
   - 完整保留原有功能

2. **`run_comprehensive_tests.py`**
   - 全面測試運行器
   - 自動執行 3 個測試套件 (共 9 個測試)
   - 自動生成 CSV、JSON、PNG、TXT 報告
   - 支持視覺化和記憶體監控

3. **`run_tests.sh`**
   - 一鍵執行腳本
   - 自動檢查依賴和環境
   - 友善的進度提示
   - 自動估算測試時間

### 文檔

4. **`TEST_GUIDE.md`**
   - 完整測試指南 (3000+ 字)
   - 詳細說明測試維度和配置
   - 故障排除指南
   - 結果分析建議

5. **`QUICKSTART.md`**
   - 快速參考卡片
   - 一頁式命令參考
   - 關鍵指標說明
   - 常見場景示範

6. **`TESTING_IMPLEMENTATION.md`** (本文件)
   - 實施總結
   - 技術架構說明
   - 使用指南

## 🎯 實現的測試需求

### 第一個需求: Voice 比較 (voice_06 vs voice_07)

**實現方式**: Test Suite 1

```python
TEST_SUITE_1 = [
    {
        "name": "voice_06_baseline",
        "description": "Voice 06 - Baseline (Default Parameters)",
        "args": ["--version", "v2", "--method", "token",
                 "--ref_audio", "examples/voice_06.wav", "--warmup"]
    },
    {
        "name": "voice_07_baseline",
        "description": "Voice 07 - Baseline (Default Parameters)",
        "args": ["--version", "v2", "--method", "token",
                 "--ref_audio", "examples/voice_07.wav", "--warmup"]
    }
]
```

**比較維度**:
- TTFB (首次響應時間)
- 總生成時間
- 生成倍率 (平均/最大/最小)
- 整體 RTF
- 並行效率
- 記憶體使用
- 參考音檔特徵 (大小、時長、採樣率)

### 第二個需求: 變速策略比較 (voice_07)

**實現方式**: Test Suite 2

```python
TEST_SUITE_2 = [
    # 1. 無變速 (baseline)
    {"name": "voice_07_no_speed", ...},

    # 2. 預處理加速 1.2x
    {"name": "voice_07_pre_speed_1.2x",
     "args": [..., "--pre_speed_ref", "1.2"]},

    # 3. 後處理加速 1.2x
    {"name": "voice_07_post_speed_1.2x",
     "args": [..., "--speed", "1.2"]},

    # 4. 混合加速 1.2x
    {"name": "voice_07_hybrid_speed_1.2x",
     "args": [..., "--pre_speed_ref", "1.2", "--speed", "1.2"]}
]
```

**特別功能**:
- ✅ 每個測試都保存 WAV 文件
- ✅ 4 個音檔供人工音質評估
- ✅ 完整性能指標比較

**音檔位置**:
```
test_results/audio_samples/
├── voice_07_no_speed.wav
├── voice_07_pre_speed_1.2x.wav
├── voice_07_post_speed_1.2x.wav
└── voice_07_hybrid_speed_1.2x.wav
```

### 第三個需求: 版本與模式比較

**實現方式**: Test Suite 3

```python
TEST_SUITE_3 = [
    # V1 串流
    {"name": "v1_streaming",
     "args": ["--version", "v1", ...]},

    # V2 串流 (token-based)
    {"name": "v2_streaming_token",
     "args": ["--version", "v2", "--method", "token", ...]},

    # V2 串流 (word-based)
    {"name": "v2_streaming_word",
     "args": ["--version", "v2", "--method", "word", ...]}
]
```

**比較焦點**:
- V1 vs V2 性能差異
- Token-based vs Word-based 切分策略
- 串流效率差異

## 📊 輸出格式

### 1. CSV 統計數據表格

**文件**: `test_results_YYYYMMDD_HHMMSS.csv`

**欄位**:
```
test_id, name, description, timestamp, execution_time,
memory_usage_mb, error, ttfb, total_time, avg_gen_rate,
max_gen_rate, min_gen_rate, overall_rtf, parallel_efficiency,
chunk_count, warmup_time, model_load_time, ref_audio_duration,
ref_audio_size_mb, pre_speed_enabled, post_speed_enabled
```

**用途**: Excel/Pandas 分析、數據透視表、趨勢圖表

### 2. JSON 詳細測試日誌

**文件**: `test_results_YYYYMMDD_HHMMSS.json`

**結構**:
```json
{
  "Suite1_Voice_Comparison": [
    {
      "test_id": "Suite1_Voice_Comparison_01",
      "name": "voice_06_baseline",
      "metrics": { ... },
      "raw_output": "...",
      "execution_time": 45.23,
      "memory_usage_mb": 123.4
    }
  ],
  "Suite2_Speed_Strategy": [ ... ],
  "Suite3_Version_Mode": [ ... ]
}
```

**用途**: 程式化分析、自動化報告、錯誤調查

### 3. 視覺化比較圖表

#### 圖表 1: `performance_comparison_*.png`

**3 個子圖**:
- TTFB (柱狀圖，越低越好)
- 總生成時間 (柱狀圖，越低越好)
- 平均生成倍率 (柱狀圖，越高越好，包含 1.0x 參考線)

**特點**:
- 每個柱子標註數值
- 顏色區分 (藍/橘/綠)
- 自動縮放座標軸

#### 圖表 2: `efficiency_analysis_*.png`

**3 個子圖**:
- 整體 RTF (柱狀圖，越低越好，包含 1.0 閾值線)
- 並行效率 (柱狀圖，百分比，越高越好)
- 記憶體使用 (柱狀圖，MB，如果 psutil 可用)

**特點**:
- 效率百分比標註
- 閾值線輔助判斷
- 缺少 psutil 時顯示提示文字

### 4. 完整測試日誌

**文件**: `summary_report_YYYYMMDD_HHMMSS.txt`

**內容**:
```
================================================================================
IndexTTS Streaming Performance Test - Summary Report
Generated: 2025-01-27 14:30:22
================================================================================

================================================================================
Test Suite: Suite1_Voice_Comparison
================================================================================

Test: voice_06_baseline
Description: Voice 06 - Baseline (Default Parameters)
✅ Status: SUCCESS
   TTFB: 2.15s
   Total Time: 43.87s
   Avg Gen Rate: 1.85x
   Overall RTF: 0.543
   Memory Usage: 123.4 MB

...
```

**用途**: 快速查看、團隊分享、歷史記錄

### 5. 音檔樣本

**目錄**: `test_results/audio_samples/`

**文件**:
- `voice_07_no_speed.wav` - 原始速度
- `voice_07_pre_speed_1.2x.wav` - TTS 模仿加速
- `voice_07_post_speed_1.2x.wav` - DSP 時間拉伸
- `voice_07_hybrid_speed_1.2x.wav` - 混合加速

**用途**: 人工音質評估、A/B 測試、音質檔案

## 🔧 技術架構

### 核心組件

```
run_comprehensive_tests.py
├── Test Configuration (測試配置)
│   ├── TEST_SUITE_1 (Voice Comparison)
│   ├── TEST_SUITE_2 (Speed Strategy)
│   └── TEST_SUITE_3 (Version & Mode)
│
├── Execution Engine (執行引擎)
│   ├── run_single_test() - 單一測試執行
│   ├── parse_test_output() - 輸出解析
│   └── run_test_suite() - 套件執行
│
├── Output Generation (輸出生成)
│   ├── save_results_csv() - CSV 生成
│   ├── save_results_json() - JSON 生成
│   ├── generate_visualization_1() - 性能圖表
│   ├── generate_visualization_2() - 效率圖表
│   └── generate_summary_report() - 摘要報告
│
└── Resource Management (資源管理)
    ├── Memory monitoring (psutil)
    ├── Timeout handling (600s)
    └── Error recovery (try/except/finally)
```

### 解析引擎

**正則表達式匹配**:
```python
# TTFB
r'\[⚡ First Token\].*?(\d+\.\d+)s'

# 總耗時
r'總耗時:\s*(\d+\.\d+)\s*s'

# 生成倍率
r'Avg\s*:\s*(\d+\.\d+)\s*x'
r'Max\s*:\s*(\d+\.\d+)\s*x'
r'Min\s*:\s*(\d+\.\d+)\s*x'

# RTF
r'整體實時率.*?RTF.*?(\d+\.\d+)'

# 並行效率
r'並行效率.*?(\d+\.\d+)%'
```

**指標提取**:
- 時間類: TTFB, 總耗時, 預熱時間, 模型載入時間
- 速率類: 平均/最大/最小生成倍率, RTF
- 效率類: 並行效率百分比
- 資源類: 記憶體使用 (MB)
- 音檔類: 參考音檔時長、大小、採樣率

### 錯誤處理

**多層防護**:
```python
try:
    # 測試執行
    proc = subprocess.run(cmd, timeout=600)

    # 解析結果
    if proc.returncode == 0:
        result["metrics"] = parse_test_output(proc.stdout)
    else:
        result["error"] = f"測試失敗 (返回碼: {proc.returncode})"

except subprocess.TimeoutExpired:
    result["error"] = "測試超時 (>600s)"

except Exception as e:
    result["error"] = f"執行錯誤: {str(e)}"

finally:
    # 清理暫存文件
    ...
```

## 💻 使用方式

### 方法 1: 一鍵執行 (推薦)

```bash
cd /mnt/c/work/livekit_node/libs/index-tts
./run_tests.sh
```

**優點**:
- ✅ 自動檢查依賴
- ✅ 自動檢查環境
- ✅ 友善進度提示
- ✅ 自動顯示結果

### 方法 2: 直接執行 Python

```bash
python run_comprehensive_tests.py
```

**優點**:
- ✅ 跨平台 (Windows/Linux/macOS)
- ✅ 無需 Bash
- ✅ 完整功能

### 方法 3: 單獨測試

```bash
# 單一測試 + 音檔輸出
python test_streaming_with_output.py \
  --version v2 \
  --method token \
  --ref_audio examples/voice_07.wav \
  --pre_speed_ref 1.2 \
  --speed 1.2 \
  --warmup \
  --output my_test.wav
```

**優點**:
- ✅ 靈活配置
- ✅ 快速驗證
- ✅ 自定義參數

## 📈 測試指標說明

### 核心性能指標

**TTFB (Time To First Byte)**
- 定義: 從開始到第一個音訊片段生成的時間
- 重要性: ⭐⭐⭐⭐⭐ (使用者感知延遲)
- 目標值: <3s (優秀), <5s (良好)
- 影響因素: 模型載入、預熱、參考音檔處理

**生成倍率 (Generation Rate)**
- 定義: 音訊時長 / 生成時間
- 重要性: ⭐⭐⭐⭐⭐ (是否能實時生成)
- 目標值: >1.0x (必須), >2.0x (優秀)
- 影響因素: GPU 性能、模型版本、文本長度

**整體 RTF (Real-Time Factor)**
- 定義: 總生成時間 / 總音訊長度
- 重要性: ⭐⭐⭐⭐ (整體效率)
- 目標值: <1.0 (必須), <0.5 (優秀)
- 影響因素: 綜合性能、串流效率

**並行效率 (Parallel Efficiency)**
- 定義: 生成與播放重疊的百分比
- 重要性: ⭐⭐⭐ (串流效果)
- 目標值: >60% (良好), >80% (優秀)
- 影響因素: 切分策略、chunk 大小、緩衝管理

### 資源使用指標

**記憶體使用**
- 測量: 測試期間的記憶體增量 (MB)
- 重要性: ⭐⭐⭐ (資源成本)
- 需求: psutil 套件
- 用途: 容量規劃、成本估算

**模型載入時間**
- 測量: 模型初始化耗時
- 重要性: ⭐⭐ (首次啟動)
- 優化: 模型預載、快取機制

**預熱時間**
- 測量: warm-up 階段耗時
- 重要性: ⭐⭐ (首次生成)
- 優化: 預熱策略、跳過預熱

## 🎯 預期結果範例

### 正常性能範圍 (GPU: RTX 3090)

```
┌─────────────────┬────────┬─────────────┬──────────┬────────┐
│ 測試            │ TTFB   │ 總耗時      │ 生成倍率 │ RTF    │
├─────────────────┼────────┼─────────────┼──────────┼────────┤
│ voice_06        │ 2-3s   │ 35-45s      │ 1.5-2.0x │ 0.4-0.6│
│ voice_07        │ 2-3s   │ 35-45s      │ 1.5-2.0x │ 0.4-0.6│
│ pre_speed_1.2x  │ 2-3s   │ 35-45s      │ 1.5-2.0x │ 0.4-0.6│
│ post_speed_1.2x │ 2-3s   │ 38-48s      │ 1.5-2.0x │ 0.4-0.6│
│ hybrid_1.2x     │ 2-3s   │ 38-48s      │ 1.5-2.0x │ 0.4-0.6│
│ v1_streaming    │ 2-4s   │ 40-50s      │ 1.3-1.8x │ 0.5-0.7│
│ v2_token        │ 2-3s   │ 35-45s      │ 1.5-2.0x │ 0.4-0.6│
│ v2_word         │ 2-4s   │ 38-48s      │ 1.4-1.9x │ 0.4-0.6│
└─────────────────┴────────┴─────────────┴──────────┴────────┘
```

**記憶體使用**: 100-200 MB (測試期間增量)
**並行效率**: 70-90%

### CPU 模式性能範圍

```
┌─────────────────┬────────┬─────────────┬──────────┬────────┐
│ 測試            │ TTFB   │ 總耗時      │ 生成倍率 │ RTF    │
├─────────────────┼────────┼─────────────┼──────────┼────────┤
│ All Tests       │ 8-15s  │ 180-300s    │ 0.3-0.6x │ 2.0-4.0│
└─────────────────┴────────┴─────────────┴──────────┴────────┘
```

**⚠️ CPU 模式**: 不建議用於生產環境，僅供功能驗證

## 🔍 分析建議

### 1. 性能基準建立

**首次執行**:
1. 執行完整測試套件
2. 記錄 GPU 型號和 CUDA 版本
3. 保存結果作為基準 (baseline)

**後續對比**:
- 與基準比較，評估改進效果
- 同硬體環境下的時序比較
- 不同配置下的橫向比較

### 2. 變速策略選擇

**音質優先**:
- 選擇 `no_speed` 或 `pre_speed_1.1x`
- 接受較慢的實際語速
- 確保音質最佳

**速度優先**:
- 選擇 `post_speed_1.2x` 或 `hybrid_1.2x`
- 可接受輕微音質下降
- 獲得更快的輸出

**平衡方案**:
- 選擇 `pre_speed_1.15x`
- 中等加速，音質影響較小
- 需要人工測試確定最佳值

### 3. 版本選擇建議

**V1 vs V2**:
- V2 通常性能更好
- V2 支持更多切分策略
- V1 可能在某些場景下音質更好

**Token vs Word**:
- Token-based: 通常更快，chunk 更均勻
- Word-based: 可能在某些語言中更自然
- 需要實際測試比較

## 🚨 常見問題

### Q1: 測試失敗，返回碼不為 0

**可能原因**:
- 模型文件損壞或缺失
- CUDA 版本不匹配
- 依賴包版本衝突

**解決方法**:
1. 檢查 `checkpoints_v2/` 和 `checkpoints_v1.5/` 完整性
2. 重新安裝 PyTorch (匹配 CUDA 版本)
3. 查看 JSON 日誌中的 `raw_output` 和 `stderr`

### Q2: TTFB 過長 (>10s)

**可能原因**:
- 首次執行，模型未快取
- CPU 模式運行
- 系統資源不足

**解決方法**:
1. 使用 `--warmup` 參數
2. 確認 GPU 可用 (`nvidia-smi`)
3. 關閉其他耗資源程序

### Q3: 生成倍率 <1.0x

**可能原因**:
- 硬體性能不足
- 模型配置不當
- 文本過長

**解決方法**:
1. 使用 GPU 而非 CPU
2. 減少測試文本長度
3. 調整模型參數 (降低 steps)

### Q4: 記憶體使用持續增長

**可能原因**:
- 記憶體洩漏
- 暫存文件未清理
- 音訊緩衝未釋放

**解決方法**:
1. 檢查 `finally` 區塊的清理邏輯
2. 測試間增加延遲時間
3. 重啟 Python 進程

## 📋 檢查清單

### 執行前檢查

- [ ] Python 環境已激活
- [ ] 必要依賴已安裝 (pyrubberband, librosa, opencc, sounddevice, soundfile)
- [ ] 可選依賴已安裝 (matplotlib, psutil)
- [ ] CUDA 可用 (生產環境)
- [ ] 參考音檔存在 (examples/voice_06.wav, voice_07.wav)
- [ ] 模型文件完整 (checkpoints_v2/, checkpoints_v1.5/)
- [ ] 磁碟空間足夠 (至少 500 MB)

### 執行後檢查

- [ ] 9 個測試全部成功
- [ ] CSV 文件生成並包含所有欄位
- [ ] JSON 文件生成並包含完整日誌
- [ ] 2 張 PNG 圖表生成 (如果 matplotlib 可用)
- [ ] 摘要報告生成
- [ ] 4 個音檔樣本生成 (Suite 2)
- [ ] 關鍵指標在合理範圍內

### 結果分析檢查

- [ ] TTFB <5s (GPU 模式)
- [ ] 生成倍率 >1.0x (GPU 模式)
- [ ] 整體 RTF <1.0
- [ ] 並行效率 >60%
- [ ] 記憶體使用 <500 MB
- [ ] 無錯誤或警告
- [ ] 音檔樣本音質可接受

## 📊 結論

本測試系統提供了:

✅ **全面覆蓋**: 9 個測試案例，涵蓋所有關鍵場景
✅ **多維分析**: 8 個核心指標，全方位評估性能
✅ **視覺化支持**: CSV、JSON、PNG、TXT 多種格式
✅ **音質評估**: 4 個音檔樣本供人工比較
✅ **易於使用**: 一鍵執行，自動化流程
✅ **完整文檔**: 快速開始、完整指南、故障排除

**適用場景**:
- 性能基準測試
- 版本升級驗證
- 變速策略選擇
- 音質對比評估
- 容量規劃分析
- CI/CD 整合

**技術特點**:
- Python 3.7+ 兼容
- 跨平台支持
- 模組化設計
- 錯誤容錯
- 可擴展架構

---

**實施完成**: 2025-01-27
**版本**: 1.0.0
**維護者**: IndexTTS Testing Team

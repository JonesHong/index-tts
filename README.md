# Index-TTS (Custom Fork)

**Languages**: [English](#) | [繁體中文](README.zh-TW.md)

> [!NOTE]
> **This is a customized fork of the official Index-TTS project.**
>
> For the original README and official documentation, please visit:
> **https://github.com/index-tts/index-tts**

---

## 🎯 Main Focus: Streaming TTS with `test_streaming.py`

This fork is primarily focused on **real-time streaming TTS capabilities** for integration with LiveKit and other real-time applications. The core demonstration is in `test_streaming.py`, which showcases advanced streaming features with speed control and parallel audio generation/playback.

### 🚀 Quick Start with Streaming

```bash
# Basic streaming test
uv run test_streaming.py

# With custom reference audio
uv run test_streaming.py --ref_audio examples/Joneshong.wav

# With speed control (playback speed)
uv run test_streaming.py --speed 1.3

# With reference audio pre-processing (speed up reference before TTS)
uv run test_streaming.py --pre_speed_ref 1.3

# Full example with all options
uv run test_streaming.py \
  --version v2 \
  --method token \
  --ref_audio examples/Joneshong.wav \
  --pre_speed_ref 1.3 \
  --speed 1.0 \
  --warmup \
  --text "Your test text"
```

### 📖 `test_streaming.py` Features

#### 1. **Dual Speed Control Strategy**
- **`--pre_speed_ref`**: Pre-process reference audio before TTS generation
  - Speeds up the reference audio file using time-stretching
  - TTS model mimics the faster speech pattern
  - Uses `pyrubberband` for high-quality time-stretching

- **`--speed`**: Post-process playback speed
  - Applies DSP time-stretching to generated audio during playback
  - Independent from TTS generation
  - Useful for real-time playback control

#### 2. **Streaming Architecture**
- **Parallel Generation & Playback**: Audio chunks are generated and played simultaneously
- **Queue-based Pipeline**: Efficient audio chunk management
- **Low Latency**: First token latency tracking
- **Real-time Statistics**: Generation rate, RTF (Real-Time Factor), parallel efficiency

#### 3. **Smart Text Segmentation**
- **Token-based** (v2 only): Automatic segmentation by model
- **Word-based**: Manual punctuation-aware segmentation
- Configurable segment length for optimal streaming

#### 4. **Comprehensive Statistics**
The script provides detailed performance metrics:
- First token latency
- Audio generation rate (Audio/Process Speed)
- Overall RTF (Real-Time Factor)
- Parallel efficiency (how much generation overlaps with playback)
- Reference audio analysis (format, bitrate, duration)

---

## 📊 Benchmark Testing

### 🚀 Quick Start - Run Complete Test Suite

```bash
# Windows (Recommended: Python version)
uv run run_tests_launcher.py

# Or use batch file
run_tests.bat
```

### 📈 Test Suite Overview

The comprehensive benchmark testing includes **3 test suites with 9 test cases** in total, automatically generating analysis reports in multiple formats:

#### **Test Suite 1: Reference Audio Comparison** (2 tests)
Compare different reference audio files (voice_06.wav vs voice_07.wav) with identical parameters
- Version: v2
- Segmentation: token
- Includes warmup

#### **Test Suite 2: Speed Strategy Comparison** (4 tests) ⭐ With Audio Output
Compare four speed modification strategies:
1. **No Speed** - Baseline without speed modification
2. **Pre-Speed 1.2x** - Pre-processing acceleration (speed up reference audio)
3. **Post-Speed 1.2x** - Post-processing acceleration (DSP time-stretching)
4. **Hybrid 1.2x** - Hybrid acceleration (combines both methods)

**Output**: 4 WAV files for manual audio quality evaluation

#### **Test Suite 3: Version & Mode Comparison** (3 tests)
Compare different versions and segmentation modes:
- v1 streaming
- v2 streaming (token-based)
- v2 streaming (word-based)

### 📁 Test Output

After execution, the following files are generated in the `test_results/` directory:

```
test_results/
├── test_results_YYYYMMDD_HHMMSS.csv           # 📊 CSV statistics table
├── test_results_YYYYMMDD_HHMMSS.json          # 📝 Complete test logs
├── performance_comparison_YYYYMMDD_HHMMSS.png # 📈 Performance comparison chart
├── efficiency_analysis_YYYYMMDD_HHMMSS.png    # 📈 Efficiency analysis chart
├── summary_report_YYYYMMDD_HHMMSS.txt         # 📄 Summary report
└── audio_samples/                              # 🎵 Audio samples (Suite 2)
    ├── voice_07_no_speed.wav
    ├── voice_07_pre_speed_1.2x.wav
    ├── voice_07_post_speed_1.2x.wav
    └── voice_07_hybrid_speed_1.2x.wav
```

### 📊 Test Metrics

#### **Performance Comparison Chart** (performance_comparison)
![Performance Comparison](test_results/performance_comparison_example.png)

1. **TTFB (Time To First Byte)** - First Response Time
   - Measures latency from start to first audio chunk
   - ✅ Excellent: <3s | ⚠️ Acceptable: <5s | ❌ Need Optimization: >5s

2. **Total Generation Time** - Complete Generation Duration
   - Total time required to generate complete audio
   - Lower is better, affects overall efficiency

3. **Average Generation Rate** - Average Generation Speed Multiplier
   - Ratio of audio duration / generation time
   - ✅ Excellent: >2.0x | ✅ Good: >1.0x | ❌ Insufficient: <1.0x
   - **Must be >1.0x for real-time streaming**

#### **Efficiency Analysis Chart** (efficiency_analysis)
![Efficiency Analysis](test_results/efficiency_analysis_example.png)

1. **Overall RTF (Real-Time Factor)** - Overall Real-Time Factor
   - Total elapsed time / audio duration
   - ✅ Excellent: <0.5 | ✅ Good: <1.0 | ⚠️ Acceptable: <1.5

2. **Parallel Efficiency** - Concurrent Processing Efficiency
   - Percentage of generation and playback overlap
   - ✅ Excellent: >80% | ✅ Good: >60% | ⚠️ Need Improvement: <60%
   - High parallel efficiency indicates good streaming performance

3. **Memory Usage** - Memory Consumption
   - Peak memory usage (MB)
   - Monitors resource consumption

### 📋 CSV Data Fields

The generated CSV file contains the following fields:

| Field | Description | Unit |
|-------|-------------|------|
| `test_name` | Test case name | - |
| `ttfb` | First response time | seconds (s) |
| `total_time` | Total generation time | seconds (s) |
| `avg_gen_rate` | Average generation rate | multiplier (x) |
| `max_gen_rate` | Maximum generation rate | multiplier (x) |
| `min_gen_rate` | Minimum generation rate | multiplier (x) |
| `overall_rtf` | Overall RTF | - |
| `parallel_efficiency` | Parallel efficiency | percentage (%) |
| `total_chunks` | Total audio chunks | count |
| `total_audio_duration` | Total audio duration | seconds (s) |
| `peak_memory_mb` | Peak memory | MB |
| `avg_memory_mb` | Average memory | MB |

### 🔧 Dependencies

#### Required Packages
```bash
uv pip install pyrubberband librosa opencc-python-reimplemented sounddevice soundfile torch numpy
```

#### Optional Packages (Enable Full Features)
```bash
# Visualization chart generation
uv pip install matplotlib

# Memory monitoring
uv pip install psutil
```

**Note**: Tests can run without optional packages, but charts or memory data will be missing.

### 📖 Detailed Documentation

For more testing information, refer to:
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Complete Test Guide**: [TEST_GUIDE.md](TEST_GUIDE.md)
- **Testing System Overview**: [README_TESTING.md](README_TESTING.md)
- **Technical Implementation Details**: [TESTING_IMPLEMENTATION.md](TESTING_IMPLEMENTATION.md)

### ⏱️ Estimated Execution Time

- **Complete Test Suite**: 30-60 minutes (depends on hardware performance)
- **Single Test**: 3-5 minutes
- **GPU Acceleration**: Significantly faster (CUDA recommended)

---

## 🔧 Environment Setup with `runtime_setup.py`

This fork includes a custom environment initialization system:

```python
import runtime_setup

# Initialize environment (handles paths, cache, CUDA, etc.)
env_paths = runtime_setup.initialize(__file__)
INDEX_TTS_DIR = env_paths["INDEX_TTS_DIR"]
```

**What it does:**
- Sets up HuggingFace cache directories
- Configures DeepSpeed environment variables
- Adds FFMPEG to PATH
- Configures Torch extensions directory
- Handles BigVGAN CUDA plugin paths

---

## 📦 Installation

### Prerequisites
1. Install [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone this repository:
```bash
git clone https://github.com/JonesHong/index-tts.git
cd index-tts
```

### Install Dependencies
```bash
# Install all dependencies
uv sync --all-extras

# Or without DeepSpeed (Windows users)
uv sync --extra webui
```

### Download Models

**For v2 (recommended):**
```bash
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints_v2
```

**For v1.5:**
```bash
hf download IndexTeam/IndexTTS-1.5 --local-dir=checkpoints_v1.5
```

### Additional Dependencies for Streaming
```bash
# Install required packages for streaming features
uv pip install pyrubberband sounddevice soundfile opencc-python-reimplemented
```

---

## 🔬 Other Testing Scripts

### `test_infer.py`
Basic inference testing without streaming.

### `prepare_speed_ref_final.py`
Utilities for pre-processing reference audio with speed adjustments.

---

## 🛠️ Key Modifications in This Fork

### 1. **Streaming Support**
- Added `indextts/infer_streaming_patch.py` - Streaming patch for IndexTTS v1
- Modified `indextts/infer_v2.py` - Enhanced streaming support for v2
- Implemented chunk-based audio generation

### 2. **Speed Control**
- Dual-stage speed control (pre-processing + post-processing)
- High-quality time-stretching using `pyrubberband`

### 3. **Environment Management**
- `runtime_setup.py` - Centralized environment initialization
- Automatic cache directory management
- DeepSpeed configuration for Windows compatibility

### 4. **Model Modifications**
- `indextts/gpt/model_v2.py` - Custom modifications for streaming
- `indextts/s2mel/modules/commons.py` - Enhanced compatibility
- `indextts/s2mel/modules/diffusion_transformer.py` - Performance optimizations

### 5. **Testing & Utilities**
- Comprehensive streaming test suite
- Performance benchmarking tools
- Audio pre-processing utilities

---

## 📊 Performance Tips

1. **Use FP16 for faster inference:**
   ```python
   tts = IndexTTS2(use_fp16=True, ...)
   ```

2. **Enable warmup for consistent latency:**
   ```bash
   uv run test_streaming.py --warmup
   ```

3. **Adjust segment length for streaming:**
   - Smaller segments = lower latency, more overhead
   - Larger segments = higher latency, better efficiency

4. **Speed control recommendations:**
   - For natural speech: `--pre_speed_ref 1.0 --speed 1.0`
   - For faster playback: `--speed 1.2` to `1.5`
   - For TTS to learn faster speech: `--pre_speed_ref 1.2` to `1.3`

---

## 🐛 Known Issues & Fixes

### Issue: `DLL load failed` on Windows
**Solution:** Ensure CUDA Toolkit 12.8+ is installed and `torch/lib` is in PATH (handled by `runtime_setup.py`)

### Issue: Slow streaming performance
**Solutions:**
- Enable `--warmup` flag
- Use `--method token` for v2 (automatic segmentation)
- Reduce segment length
- Use FP16 mode

---

## 📝 License

This fork maintains the same license as the original project. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

This is a personal fork for LiveKit integration. For contributions to the main project, please visit the [official repository](https://github.com/index-tts/index-tts).

---

## 📧 Contact

For questions about this fork:
- GitHub Issues: https://github.com/JonesHong/index-tts/issues

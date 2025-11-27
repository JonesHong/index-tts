# Index-TTS (Custom Fork)

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
  --text "你的測試文本"
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

### 🔧 Environment Setup with `runtime_setup.py`

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
uv pip install pyrubberband librosa sounddevice soundfile opencc-python-reimplemented
```

---

## 🎵 Example Audio Files

This fork includes several example reference audio files in the `examples/` directory:
- `Joneshong.wav` - Default reference voice
- `GY.wav`, `DIDI.wav`, `JADE.wav`, `Sean.wav` - Additional voices
- `阿璋.wav` - Chinese voice sample
- `voice_06_1.3x.wav` - Pre-processed speed-adjusted sample

---

## 🔬 Other Testing Scripts

### `test_infer.py`
Basic inference testing without streaming.

### `test_streaming_enhanced.py`
Enhanced version with additional experimental features.

### `prepare_speed_adjusted_ref.py` & `prepare_speed_ref_final.py`
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
- Fallback to `librosa` if needed

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

## 📚 Original Project Information

This fork is based on **IndexTTS2** by the Bilibili Index Team.

- **Original Repository**: https://github.com/index-tts/index-tts
- **Paper**: [IndexTTS2 on arXiv](https://arxiv.org/abs/2506.21619)
- **Demo**: https://index-tts.github.io/index-tts2.github.io/
- **HuggingFace**: https://huggingface.co/IndexTeam/IndexTTS-2

### Citation
If you use this fork or the original IndexTTS2 in your research, please cite:

```bibtex
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```

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

For questions about the original project:
- Email: indexspeech@bilibili.com
- QQ Group: 663272642 (No.4), 1013410623 (No.5)
- Discord: https://discord.gg/uT32E7KDmy

#!/usr/bin/env python3
"""
IndexTTS Streaming Test with Audio Output Support
==========================================
基於 test_streaming.py，添加了音檔輸出功能

新增參數:
  --output PATH    保存生成的音檔到指定路徑 (WAV 格式)
"""

import sys
import os
import time
import argparse
import threading
import queue
import warnings
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf

# 引入外部依賴
try:
    import pyrubberband as pyrb
    from opencc import OpenCC
    import librosa
except ImportError as e:
    print(f"錯誤: 缺少必要套件 {e.name}。請確保已安裝 pyrubberband, librosa 和 opencc-python-reimplemented")
    sys.exit(1)

# ==================== 1. 環境初始化 ====================
import runtime_setup

env_paths = runtime_setup.initialize(__file__)
INDEX_TTS_DIR = env_paths["INDEX_TTS_DIR"]

sys.path.append(INDEX_TTS_DIR)
sys.path.append(os.path.join(INDEX_TTS_DIR, "indextts"))

import torch

from indextts.infer_v2 import IndexTTS2
from indextts.infer import IndexTTS
from indextts.infer_streaming_patch import add_streaming_to_indextts

# ==================== 2. 全域設定與工具函數 ====================

cc = OpenCC('t2s')

def convert_to_simplified(text):
    return cc.convert(text)

def get_timestamp(start_time):
    return time.time() - start_time

def check_cuda():
    print(f"\n{'='*20} 硬體狀態 {'='*20}")
    if torch.cuda.is_available():
        print(f"CUDA 版本   : {torch.version.cuda}")
        print(f"GPU 型號    : {torch.cuda.get_device_name(0)}")
        print(f"顯存 (總量) : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    else:
        print("⚠️ CUDA 不可用, 將使用 CPU (速度會受影響)")

def split_text_smart(text, target_length=20, min_length=6):
    punctuation = '，。！？；：、,.'
    max_length = int(target_length * 1.5)
    segments = []
    current_segment = ""

    for char in text:
        current_segment += char
        if len(current_segment) >= min_length and char in punctuation:
            clean_seg = current_segment.strip()
            if clean_seg: segments.append(clean_seg)
            current_segment = ""
        elif len(current_segment) >= max_length:
            clean_seg = current_segment.strip()
            if clean_seg: segments.append(clean_seg)
            current_segment = ""

    if current_segment.strip():
        segments.append(current_segment.strip())

    return [seg.lstrip(punctuation).strip() for seg in segments if seg.lstrip(punctuation).strip()]

def time_stretch_robust(audio_data, sample_rate, speed_factor, quality='speech'):
    try:
        if quality == 'speech':
            try:
                return pyrb.time_stretch(
                    audio_data,
                    sample_rate,
                    speed_factor,
                    rbargs={'-c': 6}
                )
            except Exception:
                pass

        return pyrb.time_stretch(audio_data, sample_rate, speed_factor)

    except Exception:
        return librosa.effects.time_stretch(audio_data, rate=speed_factor)

# ==================== 3. 播放器邏輯 ====================

class AudioPlayer(threading.Thread):
    def __init__(self, sample_rate, speed_factor=1.0, save_output=False):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.active = threading.Event()
        self.active.set()
        self.sample_rate = sample_rate
        self.speed_factor = speed_factor
        self.save_output = save_output
        self.events = []
        self.start_ref_time = 0
        self.collected_audio = []  # 收集所有音訊片段用於保存

    def set_start_time(self, start_time):
        self.start_ref_time = start_time

    def put_chunk(self, audio_data, duration, chunk_id):
        self.queue.put((audio_data, duration, chunk_id))

    def stop(self):
        self.queue.put(None)
        self.join()

    def get_full_audio(self):
        """返回完整的音訊數據（已應用變速）"""
        if not self.collected_audio:
            return None
        return np.concatenate(self.collected_audio)

    def run(self):
        chunk_idx = 0
        print(f"[Player] 播放執行緒啟動 (採樣率: {self.sample_rate}, 播放倍速: {self.speed_factor}x)")

        while self.active.is_set():
            try:
                item = self.queue.get(timeout=0.5)
                if item is None: break

                audio_normalized, original_duration, chunk_id = item
                chunk_idx += 1

                # 變速處理
                if abs(self.speed_factor - 1.0) > 0.01:
                    try:
                        audio_play = time_stretch_robust(audio_normalized, self.sample_rate, self.speed_factor)
                    except Exception as e:
                        print(f"⚠️ 播放時變速失敗: {e}")
                        audio_play = audio_normalized
                else:
                    audio_play = audio_normalized

                # 收集音訊（如果需要保存）
                if self.save_output:
                    self.collected_audio.append(audio_play)

                # 記錄事件
                play_start = get_timestamp(self.start_ref_time)
                self.events.append({
                    'event': 'play_start',
                    'chunk': chunk_id,
                    'timestamp': play_start,
                    'duration': original_duration
                })

                # 播放
                sd.play(audio_play, samplerate=self.sample_rate)
                sd.wait()

                self.queue.task_done()

            except queue.Empty:
                continue

        print("[Player] 播放執行緒結束")

# ==================== 4. 主程式邏輯 ====================

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    ref_audio_dir = os.path.join(INDEX_TTS_DIR, "examples")
    ref_audio_dist = {
        "voice_03.wav":  os.path.join(ref_audio_dir, "voice_03.wav"),
        "voice_06.wav":  os.path.join(ref_audio_dir, "voice_06.wav"),
        "voice_07.wav":  os.path.join(ref_audio_dir, "voice_07.wav"),
        "voice_11.wav":  os.path.join(ref_audio_dir, "voice_11.wav"),
        "阿璋.wav":  os.path.join(ref_audio_dir, "阿璋.wav"),
        "GY.wav":  os.path.join(ref_audio_dir, "GY.wav"),
        "DIDI.wav":  os.path.join(ref_audio_dir, "DIDI.wav"),
        "JADE.wav":  os.path.join(ref_audio_dir, "JADE.wav"),
        "Joneshong.wav":  os.path.join(ref_audio_dir, "Joneshong.wav"),
        "Sean.wav":  os.path.join(ref_audio_dir, "Sean.wav"),
    }

    # --- 參數解析 ---
    parser = argparse.ArgumentParser(description="Index-TTS Streaming Test")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="模型版本")
    parser.add_argument("--method", type=str, default="token", choices=["token", "word"], help="切分方法 (v2 Only)")
    parser.add_argument("--model_dir", type=str, default=None, help="模型資料夾路徑")
    parser.add_argument("--ref_audio", type=str, default=ref_audio_dist["Joneshong.wav"], help="參考音頻路徑")
    parser.add_argument("--text", type=str, default=None, help="測試文本")
    parser.add_argument("--steps", type=int, default=5, help="擴散模型步數 (僅參考)")
    parser.add_argument("--warmup", action="store_true", help="是否執行模型預熱")

    parser.add_argument("--speed", type=float, default=1.0,
                        help="[後處理] 播放加速倍率 (生成後才加速，預設 1.0)")

    parser.add_argument("--pre_speed_ref", type=float, default=1.0,
                        help="[預處理] 參考音檔加速倍率 (TTS生成前先加速參考音檔，預設 1.0)")

    # 新增: 音檔輸出參數
    parser.add_argument("--output", type=str, default=None,
                        help="保存生成的音檔到指定路徑 (WAV 格式)")

    args = parser.parse_args()

    # --- 文本處理 ---
    default_text = (
        "劉佩真分析，行政院「開水龍頭」，9月初新青安鬆綁，及延長對先買後賣換屋族出售舊屋的期限，觀望的市場氛圍稍減，房市交易量出現小幅成長，"
        "事實上，今年房市的交易結構已從去年的價量齊揚，到今年的量縮、價格緩跌。"
        "目前房價的跌幅方面，相較於去年還有非常低個位數的下滑，顯示房市賣方實際上沒有出脫的壓力。"
    )
    target_text = args.text if args.text else default_text
    text_simplified = convert_to_simplified(target_text)

    # --- 顯示配置 ---
    print(f"\n{'='*20} 測試配置 {'='*20}")
    print(f"版本: {args.version}")
    print(f"方法: {args.method}")
    print(f"參考音檔: {os.path.basename(args.ref_audio)}")
    print(f"--------------------")
    print(f"1. 參考音檔加速 (TTS模仿): {args.pre_speed_ref}x")
    print(f"2. 播放後製加速 (DSP處理): {args.speed}x")
    if args.output:
        print(f"3. 輸出路徑: {args.output}")
    print(f"--------------------")

    check_cuda()

    # --- 參考音檔預處理 ---
    temp_file_obj = None
    actual_ref_audio_path = args.ref_audio

    if abs(args.pre_speed_ref - 1.0) > 0.01:
        print(f"\n⚡ 正在執行參考音檔預加速 (倍率: {args.pre_speed_ref}x)...")
        try:
            y, sr = sf.read(args.ref_audio)
            if len(y.shape) > 1: y = np.mean(y, axis=1)

            y_fast = time_stretch_robust(y, sr, args.pre_speed_ref, quality='speech')

            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tf.name, y_fast, sr)
            tf.close()

            temp_file_obj = tf
            actual_ref_audio_path = tf.name

            print(f"  ✓ 預加速完成")
            print(f"  ✓ 暫存參考音檔路徑: {actual_ref_audio_path}")

        except Exception as e:
            print(f"❌ 預加速處理失敗: {e}")
            actual_ref_audio_path = args.ref_audio
            if temp_file_obj and os.path.exists(temp_file_obj.name):
                os.remove(temp_file_obj.name)
            temp_file_obj = None

    # --- 載入模型 ---
    print("\n=== 載入模型中... ===")
    start_load = time.time()

    tts_model = None
    sampling_rate = 22050

    if args.version == "v2":
        model_dir = args.model_dir or os.path.join(INDEX_TTS_DIR, "checkpoints_v2")
        config_path = os.path.join(model_dir, "config.yaml")
        sampling_rate = 22050

        tts_model = IndexTTS2(
            cfg_path=config_path,
            model_dir=model_dir,
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False
        )
    else:
        model_dir = args.model_dir or os.path.join(INDEX_TTS_DIR, "checkpoints_v1.5")
        config_path = os.path.join(model_dir, "config.yaml")
        sampling_rate = 24000

        tts_model = IndexTTS(
            model_dir=model_dir,
            cfg_path=config_path,
            use_fp16=True,
            use_cuda_kernel=False
        )
        tts_model = add_streaming_to_indextts(tts_model)

    print(f"✅ 模型載入完成 (耗時: {time.time() - start_load:.2f}s)")

    # --- 模型預熱 ---
    if args.warmup:
        print(f"\n{'='*20} 🔥 模型預熱 {'='*20}")
        print("正在執行預熱...")
        warmup_start = time.time()
        warmup_text = "測試預熱。"
        try:
            if args.version == "v2":
                dummy_kwargs = {
                    "spk_audio_prompt": actual_ref_audio_path,
                    "text": convert_to_simplified(warmup_text),
                    "output_path": None,
                    "stream_return": True,
                    "interval_silence": 150,
                    "verbose": False,
                    "use_emo_text": False,
                    "emo_vector": None
                }
                if args.method == "token":
                    dummy_kwargs["max_text_tokens_per_segment"] = 68
                for _ in tts_model.infer(**dummy_kwargs): pass
            else:
                for _ in tts_model.infer_stream(actual_ref_audio_path, convert_to_simplified(warmup_text), verbose=False): pass

            if torch.cuda.is_available(): torch.cuda.synchronize()
            print(f"✅ 預熱完成 (耗時: {time.time() - warmup_start:.2f}s)")
        except Exception as e:
            print(f"⚠️ 預熱錯誤: {e}")

    # --- 準備播放器 (啟用音檔收集) ---
    player = AudioPlayer(
        sample_rate=sampling_rate,
        speed_factor=args.speed,
        save_output=bool(args.output)  # 如果有輸出路徑，啟用收集
    )
    player.start()

    # --- 準備生成 ---
    processing_queue = []
    if args.version == "v2" and args.method == "token":
        processing_queue.append((text_simplified, "full_text"))
    else:
        segments = split_text_smart(text_simplified)
        print(f"📝 手動切分: 共 {len(segments)} 段")
        for i, seg in enumerate(segments):
            processing_queue.append((seg, f"segment_{i+1}"))

    # --- 開始生成 ---
    global_start_time = time.time()
    player.set_start_time(global_start_time)

    chunk_count = 0
    first_chunk_time = None
    generation_events = []
    speed_stats = []

    print(f"\n[🚀 Start] 開始串流生成...")

    try:
        try:
            for text_input, label in processing_queue:
                print(f"[🎬 Gen] 正在處理: {label} ({len(text_input)}字)")

                audio_generator = None

                if args.version == "v2":
                    kwargs = {
                        "spk_audio_prompt": actual_ref_audio_path,
                        "text": text_input,
                        "output_path": None,
                        "stream_return": True,
                        "interval_silence": 150,
                        "verbose": False,
                        "use_emo_text": False,
                        "emo_vector": None
                    }
                    if args.method == "token":
                        kwargs["max_text_tokens_per_segment"] = 68
                    audio_generator = tts_model.infer(**kwargs)
                else:
                    audio_generator = tts_model.infer_stream(actual_ref_audio_path, text_input, verbose=False)

                t_last_chunk_finish = time.time()

                for audio_chunk in audio_generator:
                    t_now_abs = time.time()
                    t_now_rel = get_timestamp(global_start_time)
                    chunk_latency = t_now_abs - t_last_chunk_finish
                    t_last_chunk_finish = t_now_abs
                    chunk_count += 1

                    if isinstance(audio_chunk, list):
                        audio_chunk = torch.cat(audio_chunk, dim=-1) if len(audio_chunk) > 0 else torch.zeros(1)
                    audio_np = audio_chunk.cpu().numpy().squeeze()
                    audio_normalized = audio_np.astype(np.float32) / 32767.0
                    duration = audio_np.shape[-1] / sampling_rate

                    if chunk_latency > 0.01:
                        gen_rate = duration / chunk_latency
                        speed_stats.append(gen_rate)

                    if duration < 0.05: continue

                    if first_chunk_time is None:
                        first_chunk_time = t_now_rel
                        print(f"[⚡ First Token] 首個音訊已生成: {first_chunk_time:.2f}s")

                    generation_events.append({
                        'event': 'generate',
                        'chunk': chunk_count,
                        'timestamp': t_now_rel,
                        'duration': duration
                    })

                    player.put_chunk(audio_normalized, duration, chunk_count)
                    if duration > 0.1:
                        print(f"  -> [Queue] 片段 {chunk_count} (音長 {duration:.2f}s, 耗時 {chunk_latency:.2f}s, 倍率 {gen_rate:.2f}x)")
        except KeyboardInterrupt:
            print("\n⚠️ 使用者中斷")
        except Exception as e:
            print(f"\n❌ 生成過程發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    finally:
        if temp_file_obj:
            try:
                if not temp_file_obj.closed:
                    temp_file_obj.close()

                if os.path.exists(temp_file_obj.name):
                    os.remove(temp_file_obj.name)
                    print(f"\n🗑️ 已清理暫存參考音檔: {temp_file_obj.name}")
            except Exception as e:
                print(f"⚠️ 清理暫存檔時發生錯誤: {e}")

    total_gen_time = get_timestamp(global_start_time)
    print(f"\n[🏁 Finish] 所有生成任務完成 (總耗時: {total_gen_time:.2f}s)")

    player.stop()

    # --- 保存音檔 (新增功能) ---
    if args.output:
        print(f"\n{'='*20} 💾 保存音檔 {'='*20}")
        try:
            full_audio = player.get_full_audio()
            if full_audio is not None and len(full_audio) > 0:
                # 確保目錄存在
                output_dir = os.path.dirname(args.output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                # 保存 WAV
                sf.write(args.output, full_audio, sampling_rate)
                file_size = os.path.getsize(args.output) / 1024 / 1024  # MB
                duration = len(full_audio) / sampling_rate

                print(f"✅ 音檔已保存:")
                print(f"   路徑: {args.output}")
                print(f"   大小: {file_size:.2f} MB")
                print(f"   時長: {duration:.2f} s")
                print(f"   採樣率: {sampling_rate} Hz")
            else:
                print(f"⚠️ 無音訊數據可保存")
        except Exception as e:
            print(f"❌ 保存音檔失敗: {e}")
            import traceback
            traceback.print_exc()

    # --- 統計報告 ---
    print(f"\n{'='*80}")
    print(f"📊 綜合統計報告")
    print(f"{'='*80}")

    print(f"\n🔧 執行參數 (Arguments):")
    for k, v in vars(args).items():
        print(f"  • {k:<12} : {v}")

    print(f"\n🎵 參考音訊資訊 (Ref Audio):")
    if os.path.exists(args.ref_audio):
        try:
            file_size_bytes = os.path.getsize(args.ref_audio)
            file_size_mb = file_size_bytes / (1024 * 1024)
            sf_info = sf.info(args.ref_audio)
            duration = sf_info.duration
            samplerate = sf_info.samplerate
            subtype = sf_info.subtype
            bitrate_kbps = (file_size_bytes * 8) / duration / 1000 if duration > 0 else 0

            print(f"  • 檔名 (File)    : {os.path.basename(args.ref_audio)}")
            print(f"  • 大小 (Size)    : {file_size_mb:.2f} MB")
            print(f"  • 秒數 (Length)  : {duration:.2f} s")
            print(f"  • 格式 (Format)  : {sf_info.format} ({subtype})")
            print(f"  • 採樣 (Rate)    : {samplerate} Hz")
            print(f"  • 碼率 (Bitrate) : {bitrate_kbps:.0f} kbps")
        except Exception as e:
            print(f"  ⚠️ 無法讀取音訊資訊: {e}")
    else:
        print(f"  ⚠️ 找不到檔案: {args.ref_audio}")

    print(f"\n🚀 效能指標:")
    print(f"  • 首次響應   : {first_chunk_time if first_chunk_time else 'N/A'}")
    print(f"  • 總耗時     : {total_gen_time:.2f} s")

    if speed_stats:
        avg_rate = np.mean(speed_stats)
        max_rate = np.max(speed_stats)
        min_rate = np.min(speed_stats)
        print(f"  • 生成倍率 (Audio/Process Speed):")
        print(f"    (數值 > 1.0 代表生成速度比播放速度快)")
        print(f"      Avg : {avg_rate:.2f} x")
        print(f"      Max : {max_rate:.2f} x")
        print(f"      Min : {min_rate:.2f} x")

        total_audio_len = sum(e['duration'] for e in generation_events)
        overall_rtf = total_gen_time / total_audio_len if total_audio_len > 0 else 0
        print(f"  • 整體實時率 (RTF): {overall_rtf:.3f} (越低越好)")
    else:
        print(f"  • 生成倍率: N/A")

    # 並行效率
    overlap_count = 0
    playback_events = player.events
    if len(generation_events) > 1 and len(playback_events) > 0:
        for gen_event in generation_events[1:]:
            g_time = gen_event['timestamp']
            for play_event in playback_events:
                p_start = play_event['timestamp']
                p_end = p_start + (play_event['duration'] / args.speed)
                if p_start <= g_time <= p_end:
                    overlap_count += 1
                    break
        efficiency = (overlap_count / max(chunk_count - 1, 1)) * 100
        print(f"  • 並行效率 (Parallel): {efficiency:.1f}% ({overlap_count}/{chunk_count-1} chunks overlapped)")

    print(f"{'='*80}\n")
    print("Done.")

if __name__ == "__main__":
    main()

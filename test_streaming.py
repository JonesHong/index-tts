import sys
import os
import time
import argparse
import threading
import queue
import gc
import warnings
import tempfile  # 用於處理暫存音檔
import numpy as np
import sounddevice as sd
import soundfile as sf

# 引入外部依賴
try:
    import pyrubberband as pyrb
    from opencc import OpenCC
    # 如果沒有安裝 librosa，這裡可能需要處理，但 pyrb 通常依賴它
    import librosa 
except ImportError as e:
    print(f"錯誤: 缺少必要套件 {e.name}。請確保已安裝 pyrubberband, librosa 和 opencc-python-reimplemented")
    sys.exit(1)

# ==================== 1. 環境初始化 (使用 runtime_setup) ====================
import runtime_setup

# 初始化並取得路徑
env_paths = runtime_setup.initialize(__file__)
INDEX_TTS_DIR = env_paths["INDEX_TTS_DIR"]

# 設定 Python Path
sys.path.append(INDEX_TTS_DIR)
sys.path.append(os.path.join(INDEX_TTS_DIR, "indextts"))

# 必須在導入 torch 之前設定,避免 DeepSpeed 編譯檢查
import torch

# IndexTTS 模組導入 (必須在 sys.path 設定後)
from indextts.infer_v2 import IndexTTS2
from indextts.infer import IndexTTS
from indextts.infer_streaming_patch import add_streaming_to_indextts

# ==================== 2. 全域設定與工具函數 ====================

# 繁簡轉換
cc = OpenCC('t2s')

def convert_to_simplified(text):
    return cc.convert(text)

def get_timestamp(start_time):
    """獲取相對時間戳記（秒）"""
    return time.time() - start_time

def check_cuda():
    """檢查並打印 CUDA 狀態"""
    print(f"\n{'='*20} 硬體狀態 {'='*20}")
    if torch.cuda.is_available():
        print(f"CUDA 版本   : {torch.version.cuda}")
        print(f"GPU 型號    : {torch.cuda.get_device_name(0)}")
        print(f"顯存 (總量) : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    else:
        print("⚠️ CUDA 不可用, 將使用 CPU (速度會受影響)")

# 文字切分邏輯
def split_text_smart(text, target_length=20, min_length=6):
    """優化版切分：優先在標點切分"""
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
        
    # 二次清理
    return [seg.lstrip(punctuation).strip() for seg in segments if seg.lstrip(punctuation).strip()]

# ==================== [優化移植] 變速處理邏輯 ====================
def time_stretch_robust(audio_data, sample_rate, speed_factor, quality='speech'):
    """
    穩健的變速處理 (整合自 prepare_speed_ref_final.py)
    """
    # 嘗試使用 pyrubberband
    try:
        if quality == 'speech':
            # 優先嘗試語音優化參數 (高清晰度)
            try:
                # 修正後的參數傳遞方式，避免 TypeError
                return pyrb.time_stretch(
                    audio_data, 
                    sample_rate, 
                    speed_factor,
                    rbargs={'-c': 6} # Crispness 6 (High) 防止混響
                )
            except Exception:
                # 參數失敗則降級
                pass
        
        # 基本模式
        return pyrb.time_stretch(audio_data, sample_rate, speed_factor)
        
    except Exception:
        # 如果 pyrubberband 完全失敗，嘗試 librosa
        # print("  ⚠️ pyrubberband 失敗，切換至 librosa (音質可能較差)")
        return librosa.effects.time_stretch(audio_data, rate=speed_factor)

# ==================== 3. 播放器邏輯 (獨立執行緒) ====================

class AudioPlayer(threading.Thread):
    def __init__(self, sample_rate, speed_factor=1.0):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.active = threading.Event()
        self.active.set()
        self.sample_rate = sample_rate
        self.speed_factor = speed_factor
        self.events = [] # 記錄播放事件
        self.start_ref_time = 0

    def set_start_time(self, start_time):
        self.start_ref_time = start_time

    def put_chunk(self, audio_data, duration, chunk_id):
        self.queue.put((audio_data, duration, chunk_id))

    def stop(self):
        self.queue.put(None) # 結束信號
        self.join()

    def run(self):
        chunk_idx = 0
        print(f"[Player] 播放執行緒啟動 (採樣率: {self.sample_rate}, 播放倍速: {self.speed_factor}x)")
        
        while self.active.is_set():
            try:
                item = self.queue.get(timeout=0.5)
                if item is None: break # 收到結束信號

                audio_normalized, original_duration, chunk_id = item
                chunk_idx += 1
                
                # 變速處理 (後處理/播放加速)
                if abs(self.speed_factor - 1.0) > 0.01:
                    try:
                        audio_play = time_stretch_robust(audio_normalized, self.sample_rate, self.speed_factor)
                    except Exception as e:
                        print(f"⚠️ 播放時變速失敗: {e}")
                        audio_play = audio_normalized
                else:
                    audio_play = audio_normalized

                # 記錄開始
                play_start = get_timestamp(self.start_ref_time)
                self.events.append({'event': 'play_start', 'chunk': chunk_id, 'timestamp': play_start, 'duration': original_duration})
                
                # print(f"[🔊 Play] 片段 {chunk_id} 開始播放")
                
                # 播放 (阻塞直到播完)
                sd.play(audio_play, samplerate=self.sample_rate)
                sd.wait()

                # 記錄結束
                self.queue.task_done()
                
            except queue.Empty:
                continue
        print("[Player] 播放執行緒結束")

# ==================== 4. 主程式邏輯 ====================

def main():
    # --- 0. 屏蔽干擾訊息 ---
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
    
    # 變速相關參數 (修改部分)
    parser.add_argument("--speed", type=float, default=1.0, 
                        help="[後處理] 播放加速倍率 (生成後才加速，預設 1.0)")
    
    parser.add_argument("--pre_speed_ref", type=float, default=1.0, 
                        help="[預處理] 參考音檔加速倍率 (TTS生成前先加速參考音檔，預設 1.0)")
    
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
    print(f"--------------------")
    
    check_cuda()

    # --- 變速策略處理 (核心修改) ---
    # Logic:
    # 1. actual_ref_audio_path: 實際傳給 TTS 的路徑。若 pre_speed_ref != 1.0，則為暫存檔路徑。
    # 2. player_speed_factor: 播放器的速度，直接使用 args.speed。
    
    temp_file_obj = None     # 保存 temp file 物件
    actual_ref_audio_path = args.ref_audio
    
    # 執行 [預處理] 參考音檔加速
    if abs(args.pre_speed_ref - 1.0) > 0.01:
        print(f"\n⚡ 正在執行參考音檔預加速 (倍率: {args.pre_speed_ref}x)...")
        try:
            # 1. 讀取原始參考音檔
            y, sr = sf.read(args.ref_audio)
            if len(y.shape) > 1: y = np.mean(y, axis=1) # 轉單聲道
            
            # 2. 變速處理 (使用 robust 方法)
            y_fast = time_stretch_robust(y, sr, args.pre_speed_ref, quality='speech')
            
            # 3. 寫入暫存檔 (Windows 兼容寫法: delete=False, 手動刪除)
            # delete=False 是為了確保在 close 之後，檔案還在磁碟上供 TTS 讀取
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tf.name, y_fast, sr)
            tf.close() # 關閉檔案 handle，釋放 lock，讓 TTS 可以讀取
            
            temp_file_obj = tf # 保存引用以便後續刪除
            actual_ref_audio_path = tf.name
            
            print(f"  ✓ 預加速完成")
            print(f"  ✓ 暫存參考音檔路徑: {actual_ref_audio_path}")
            
        except Exception as e:
            print(f"❌ 預加速處理失敗: {e}")
            # 如果失敗，回退到原始音檔
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
    else: # v1
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

    # ==================== 模型預熱 ====================
    if args.warmup:
        print(f"\n{'='*20} 🔥 模型預熱 {'='*20}")
        print("正在執行預熱...")
        warmup_start = time.time()
        warmup_text = "測試預熱。"
        try:
            if args.version == "v2":
                dummy_kwargs = {
                    "spk_audio_prompt": actual_ref_audio_path, # 使用處理後的路徑
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

    # --- 準備播放器 ---
    # args.speed 用於後處理 (DSP Time Stretch)
    player = AudioPlayer(sample_rate=sampling_rate, speed_factor=args.speed)
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

    # --- 開始生成迴圈 ---
    global_start_time = time.time()
    player.set_start_time(global_start_time)
    
    chunk_count = 0
    first_chunk_time = None
    generation_events = []
    speed_stats = [] 

    print(f"\n[🚀 Start] 開始串流生成...")

    try:
        # 使用 try...finally 確保暫存檔被刪除
        try:
            for text_input, label in processing_queue:
                print(f"[🎬 Gen] 正在處理: {label} ({len(text_input)}字)")

                audio_generator = None
                
                if args.version == "v2":
                    kwargs = {
                        "spk_audio_prompt": actual_ref_audio_path, # 使用正確的參考音檔路徑
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
        # 清理暫存檔 (無論是否發生錯誤)
        if temp_file_obj:
            try:
                # 確保關閉
                if not temp_file_obj.closed:
                    temp_file_obj.close()
                
                # 刪除實體檔案
                if os.path.exists(temp_file_obj.name):
                    os.remove(temp_file_obj.name)
                    print(f"\n🗑️ 已清理暫存參考音檔: {temp_file_obj.name}")
            except Exception as e:
                print(f"⚠️ 清理暫存檔時發生錯誤: {e}")

    total_gen_time = get_timestamp(global_start_time)
    print(f"\n[🏁 Finish] 所有生成任務完成 (總耗時: {total_gen_time:.2f}s)")
    
    player.stop()

    # ==================== 5. 綜合統計報告 ====================
    print(f"\n{'='*80}")
    print(f"📊 綜合統計報告")
    print(f"{'='*80}")

    # --- A. 參數配置 ---
    print(f"🔧 執行參數 (Arguments):")
    for k, v in vars(args).items():
        print(f"  • {k:<12} : {v}")
    
    # print(f"🔧 執行參數:")
    # print(f"  • 參考音檔   : {os.path.basename(args.ref_audio)}")
    # print(f"  • 預處理加速 : {args.pre_speed_ref}x {'(使用暫存檔)' if args.pre_speed_ref != 1.0 else '(無)'}")
    # print(f"  • 後處理加速 : {args.speed}x {'(播放時 DSP 處理)' if args.speed != 1.0 else '(無)'}")
    # --- B. 參考音訊分析 ---
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
    
    # [修正] 顯示生成倍率 (Audio Generation Rate)
    if speed_stats:
        avg_rate = np.mean(speed_stats)
        max_rate = np.max(speed_stats)
        min_rate = np.min(speed_stats)
        avg_rate = np.mean(speed_stats)
        print(f"  • 生成倍率 (Audio/Process Speed):")
        print(f"    (數值 > 1.0 代表生成速度比播放速度快)")
        print(f"      Avg : {avg_rate:.2f} x")
        print(f"      Max : {max_rate:.2f} x")
        print(f"      Min : {min_rate:.2f} x")
        
        # 估算整體 RTF
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
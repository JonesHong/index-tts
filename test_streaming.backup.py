import sys
import os
import time
import argparse
import threading
import queue
import gc
import warnings
import numpy as np
import sounddevice as sd
import soundfile as sf

# 引入外部依賴
try:
    import pyrubberband as pyrb
    from opencc import OpenCC
except ImportError as e:
    print(f"錯誤: 缺少必要套件 {e.name}。請確保已安裝 pyrubberband 和 opencc-python-reimplemented")
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
        print(f"[Player] 播放執行緒啟動 (採樣率: {self.sample_rate}, 倍速: {self.speed_factor})")
        
        while self.active.is_set():
            try:
                item = self.queue.get(timeout=0.5)
                if item is None: break # 收到結束信號

                audio_normalized, original_duration, chunk_id = item
                chunk_idx += 1
                
                # 變速處理 (pyrubberband) - 放在這裡做是為了不阻塞生成執行緒
                if abs(self.speed_factor - 1.0) > 0.01:
                    # 使用字典參數優化音質 (防止混響)
                    try:
                        audio_play = pyrb.time_stretch(
                            audio_normalized, 
                            self.sample_rate, 
                            self.speed_factor,
                            rbargs={'-c': 6} # Crispness 6 (High)
                        )
                    except:
                        # Fallback
                        audio_play = pyrb.time_stretch(audio_normalized, self.sample_rate, self.speed_factor)
                else:
                    audio_play = audio_normalized

                # 記錄開始
                play_start = get_timestamp(self.start_ref_time)
                self.events.append({'event': 'play_start', 'chunk': chunk_id, 'timestamp': play_start, 'duration': original_duration})
                
                print(f"[🔊 Play] 片段 {chunk_id} 開始播放 (原始時長 {original_duration:.2f}s)")
                
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
    parser.add_argument("--ref_audio", type=str, default=ref_audio_dist["voice_06.wav"], help="參考音頻路徑")
    parser.add_argument("--speed", type=float, default=1.3, help="播放語速")
    parser.add_argument("--text", type=str, default=None, help="測試文本")
    parser.add_argument("--steps", type=int, default=5, help="擴散模型步數 (僅參考)")
    parser.add_argument("--warmup", action="store_true", help="是否執行模型預熱") # <--- 新增參數
    
    args = parser.parse_args()

    # --- 文本處理 ---
    default_text = (
        # "一名自稱台大大氣系學生的網友在臉書「黑特帝大」立下豪言，預測台北市十一月十二日至十四日至少會因鳳凰颱風放假兩天，"
        # "並以三百份雞排珍奶作為失準時的祭品。結果這位學生被現實狠狠打臉，最終他也實現承諾，"
        # "宣布十一月十六日中午就在台大校園內發送雞排、珍奶，讓網友搶留言卡位。"
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
    print(f"語速: {args.speed}")
    print(f"預熱: {'開啟' if args.warmup else '關閉'}")
    print(f"原文: {target_text[:30]}...")
    
    check_cuda()

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

    # ==================== 新增：模型預熱 ====================
    if args.warmup:
        print(f"\n{'='*20} 🔥 模型預熱 {'='*20}")
        print("正在執行預熱 (Run dry-run)...")
        warmup_start = time.time()
        warmup_text = "你好，這是一段用來預熱模型的測試文本。"
        
        # 簡單跑一次生成，不放入播放隊列
        try:
            # 準備參數
            if args.version == "v2":
                dummy_kwargs = {
                    "spk_audio_prompt": args.ref_audio,
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
                
                # 消耗生成器 (計算但不使用)
                for _ in tts_model.infer(**dummy_kwargs):
                    pass
            else:
                # v1
                for _ in tts_model.infer_stream(args.ref_audio, convert_to_simplified(warmup_text), verbose=False):
                    pass
            
            # 強制同步 CUDA 確保預熱真的做完了
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            print(f"✅ 預熱完成 (耗時: {time.time() - warmup_start:.2f}s)")
        except Exception as e:
            print(f"⚠️ 預熱過程發生錯誤 (已略過): {e}")
    # =======================================================

    # --- 準備播放器 ---
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
    # 注意：這裡才開始正式計時
    global_start_time = time.time()
    player.set_start_time(global_start_time)
    
    chunk_count = 0
    first_chunk_time = None
    generation_events = []
    
    # [統計修正] 改用 Audio/Sec (生成倍率)
    speed_stats = [] 

    print(f"\n[🚀 Start] 開始串流生成 (計時開始)...")

    for text_input, label in processing_queue:
        print(f"[🎬 Gen] 正在處理: {label} ({len(text_input)}字)")

        audio_generator = None
        
        if args.version == "v2":
            kwargs = {
                "spk_audio_prompt": args.ref_audio,
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
            audio_generator = tts_model.infer_stream(args.ref_audio, text_input, verbose=False)

        # 計時初始化
        t_last_chunk_finish = time.time()

        for audio_chunk in audio_generator:
            # 取得當前時間
            t_now_abs = time.time()
            t_now_rel = get_timestamp(global_start_time)
            
            # 計算本次生成耗時 (Latency)
            chunk_latency = t_now_abs - t_last_chunk_finish
            t_last_chunk_finish = t_now_abs 
            
            chunk_count += 1

            # 處理音訊
            if isinstance(audio_chunk, list):
                audio_chunk = torch.cat(audio_chunk, dim=-1) if len(audio_chunk) > 0 else torch.zeros(1)
            audio_np = audio_chunk.cpu().numpy().squeeze()
            audio_normalized = audio_np.astype(np.float32) / 32767.0
            duration = audio_np.shape[-1] / sampling_rate
            
            # 計算生成倍率 (Audio Sec / Process Sec)
            # 數值 > 1 代表生成比播放快 (理想情況)
            if chunk_latency > 0.01:
                gen_rate = duration / chunk_latency
                speed_stats.append(gen_rate)
            else:
                gen_rate = 0
            
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

            # 這裡顯示 Gen Rate (倍率)
            if duration > 0.1:
                print(f"  -> [Queue] 片段 {chunk_count} (音長 {duration:.2f}s, 耗時 {chunk_latency:.2f}s, 倍率 {gen_rate:.2f}x)")
                player.put_chunk(audio_normalized, duration, chunk_count)
            else:
                print(f"  -> [Mute ] 靜音片段 {chunk_count} (音長 {duration:.2f}s)")
                player.put_chunk(audio_normalized, duration, chunk_count)

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

    # --- C. 效能統計 ---
    print(f"{'-'*40}")
    print(f"🚀 效能指標 (Performance):")
    print(f"  • 首次響應 (TTFT): {first_chunk_time if first_chunk_time else 'N/A'}")
    print(f"  • 總耗時   (Total): {total_gen_time:.2f} s")
    print(f"  • 總片段數 (Chunks): {chunk_count}")

    # [修正] 顯示生成倍率 (Audio Generation Rate)
    if speed_stats:
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
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
import torch

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
    parser.add_argument("--ref_audio", type=str, default=ref_audio_dist["voice_07.wav"], help="參考音頻路徑")
    parser.add_argument("--speed", type=float, default=1.3, help="播放語速")
    parser.add_argument("--text", type=str, default=None, help="測試文本")
    parser.add_argument("--steps", type=int, default=25, help="擴散模型步數 (僅參考)")
    
    args = parser.parse_args()

    # --- 文本處理 ---
    default_text = (
        "一名自稱台大大氣系學生的網友在臉書「黑特帝大」立下豪言，預測台北市十一月十二日至十四日至少會因鳳凰颱風放假兩天，"
        "並以三百份雞排珍奶作為失準時的祭品。結果這位學生被現實狠狠打臉，最終他也實現承諾，"
        "宣布十一月十六日中午就在台大校園內發送雞排、珍奶，讓網友搶留言卡位。"
    )
    target_text = args.text if args.text else default_text
    text_simplified = convert_to_simplified(target_text)

    # --- 顯示配置 ---
    print(f"\n{'='*20} 測試配置 {'='*20}")
    print(f"版本: {args.version}")
    print(f"方法: {args.method}")
    print(f"語速: {args.speed}")
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
    global_start_time = time.time()
    player.set_start_time(global_start_time)
    
    chunk_count = 0
    first_chunk_time = None
    generation_events = []
    
    # [統計修正] 改用 Audio/Sec (生成倍率)
    speed_stats = [] 
    
    # [新增] 詳細時間分解統計
    time_breakdown = {
        'audio_processing': [],     # 音訊處理時間（轉換、解碼）
        'gpu_transfer': [],         # GPU -> CPU 傳輸時間
        'waiting': [],              # 等待時間（生成器之間的間隔）
        'total_latency': []         # 總延遲
    }

    print(f"\n[🚀 Start] 開始串流生成...")

    for text_input, label in processing_queue:
        print(f"[🎬 Gen] 正在處理: {label} ({len(text_input)}字)")

        audio_generator = None
        
        if args.version == "v2":
            kwargs = {
                "spk_audio_prompt": args.ref_audio,
                "text": text_input,
                "output_path": None,
                "stream_return": True,
                "interval_silence": 250,
                "verbose": False,
                "use_emo_text": False,
                "emo_vector": None
            }
            if args.method == "token":
                kwargs["max_text_tokens_per_segment"] = 60
            audio_generator = tts_model.infer(**kwargs)
        else:
            audio_generator = tts_model.infer_stream(args.ref_audio, text_input, verbose=False)

        # 計時初始化
        t_last_chunk_finish = time.time()
        t_iterator_start = None

        for audio_chunk in audio_generator:
            # [時間點 1] 收到 chunk 的時間（此時模型已完成推理）
            t_chunk_received = time.time()
            t_now_rel = get_timestamp(global_start_time)
            
            # 計算等待時間（從上一個 chunk 完成到這個 chunk 開始）
            if t_iterator_start is not None:
                waiting_time = t_iterator_start - t_last_chunk_finish
                time_breakdown['waiting'].append(max(0, waiting_time))
            
            chunk_count += 1

            # [時間點 2] Tensor 預處理（在 CPU/GPU 上）
            t_preprocess_start = time.time()
            if isinstance(audio_chunk, list):
                audio_chunk = torch.cat(audio_chunk, dim=-1) if len(audio_chunk) > 0 else torch.zeros(1)
            t_preprocess_end = time.time()
            preprocess_time = t_preprocess_end - t_preprocess_start
            
            # [時間點 3] GPU -> CPU 傳輸
            t_before_cpu = time.time()
            audio_np = audio_chunk.cpu().numpy().squeeze()
            t_after_cpu = time.time()
            gpu_transfer_time = t_after_cpu - t_before_cpu
            
            # [時間點 4] CPU 端音訊格式轉換
            t_format_start = time.time()
            audio_normalized = audio_np.astype(np.float32) / 32767.0
            duration = audio_np.shape[-1] / sampling_rate
            t_format_end = time.time()
            format_time = t_format_end - t_format_start
            
            # 計算各階段時間
            audio_processing_time = preprocess_time + format_time  # 不包含 GPU 傳輸
            chunk_latency = t_format_end - t_last_chunk_finish
            
            # 儲存時間分解數據（排除太短的片段）
            if duration > 0.05:
                time_breakdown['total_latency'].append(chunk_latency)
                time_breakdown['gpu_transfer'].append(gpu_transfer_time)
                time_breakdown['audio_processing'].append(audio_processing_time)
            
            t_last_chunk_finish = t_format_end
            
            # 計算生成倍率 (Audio Sec / Process Sec)
            # 數值 > 1 代表生成比播放快 (理想情況)
            if chunk_latency > 0.01:
                gen_rate = duration / chunk_latency
                speed_stats.append(gen_rate)
            else:
                gen_rate = 0
            
            if duration < 0.05: 
                t_iterator_start = time.time()
                continue 

            if first_chunk_time is None:
                first_chunk_time = t_now_rel
                print(f"[⚡ First Token] 首個音訊已生成: {first_chunk_time:.2f}s")

            generation_events.append({
                'event': 'generate',
                'chunk': chunk_count,
                'timestamp': t_now_rel,
                'duration': duration,
                'latency': chunk_latency,
                'gpu_transfer': gpu_transfer_time,
                'audio_processing': audio_processing_time
            })

            # 這裡顯示 Gen Rate (倍率) 和詳細時間
            if duration > 0.1:
                print(f"  -> [Queue] 片段 {chunk_count} (音長 {duration:.2f}s, 耗時 {chunk_latency:.2f}s, 倍率 {gen_rate:.2f}x)")
                print(f"             [細節] GPU傳輸 {gpu_transfer_time*1000:.1f}ms | 音訊處理 {audio_processing_time*1000:.1f}ms")
                player.put_chunk(audio_normalized, duration, chunk_count)
            else:
                print(f"  -> [Queue] 片段 {chunk_count} (音長 {duration:.2f}s, 耗時 {chunk_latency:.2f}s, 倍率 {gen_rate:.2f}x)")
                player.put_chunk(audio_normalized, duration, chunk_count)
            
            # 記錄下一次迭代開始時間
            t_iterator_start = time.time()

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

    # [新增] 時間分解統計
    if time_breakdown['total_latency']:
        print(f" ⏱️  時間分解 (Time Breakdown):")
        
        # 計算平均值
        avg_total = np.mean(time_breakdown['total_latency'])
        avg_gpu = np.mean(time_breakdown['gpu_transfer'])
        avg_audio_proc = np.mean(time_breakdown['audio_processing'])
        
        # 計算模型內部處理時間（總延遲 - GPU傳輸 - 音訊處理）
        # 這包括：文本編碼、模型推理、mel生成、vocoder解碼等
        model_inference_times = []
        for i in range(len(time_breakdown['total_latency'])):
            model_time = (time_breakdown['total_latency'][i] - 
                         time_breakdown['gpu_transfer'][i] - 
                         time_breakdown['audio_processing'][i])
            model_inference_times.append(max(0, model_time))  # 避免負數
        avg_model = np.mean(model_inference_times)
        
        # 計算佔比
        model_ratio = (avg_model / avg_total * 100) if avg_total > 0 else 0
        audio_ratio = (avg_audio_proc / avg_total * 100) if avg_total > 0 else 0
        gpu_ratio = (avg_gpu / avg_total * 100) if avg_total > 0 else 0
        
        print(f"    • 單個 Chunk 平均延遲: {avg_total:.3f}s")
        print(f"      ├─ 模型內部處理:   {avg_model:.3f}s ({model_ratio:.1f}%)")
        print(f"      │  (文本編碼 + 模型推理 + mel生成 + vocoder)")
        print(f"      ├─ GPU→CPU 傳輸:   {avg_gpu:.3f}s ({gpu_ratio:.1f}%)")
        print(f"      └─ 音訊後處理:     {avg_audio_proc:.3f}s ({audio_ratio:.1f}%)")
        print(f"         (Tensor處理 + 格式轉換)")
        
        # 瓶頸分析
        print(f" 💡 瓶頸分析:")
        if model_ratio > 80:
            print(f"      主要瓶頸在「模型內部處理」({model_ratio:.1f}%)")
            print(f"      建議: 優化模型推理速度、使用更快的 GPU、降低 steps")
        elif audio_ratio > 50:
            print(f"      主要瓶頸在「音訊後處理」({audio_ratio:.1f}%)")
            print(f"      建議: 優化音訊解碼、減少 CPU 處理、使用批次處理")
        else:
            print(f"      各階段耗時相對均衡")
            print(f"      建議: 全面優化，特別關注佔比最高的部分")
        
        # GPU 傳輸警告
        if gpu_ratio > 10:
            print(f"      ⚠️  GPU 傳輸佔比較高 ({gpu_ratio:.1f}%)，考慮減少傳輸次數")
        
        # 顯示實際數值
        print(f" 📈 詳細數值:")
        print(f"      模型內部: {avg_model*1000:.1f}ms ({model_ratio:.1f}%)")
        print(f"      GPU傳輸:  {avg_gpu*1000:.1f}ms ({gpu_ratio:.1f}%)")
        print(f"      音訊處理: {avg_audio_proc*1000:.1f}ms ({audio_ratio:.1f}%)")

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
        print(f" • 並行效率 (Parallel): {efficiency:.1f}% ({overlap_count}/{chunk_count-1} chunks overlapped)")
    
    print(f"{'='*80}\n")
    print("Done.")

if __name__ == "__main__":
    main()
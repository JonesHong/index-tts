#!/usr/bin/env python3
"""
參考音檔變速工具 - 最終修正版

解決 pyrubberband 參數格式問題，支援多種 fallback 方案
"""

import sys
import os
import argparse
import numpy as np
import soundfile as sf

# 檢查並導入變速庫
SPEED_METHOD = None

try:
    import pyrubberband as pyrb
    SPEED_METHOD = 'pyrubberband'
    print("✓ 使用 pyrubberband 進行變速")
except ImportError:
    pass

if SPEED_METHOD is None:
    try:
        import librosa
        SPEED_METHOD = 'librosa'
        print("✓ 使用 librosa 進行變速")
    except ImportError:
        pass

if SPEED_METHOD is None:
    print("❌ 錯誤: 需要安裝 pyrubberband 或 librosa")
    print("請執行: pip install pyrubberband 或 pip install librosa")
    sys.exit(1)

def time_stretch_robust(audio_data, sample_rate, speed_factor, quality='speech'):
    """
    穩健的變速處理 (修正參數傳遞錯誤)
    """
    
    if SPEED_METHOD == 'pyrubberband':
        # 修正版：只使用字典格式，並且避開會導致報錯的 Boolean 值
        
        if quality == 'speech':
            try:
                print("  → 嘗試語音優化參數 (高清晰度)...")
                # 修正重點：
                # 1. 移除 '--formant': True (這會導致 rubberband 參數錯誤)
                # 2. 保留 '-c': 6。這是 "Crispness" (清晰度)，設為 6 能最大程度減少混響
                #    (如果不設這個，聲音就會像在空曠教室)
                result = pyrb.time_stretch(
                    audio_data, 
                    sample_rate, 
                    speed_factor,
                    rbargs={'-c': 6}
                )
                print("  ✓ 語音優化參數成功")
                return result
            except Exception as e:
                print(f"  × 語音優化參數失敗: {e}")
                # 如果這裡失敗，才會往下走，但通常 -c 6 不會失敗

        # 基本模式 (最後手段)
        try:
            print("  → 降級使用基本模式 (注意：可能會有混響)...")
            return pyrb.time_stretch(audio_data, sample_rate, speed_factor)
        except Exception as e:
            print(f"  × pyrubberband 所有模式皆失敗: {e}")
    
    elif SPEED_METHOD == 'librosa':
        print("  → 使用 librosa 進行變速...")
        return librosa.effects.time_stretch(audio_data, rate=speed_factor)
    
    raise RuntimeError("無可用的變速方法")

def adjust_audio_speed_final(input_path, output_path, speed_factor, quality='speech'):
    """
    對音檔進行變速處理（最終修正版）
    
    Args:
        input_path: 輸入音檔路徑
        output_path: 輸出音檔路徑
        speed_factor: 變速倍率（>1 加速, <1 減速）
        quality: 品質模式 ('speech' 或 'music')
    """
    print(f"\n{'='*70}")
    print(f"🎵 參考音檔變速處理 (最終修正版)")
    print(f"{'='*70}")
    
    # 讀取音檔
    print(f"📂 讀取: {input_path}")
    audio_data, sample_rate = sf.read(input_path)
    
    # 獲取原始資訊
    original_duration = len(audio_data) / sample_rate
    print(f"  • 採樣率: {sample_rate} Hz")
    print(f"  • 原始長度: {original_duration:.2f} 秒")
    print(f"  • 聲道: {'單聲道' if len(audio_data.shape) == 1 else f'{audio_data.shape[1]} 聲道'}")
    
    # 確保是單聲道
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        print(f"  ⚠️ 已轉換為單聲道")
    
    # 變速處理
    print(f"\n⚡ 變速處理中... (倍率: {speed_factor}x, 品質: {quality})")
    print(f"  • 使用方法: {SPEED_METHOD}")
    
    try:
        adjusted_audio = time_stretch_robust(audio_data, sample_rate, speed_factor, quality)
        adjusted_duration = len(adjusted_audio) / sample_rate
        
        print(f"\n  ✅ 變速完成")
        print(f"  • 變速後長度: {adjusted_duration:.2f} 秒")
        print(f"  • 理論長度: {original_duration / speed_factor:.2f} 秒")
        print(f"  • 實際壓縮率: {original_duration / adjusted_duration:.2f}x")
        
    except Exception as e:
        print(f"\n  ❌ 變速失敗: {e}")
        raise
    
    # 保存
    print(f"\n💾 保存至: {output_path}")
    sf.write(output_path, adjusted_audio, sample_rate)
    
    # 驗證
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  • 檔案大小: {file_size:.2f} MB")
    print(f"  ✅ 完成!")
    
    return output_path, adjusted_duration


def main():
    parser = argparse.ArgumentParser(
        description="參考音檔變速工具 - 最終修正版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本用法
  python prepare_speed_ref_final.py -i voice_06.wav -s 1.3 -o voice_06_1.3x.wav
  
  # 指定品質
  python prepare_speed_ref_final.py -i voice_06.wav -s 1.3 -o output.wav -q speech

說明:
  - 自動選擇最佳變速方法 (pyrubberband 或 librosa)
  - 支援多種參數格式 fallback
  - 語音優化（如果支援）
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="輸入參考音檔路徑")
    parser.add_argument("--output", "-o", required=True, help="輸出音檔路徑")
    parser.add_argument("--speed", "-s", type=float, required=True, help="變速倍率 (例如 1.3)")
    parser.add_argument("--quality", "-q", type=str, default="speech", 
                        choices=["speech", "music"],
                        help="品質模式 (預設: speech)")
    
    args = parser.parse_args()
    
    # 檢查輸入檔案
    if not os.path.exists(args.input):
        print(f"❌ 錯誤: 找不到輸入檔案 {args.input}")
        sys.exit(1)
    
    # 執行變速
    try:
        adjust_audio_speed_final(args.input, args.output, args.speed, args.quality)
        
        print(f"\n{'='*70}")
        print(f"💡 下一步:")
        print(f"{'='*70}")
        print(f"1. 聆聽輸出檔案: {args.output}")
        print(f"2. 使用變速後的音檔測試 TTS:")
        print(f"   python test_streaming.py --ref_audio {args.output} --speed 1.0")
        
        if SPEED_METHOD == 'pyrubberband':
            print(f"\n💬 注意:")
            print(f"   pyrubberband 可能使用預設參數（可能有輕微混響）")
            print(f"   如需更好音質，建議安裝 librosa: pip install librosa")
        
    except Exception as e:
        print(f"\n❌ 處理失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

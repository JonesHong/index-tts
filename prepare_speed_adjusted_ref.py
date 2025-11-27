#!/usr/bin/env python3
"""
參考音檔變速工具

功能：
1. 對參考音檔進行變速處理並保存
2. 可選：直接測試變速參考音檔的 TTS 效果

使用方式：
    # 僅生成變速音檔
    python prepare_speed_adjusted_ref.py --input voice_06.wav --speed 1.3 --output voice_06_1.3x.wav
    
    # 生成並直接測試
    python prepare_speed_adjusted_ref.py --input voice_06.wav --speed 1.3 --output voice_06_1.3x.wav --test
"""

import sys
import os
import argparse
import numpy as np
import soundfile as sf

try:
    import pyrubberband as pyrb
except ImportError:
    print("錯誤: 需要安裝 pyrubberband")
    print("請執行: pip install pyrubberband")
    sys.exit(1)


def adjust_audio_speed(input_path, output_path, speed_factor):
    """
    對音檔進行變速處理
    
    Args:
        input_path: 輸入音檔路徑
        output_path: 輸出音檔路徑
        speed_factor: 變速倍率（>1 加速, <1 減速）
    """
    print(f"\n{'='*60}")
    print(f"🎵 參考音檔變速處理")
    print(f"{'='*60}")
    
    # 讀取音檔
    print(f"📂 讀取: {input_path}")
    audio_data, sample_rate = sf.read(input_path)
    
    # 獲取原始資訊
    original_duration = len(audio_data) / sample_rate
    print(f"  • 採樣率: {sample_rate} Hz")
    print(f"  • 原始長度: {original_duration:.2f} 秒")
    print(f"  • 聲道: {'單聲道' if len(audio_data.shape) == 1 else f'{audio_data.shape[1]} 聲道'}")
    
    # 確保是單聲道（TTS 通常需要單聲道）
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        print(f"  ⚠️ 已轉換為單聲道")
    
    # 變速處理
    print(f"\n⚡ 變速處理中... (倍率: {speed_factor}x)")
    adjusted_audio = pyrb.time_stretch(audio_data, sample_rate, speed_factor)
    adjusted_duration = len(adjusted_audio) / sample_rate
    
    print(f"  • 變速後長度: {adjusted_duration:.2f} 秒")
    print(f"  • 理論長度: {original_duration / speed_factor:.2f} 秒")
    print(f"  • 實際壓縮率: {original_duration / adjusted_duration:.2f}x")
    
    # 保存
    print(f"\n💾 保存至: {output_path}")
    sf.write(output_path, adjusted_audio, sample_rate)
    
    # 驗證
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  • 檔案大小: {file_size:.2f} MB")
    print(f"  ✅ 完成!")
    
    return output_path, adjusted_duration


def test_with_tts(ref_audio_path, test_text=None, version="v2", method="token"):
    """
    使用變速後的參考音檔進行 TTS 測試
    """
    print(f"\n{'='*60}")
    print(f"🧪 TTS 測試 (使用變速後的參考音檔)")
    print(f"{'='*60}")
    
    # 導入 test_streaming.py 的相關邏輯
    try:
        import subprocess
        test_script = "test_streaming.py"
        
        if not os.path.exists(test_script):
            print(f"⚠️ 找不到 {test_script}")
            print(f"請確保 {test_script} 在同一目錄下，或手動執行：")
            print(f"  python test_streaming.py --ref_audio {ref_audio_path} --speed 1.0")
            return
        
        # 構建命令
        cmd = [
            "python", test_script,
            "--ref_audio", ref_audio_path,
            "--speed", "1.0",  # 重要：這裡不再變速
            "--version", version,
            "--method", method,
        ]
        
        if test_text:
            cmd.extend(["--text", test_text])
        
        print(f"執行命令: {' '.join(cmd)}\n")
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        print(f"\n手動測試方式:")
        print(f"  python test_streaming.py --ref_audio {ref_audio_path} --speed 1.0")


def batch_process(input_path, speeds, output_dir="speed_adjusted_refs"):
    """
    批次處理多個變速倍率
    """
    print(f"\n{'='*60}")
    print(f"📦 批次變速處理")
    print(f"{'='*60}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    basename = os.path.splitext(os.path.basename(input_path))[0]
    
    for speed in speeds:
        output_filename = f"{basename}_{speed:.1f}x.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n處理 {speed}x...")
        adjusted_path, duration = adjust_audio_speed(input_path, output_path, speed)
        results.append({
            'speed': speed,
            'path': adjusted_path,
            'duration': duration
        })
    
    print(f"\n{'='*60}")
    print(f"✅ 批次處理完成")
    print(f"{'='*60}")
    print(f"共生成 {len(results)} 個變速音檔:")
    for r in results:
        print(f"  • {r['speed']:.1f}x → {r['path']} ({r['duration']:.2f}s)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="參考音檔變速工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 單個變速
  python prepare_speed_adjusted_ref.py --input voice_06.wav --speed 1.3 --output voice_06_1.3x.wav
  
  # 變速後立即測試
  python prepare_speed_adjusted_ref.py --input voice_06.wav --speed 1.3 --output voice_06_1.3x.wav --test
  
  # 批次生成多個變速版本
  python prepare_speed_adjusted_ref.py --input voice_06.wav --batch 1.0,1.2,1.3,1.5
  
比較測試流程:
  1. 生成變速參考音檔 (speed 1.3x)
  2. 方式 A: 使用原始參考音檔 + 播放時變速 1.3x
     python test_streaming.py --ref_audio voice_06.wav --speed 1.3
  
  3. 方式 B: 使用變速參考音檔 + 正常播放
     python test_streaming.py --ref_audio voice_06_1.3x.wav --speed 1.0
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="輸入參考音檔路徑")
    parser.add_argument("--output", "-o", help="輸出音檔路徑 (單個變速時使用)")
    parser.add_argument("--speed", "-s", type=float, help="變速倍率 (例如 1.3)")
    parser.add_argument("--batch", "-b", type=str, help="批次變速，以逗號分隔 (例如 1.0,1.2,1.3,1.5)")
    parser.add_argument("--test", action="store_true", help="變速後立即用 TTS 測試")
    parser.add_argument("--text", type=str, help="測試用文字")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="TTS 版本")
    parser.add_argument("--method", type=str, default="token", choices=["token", "word"], help="切分方法")
    
    args = parser.parse_args()
    
    # 檢查輸入檔案
    if not os.path.exists(args.input):
        print(f"❌ 錯誤: 找不到輸入檔案 {args.input}")
        sys.exit(1)
    
    # 批次處理模式
    if args.batch:
        speeds = [float(s.strip()) for s in args.batch.split(',')]
        batch_process(args.input, speeds)
        return
    
    # 單個變速模式
    if not args.speed:
        print("❌ 錯誤: 請指定 --speed 或 --batch")
        parser.print_help()
        sys.exit(1)
    
    # 生成輸出路徑
    if not args.output:
        basename = os.path.splitext(args.input)[0]
        args.output = f"{basename}_{args.speed:.1f}x.wav"
    
    # 執行變速
    adjusted_path, _ = adjust_audio_speed(args.input, args.output, args.speed)
    
    # 可選：測試
    if args.test:
        test_with_tts(adjusted_path, args.text, args.version, args.method)
    else:
        print(f"\n{'='*60}")
        print(f"💡 提示:")
        print(f"{'='*60}")
        print(f"使用變速後的參考音檔進行測試:")
        print(f"  python test_streaming.py --ref_audio {adjusted_path} --speed 1.0")
        print(f"\n比較原始方式:")
        print(f"  python test_streaming.py --ref_audio {args.input} --speed {args.speed}")


if __name__ == "__main__":
    main()

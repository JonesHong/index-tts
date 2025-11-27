#!/usr/bin/env python3
"""
IndexTTS 測試啟動器 (Windows + uv 版本)
================================================================================
使用方式:
    uv run run_tests_launcher.py
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
import time

# ANSI 顏色碼 (Windows 10+ 支持)
try:
    import colorama
    colorama.init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

if HAS_COLOR:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    NC = '\033[0m'
else:
    RED = GREEN = YELLOW = BLUE = NC = ''

def print_header(text):
    """印出標題"""
    print(f"\n{BLUE}{'='*80}{NC}")
    print(f"{BLUE}{text}{NC}")
    print(f"{BLUE}{'='*80}{NC}\n")

def print_success(text):
    """印出成功訊息"""
    print(f"{GREEN}✓{NC} {text}")

def print_error(text):
    """印出錯誤訊息"""
    print(f"{RED}✗{NC} {text}")

def print_warning(text):
    """印出警告訊息"""
    print(f"{YELLOW}⚠{NC}  {text}")

def print_info(text):
    """印出資訊訊息"""
    print(f"  {text}")

def check_command(cmd):
    """檢查命令是否可用"""
    return shutil.which(cmd) is not None

def check_python_package(package_name):
    """檢查 Python 套件是否已安裝"""
    try:
        __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def main():
    print_header("IndexTTS 全面性能測試 (Windows + uv)")

    # 檢查 Python 版本
    print(f"Python 版本: {sys.version.split()[0]}")
    if sys.version_info < (3, 7):
        print_error("需要 Python 3.7 或更高版本")
        return 1
    print_success(f"Python {sys.version.split()[0]}")

    # 檢查依賴
    print("\n" + YELLOW + "檢查依賴..." + NC)

    required_packages = [
        ('pyrubberband', 'pyrubberband'),
        ('librosa', 'librosa'),
        ('opencc', 'opencc-python-reimplemented'),
        ('sounddevice', 'sounddevice'),
        ('soundfile', 'soundfile'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
    ]

    missing_required = []
    for import_name, package_name in required_packages:
        if check_python_package(import_name):
            print_success(package_name)
        else:
            print_error(package_name)
            missing_required.append(package_name)

    if missing_required:
        print()
        print_error("缺少必要依賴:")
        for pkg in missing_required:
            print(f"   - {pkg}")
        print()
        print("請執行:")
        print(f"   uv pip install {' '.join(missing_required)}")
        return 1

    # 檢查可選依賴
    print()
    optional_packages = [
        ('matplotlib', 'matplotlib'),
        ('psutil', 'psutil'),
    ]

    missing_optional = []
    for import_name, package_name in optional_packages:
        if not check_python_package(import_name):
            missing_optional.append(package_name)

    if missing_optional:
        print_warning("缺少可選依賴 (不影響主要功能):")
        for pkg in missing_optional:
            print(f"   - {pkg}")
        print()
        print("建議安裝以啟用完整功能:")
        print(f"   uv pip install {' '.join(missing_optional)}")
        print()

    # 檢查參考音檔
    print(YELLOW + "檢查參考音檔..." + NC)

    examples_dir = Path(__file__).parent / "examples"
    required_audio = ["voice_06.wav", "voice_07.wav"]
    missing_audio = []

    for audio_file in required_audio:
        audio_path = examples_dir / audio_file
        if audio_path.exists():
            print_success(f"examples/{audio_file}")
        else:
            print_error(f"examples/{audio_file}")
            missing_audio.append(audio_file)

    if missing_audio:
        print()
        print_error("找不到必要的參考音檔")
        return 1

    # 檢查測試腳本
    print()
    print(YELLOW + "檢查測試腳本..." + NC)

    script_dir = Path(__file__).parent
    required_scripts = [
        "test_streaming_with_output.py",
        "run_comprehensive_tests.py"
    ]

    for script in required_scripts:
        script_path = script_dir / script
        if script_path.exists():
            print_success(script)
        else:
            print_error(script)
            return 1

    # 檢查 CUDA
    print()
    print(YELLOW + "檢查 CUDA 狀態..." + NC)

    if check_command("nvidia-smi"):
        print_success("CUDA 可用")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"   {result.stdout.strip().split(',')[0]}")
        except Exception:
            pass
    else:
        print_warning("CUDA 不可用 (將使用 CPU，測試會較慢)")

    # 測試配置
    print_header("測試配置")

    print("📋 測試套件:")
    print("   1️⃣  Voice Comparison (voice_06 vs voice_07)")
    print("   2️⃣  Speed Strategy (No/Pre/Post/Hybrid)")
    print("   3️⃣  Version & Mode (v1/v2, token/word)")
    print()
    print("📊 輸出內容:")
    print("   • CSV 數據表格")
    print("   • JSON 詳細日誌")
    print("   • 2 張視覺化圖表 (PNG)")
    print("   • 文字摘要報告")
    print("   • 4 個音檔樣本 (Test Suite 2)")
    print()
    print("⏱️  預估時間: 30-60 分鐘 (視硬體性能而定)")
    print("💾 輸出目錄: benchmark_output/")
    print()

    # 確認執行
    try:
        response = input(f"{GREEN}是否開始測試? [y/N]: {NC}").strip().lower()
        if response not in ['y', 'yes']:
            print(f"\n{YELLOW}測試已取消{NC}")
            return 0
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}測試已取消{NC}")
        return 0

    # 創建輸出目錄
    output_dir = script_dir / "benchmark_output"
    audio_dir = output_dir / "audio_samples"
    output_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    # 執行測試
    print_header("開始執行測試...")

    start_time = time.time()

    try:
        test_script = script_dir / "run_comprehensive_tests.py"
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=script_dir
        )

        if result.returncode != 0:
            print()
            print_error("測試執行失敗")
            return result.returncode

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}測試被使用者中斷{NC}")
        return 1
    except Exception as e:
        print()
        print_error(f"執行錯誤: {e}")
        return 1

    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    # 測試完成
    print_header("✅ 測試完成!")

    print(f"⏱️  總耗時: {minutes} 分 {seconds} 秒")
    print()
    print(f"📂 結果已保存至: {output_dir}")
    print()

    # 列出生成的文件
    print("生成的文件:")
    for ext in ['*.csv', '*.json', '*.txt', '*.png']:
        for file in output_dir.glob(ext):
            size_mb = file.stat().st_size / 1024 / 1024
            if size_mb >= 1:
                print(f"   {file.name} ({size_mb:.1f} MB)")
            else:
                size_kb = file.stat().st_size / 1024
                print(f"   {file.name} ({size_kb:.1f} KB)")

    print()
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav"))
        if audio_files:
            print("音檔樣本:")
            for file in audio_files:
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"   {file.name} ({size_mb:.1f} MB)")
            print()

    print(f"{GREEN}可以開始分析結果了! 🎉{NC}")
    print()
    print("建議步驟:")
    print("   1. 檢視 CSV 文件 (用 Excel 打開)")
    print("   2. 查看圖表 (*.png)")
    print("   3. 閱讀摘要報告 (*_summary_report_*.txt)")
    print("   4. 聆聽音檔樣本 (audio_samples/*.wav)")
    print()
    print(f"詳細說明請參閱: {BLUE}TEST_GUIDE.md{NC}")
    print()

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}程式被中斷{NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}發生未預期的錯誤: {e}{NC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

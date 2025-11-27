@echo off
REM IndexTTS 測試執行腳本 (Windows 版本)
REM 使用 uv 管理依賴

setlocal enabledelayedexpansion

echo ========================================
echo IndexTTS 全面性能測試 (Windows)
echo ========================================
echo.

REM 檢查 uv
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [91m錯誤: 找不到 uv[0m
    echo 請先安裝 uv: https://github.com/astral-sh/uv
    pause
    exit /b 1
)

echo [92m✓[0m uv 已安裝

REM 檢查 Python
uv run python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [91m錯誤: uv run python 失敗[0m
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('uv run python --version') do set PYTHON_VERSION=%%i
echo [92m✓[0m %PYTHON_VERSION%

REM 檢查依賴
echo.
echo [93m檢查依賴...[0m

set MISSING_DEPS=0

uv run python -c "import pyrubberband" >nul 2>nul
if %errorlevel% neq 0 (
    echo [91m  ✗ pyrubberband[0m
    set MISSING_DEPS=1
) else (
    echo [92m  ✓ pyrubberband[0m
)

uv run python -c "import librosa" >nul 2>nul
if %errorlevel% neq 0 (
    echo [91m  ✗ librosa[0m
    set MISSING_DEPS=1
) else (
    echo [92m  ✓ librosa[0m
)

uv run python -c "import opencc" >nul 2>nul
if %errorlevel% neq 0 (
    echo [91m  ✗ opencc-python-reimplemented[0m
    set MISSING_DEPS=1
) else (
    echo [92m  ✓ opencc-python-reimplemented[0m
)

uv run python -c "import sounddevice" >nul 2>nul
if %errorlevel% neq 0 (
    echo [91m  ✗ sounddevice[0m
    set MISSING_DEPS=1
) else (
    echo [92m  ✓ sounddevice[0m
)

uv run python -c "import soundfile" >nul 2>nul
if %errorlevel% neq 0 (
    echo [91m  ✗ soundfile[0m
    set MISSING_DEPS=1
) else (
    echo [92m  ✓ soundfile[0m
)

if %MISSING_DEPS%==1 (
    echo.
    echo [91m缺少必要依賴[0m
    echo.
    echo 請執行:
    echo   uv pip install pyrubberband librosa opencc-python-reimplemented sounddevice soundfile
    echo.
    pause
    exit /b 1
)

echo [92m✓[0m 所有必要依賴已安裝

REM 檢查可選依賴
echo.
set OPTIONAL_MISSING=0

uv run python -c "import matplotlib" >nul 2>nul
if %errorlevel% neq 0 (
    echo [93m⚠  matplotlib 未安裝 (圖表生成將被跳過)[0m
    set OPTIONAL_MISSING=1
)

uv run python -c "import psutil" >nul 2>nul
if %errorlevel% neq 0 (
    echo [93m⚠  psutil 未安裝 (記憶體監控將被跳過)[0m
    set OPTIONAL_MISSING=1
)

if %OPTIONAL_MISSING%==1 (
    echo.
    echo [93m建議安裝以啟用完整功能:[0m
    echo   uv pip install matplotlib psutil
    echo.
)

REM 檢查參考音檔
echo [93m檢查參考音檔...[0m

set MISSING_AUDIO=0

if not exist "examples\voice_06.wav" (
    echo [91m  ✗ examples\voice_06.wav[0m
    set MISSING_AUDIO=1
) else (
    echo [92m  ✓ examples\voice_06.wav[0m
)

if not exist "examples\voice_07.wav" (
    echo [91m  ✗ examples\voice_07.wav[0m
    set MISSING_AUDIO=1
) else (
    echo [92m  ✓ examples\voice_07.wav[0m
)

if %MISSING_AUDIO%==1 (
    echo.
    echo [91m找不到必要的參考音檔[0m
    pause
    exit /b 1
)

echo [92m✓[0m 參考音檔檢查通過

REM 檢查測試腳本
echo.
echo [93m檢查測試腳本...[0m

if not exist "test_streaming_with_output.py" (
    echo [91m✗ test_streaming_with_output.py[0m
    pause
    exit /b 1
)
echo [92m  ✓ test_streaming_with_output.py[0m

if not exist "run_comprehensive_tests.py" (
    echo [91m✗ run_comprehensive_tests.py[0m
    pause
    exit /b 1
)
echo [92m  ✓ run_comprehensive_tests.py[0m

REM 檢查 CUDA
echo.
echo [93m檢查 CUDA 狀態...[0m

where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [92m✓[0m CUDA 可用
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul | findstr /n "^" | findstr "^1:"
) else (
    echo [93m⚠  CUDA 不可用 (將使用 CPU，測試會較慢)[0m
)

REM 測試配置
echo.
echo ========================================
echo 測試配置
echo ========================================
echo.
echo 📋 測試套件:
echo    1️⃣  Voice Comparison (voice_06 vs voice_07)
echo    2️⃣  Speed Strategy (No/Pre/Post/Hybrid)
echo    3️⃣  Version ^& Mode (v1/v2, token/word)
echo.
echo 📊 輸出內容:
echo    • CSV 數據表格
echo    • JSON 詳細日誌
echo    • 2 張視覺化圖表 (PNG)
echo    • 文字摘要報告
echo    • 4 個音檔樣本 (Test Suite 2)
echo.
echo ⏱️  預估時間: 30-60 分鐘 (視硬體性能而定)
echo 💾 輸出目錄: benchmark_output\
echo.

REM 確認執行
set /p CONFIRM="是否開始測試? [y/N]: "
if /i not "%CONFIRM%"=="y" (
    echo [93m測試已取消[0m
    pause
    exit /b 0
)

REM 創建輸出目錄
if not exist "benchmark_output" mkdir benchmark_output
if not exist "benchmark_output\audio_samples" mkdir benchmark_output\audio_samples

REM 執行測試
echo.
echo ========================================
echo 開始執行測試...
echo ========================================
echo.

set START_TIME=%TIME%

uv run python run_comprehensive_tests.py

set END_TIME=%TIME%

REM 計算耗時 (簡化版本)
echo.
echo ========================================
echo [92m✅ 測試完成![0m
echo ========================================
echo.
echo ⏱️  開始時間: %START_TIME%
echo ⏱️  結束時間: %END_TIME%
echo.

echo 📂 結果已保存至: benchmark_output\
echo.

REM 列出生成的文件
echo 生成的文件:
dir /b benchmark_output\*.csv benchmark_output\*.json benchmark_output\*.txt benchmark_output\*.png 2>nul
echo.

if exist "benchmark_output\audio_samples" (
    echo 音檔樣本:
    dir /b benchmark_output\audio_samples\*.wav 2>nul
    echo.
)

echo [92m可以開始分析結果了! 🎉[0m
echo.
echo 建議步驟:
echo    1. 檢視 CSV 文件 (用 Excel 打開)
echo    2. 查看圖表 (*.png)
echo    3. 閱讀摘要報告 (*_summary_report_*.txt)
echo    4. 聆聽音檔樣本 (audio_samples\*.wav)
echo.
echo 詳細說明請參閱: TEST_GUIDE.md
echo.

pause

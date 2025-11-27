#!/bin/bash
#
# IndexTTS 測試執行腳本
# 快速執行全面性能測試
#

set -e  # 遇到錯誤立即退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 腳本目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IndexTTS 全面性能測試${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 檢查 Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ 錯誤: 找不到 Python${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python: $(python --version)"

# 檢查依賴
echo ""
echo -e "${YELLOW}檢查依賴...${NC}"

MISSING_DEPS=()

# 基本依賴
for pkg in pyrubberband librosa opencc sounddevice soundfile; do
    if ! python -c "import ${pkg//-/_}" &> /dev/null; then
        MISSING_DEPS+=("$pkg")
    fi
done

# 可選依賴
OPTIONAL_MISSING=()
for pkg in matplotlib psutil; do
    if ! python -c "import $pkg" &> /dev/null; then
        OPTIONAL_MISSING+=("$pkg")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${RED}❌ 缺少必要依賴:${NC}"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "   - $dep"
    done
    echo ""
    echo -e "${YELLOW}請執行:${NC}"
    echo "   pip install ${MISSING_DEPS[*]}"
    exit 1
fi

echo -e "${GREEN}✓${NC} 所有必要依賴已安裝"

if [ ${#OPTIONAL_MISSING[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  缺少可選依賴 (不影響主要功能):${NC}"
    for dep in "${OPTIONAL_MISSING[@]}"; do
        echo "   - $dep"
    done
    echo ""
    echo -e "${YELLOW}建議安裝以啟用完整功能:${NC}"
    echo "   pip install ${OPTIONAL_MISSING[*]}"
    echo ""
fi

# 檢查參考音檔
echo ""
echo -e "${YELLOW}檢查參考音檔...${NC}"

MISSING_AUDIO=()
for audio in "examples/voice_06.wav" "examples/voice_07.wav"; do
    if [ ! -f "$audio" ]; then
        MISSING_AUDIO+=("$audio")
    fi
done

if [ ${#MISSING_AUDIO[@]} -gt 0 ]; then
    echo -e "${RED}❌ 找不到必要的參考音檔:${NC}"
    for audio in "${MISSING_AUDIO[@]}"; do
        echo "   - $audio"
    done
    exit 1
fi

echo -e "${GREEN}✓${NC} 參考音檔檢查通過"

# 檢查測試腳本
echo ""
echo -e "${YELLOW}檢查測試腳本...${NC}"

if [ ! -f "test_streaming_with_output.py" ]; then
    echo -e "${RED}❌ 找不到 test_streaming_with_output.py${NC}"
    exit 1
fi

if [ ! -f "run_comprehensive_tests.py" ]; then
    echo -e "${RED}❌ 找不到 run_comprehensive_tests.py${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} 測試腳本檢查通過"

# 檢查 CUDA
echo ""
echo -e "${YELLOW}檢查 CUDA 狀態...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} CUDA 可用"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠️  CUDA 不可用 (將使用 CPU，測試會較慢)${NC}"
fi

# 估算測試時間
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}測試配置${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "📋 測試套件:"
echo "   1️⃣  Voice Comparison (voice_06 vs voice_07)"
echo "   2️⃣  Speed Strategy (No/Pre/Post/Hybrid)"
echo "   3️⃣  Version & Mode (v1/v2, token/word)"
echo ""
echo "📊 輸出內容:"
echo "   • CSV 數據表格"
echo "   • JSON 詳細日誌"
echo "   • 2 張視覺化圖表 (PNG)"
echo "   • 文字摘要報告"
echo "   • 4 個音檔樣本 (Test Suite 2)"
echo ""
echo "⏱️  預估時間: 30-60 分鐘 (視硬體性能而定)"
echo "💾 輸出目錄: benchmark_output/"
echo ""

# 確認執行
read -p "$(echo -e ${GREEN}是否開始測試? [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}測試已取消${NC}"
    exit 0
fi

# 創建輸出目錄
mkdir -p benchmark_output/audio_samples

# 執行測試
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}開始執行測試...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

START_TIME=$(date +%s)

python run_comprehensive_tests.py

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ 測試完成!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "⏱️  總耗時: ${MINUTES} 分 ${SECONDS} 秒"
echo ""
echo "📂 結果已保存至: benchmark_output/"
echo ""

# 列出生成的文件
echo "生成的文件:"
ls -lh benchmark_output/*.{csv,json,txt,png} 2>/dev/null | awk '{printf "   %s %s %s\n", $9, "("$5")", $6" "$7" "$8}'

if [ -d "benchmark_output/audio_samples" ]; then
    echo ""
    echo "音檔樣本:"
    ls -lh benchmark_output/audio_samples/*.wav 2>/dev/null | awk '{printf "   %s %s %s\n", $9, "("$5")", $6" "$7" "$8}'
fi

echo ""
echo -e "${GREEN}可以開始分析結果了! 🎉${NC}"
echo ""
echo "建議步驟:"
echo "   1. 檢視 CSV 文件 (用 Excel/Numbers 打開)"
echo "   2. 查看圖表 (*.png)"
echo "   3. 閱讀摘要報告 (*_summary_report_*.txt)"
echo "   4. 聆聽音檔樣本 (audio_samples/*.wav)"
echo ""
echo -e "詳細說明請參閱: ${BLUE}TEST_GUIDE.md${NC}"
echo ""

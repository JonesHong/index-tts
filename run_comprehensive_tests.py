#!/usr/bin/env python3
"""
Comprehensive Test Runner for IndexTTS Streaming Performance Analysis
================================================================================
測試維度:
1. 首次響應時間 (TTFB - Time To First Byte)
2. 總生成時間
3. 生成倍率 (速度)
4. 記憶體使用
5. 音質穩定性
6. 總耗時
7. 使用者感知延遲
8. 資源使用效率

輸出:
- CSV 統計數據表格
- JSON 詳細測試日誌
- 2 張視覺化比較圖表 (PNG)
"""

import sys
import os
import subprocess
import json
import csv
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# 可選依賴 - 用於視覺化
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非互動式後端
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib 未安裝，將跳過圖表生成")
    print("   安裝方式: pip install matplotlib")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil 未安裝，將跳過記憶體監控")
    print("   安裝方式: pip install psutil")

# ==================== 配置區 ====================

# 測試文本 (統一使用，確保可比性)
DEFAULT_TEXT = (
    "劉佩真分析，行政院「開水龍頭」，9月初新青安鬆綁，及延長對先買後賣換屋族出售舊屋的期限，"
    "觀望的市場氛圍稍減，房市交易量出現小幅成長，"
    "事實上，今年房市的交易結構已從去年的價量齊揚，到今年的量縮、價格緩跌。"
    "目前房價的跌幅方面，相較於去年還有非常低個位數的下滑，顯示房市賣方實際上沒有出脫的壓力。"
)

# 測試腳本路徑
SCRIPT_DIR = Path(__file__).parent
TEST_SCRIPT = SCRIPT_DIR / "test_streaming_with_output.py"  # 使用支持輸出的版本
OUTPUT_DIR = SCRIPT_DIR / "benchmark_output"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio_samples"

# 參考音檔路徑
INDEX_TTS_DIR = SCRIPT_DIR
REF_AUDIO_DIR = INDEX_TTS_DIR / "examples"

# 確保輸出目錄存在
OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 測試配置 ====================

# Test Suite 1: Voice Comparison (voice_06 vs voice_07)
TEST_SUITE_1 = [
    {
        "name": "voice_06_baseline",
        "description": "Voice 06 - Baseline (Default Parameters)",
        "args": [
            "--version", "v2",
            "--method", "token",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_06.wav"),
            "--text", DEFAULT_TEXT,
            "--warmup"
        ]
    },
    {
        "name": "voice_07_baseline",
        "description": "Voice 07 - Baseline (Default Parameters)",
        "args": [
            "--version", "v2",
            "--method", "token",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_07.wav"),
            "--text", DEFAULT_TEXT,
            "--warmup"
        ]
    }
]

# Test Suite 2: Speed Strategy Comparison (voice_07)
TEST_SUITE_2 = [
    {
        "name": "voice_07_no_speed",
        "description": "Voice 07 - No Speed Modification",
        "args": [
            "--version", "v2",
            "--method", "token",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_07.wav"),
            "--text", DEFAULT_TEXT,
            "--warmup",
            "--output", str(AUDIO_OUTPUT_DIR / "voice_07_no_speed.wav")
        ]
    },
    {
        "name": "voice_07_pre_speed_1.2x",
        "description": "Voice 07 - Pre-Speed 1.2x (Reference Audio Acceleration)",
        "args": [
            "--version", "v2",
            "--method", "token",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_07.wav"),
            "--text", DEFAULT_TEXT,
            "--pre_speed_ref", "1.2",
            "--warmup",
            "--output", str(AUDIO_OUTPUT_DIR / "voice_07_pre_speed_1.2x.wav")
        ]
    },
    {
        "name": "voice_07_post_speed_1.2x",
        "description": "Voice 07 - Post-Speed 1.2x (Playback Acceleration)",
        "args": [
            "--version", "v2",
            "--method", "token",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_07.wav"),
            "--text", DEFAULT_TEXT,
            "--speed", "1.2",
            "--warmup",
            "--output", str(AUDIO_OUTPUT_DIR / "voice_07_post_speed_1.2x.wav")
        ]
    },
    {
        "name": "voice_07_hybrid_speed_1.2x",
        "description": "Voice 07 - Hybrid Speed 1.2x (Pre + Post)",
        "args": [
            "--version", "v2",
            "--method", "token",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_07.wav"),
            "--text", DEFAULT_TEXT,
            "--pre_speed_ref", "1.2",
            "--speed", "1.2",
            "--warmup",
            "--output", str(AUDIO_OUTPUT_DIR / "voice_07_hybrid_speed_1.2x.wav")
        ]
    }
]

# Test Suite 3: Version & Mode Comparison (voice_07)
TEST_SUITE_3 = [
    {
        "name": "v1_streaming",
        "description": "V1 - Streaming Mode",
        "args": [
            "--version", "v1",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_07.wav"),
            "--text", DEFAULT_TEXT,
            "--warmup"
        ]
    },
    {
        "name": "v2_streaming_token",
        "description": "V2 - Streaming Mode (Token-based)",
        "args": [
            "--version", "v2",
            "--method", "token",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_07.wav"),
            "--text", DEFAULT_TEXT,
            "--warmup"
        ]
    },
    {
        "name": "v2_streaming_word",
        "description": "V2 - Streaming Mode (Word-based)",
        "args": [
            "--version", "v2",
            "--method", "word",
            "--ref_audio", str(REF_AUDIO_DIR / "voice_07.wav"),
            "--text", DEFAULT_TEXT,
            "--warmup"
        ]
    }
]

# ==================== 解析函數 ====================

def parse_test_output(output: str) -> Dict[str, Any]:
    """
    解析測試輸出，提取關鍵指標
    """
    metrics = {
        "ttfb": None,  # Time to first byte (首次響應時間)
        "total_time": None,  # 總耗時
        "avg_gen_rate": None,  # 平均生成倍率
        "max_gen_rate": None,  # 最大生成倍率
        "min_gen_rate": None,  # 最小生成倍率
        "overall_rtf": None,  # 整體實時率
        "parallel_efficiency": None,  # 並行效率
        "chunk_count": 0,  # 音訊片段數
        "warmup_time": None,  # 預熱時間
        "model_load_time": None,  # 模型載入時間
        "ref_audio_duration": None,  # 參考音檔長度
        "ref_audio_size_mb": None,  # 參考音檔大小
        "pre_speed_enabled": False,  # 是否使用預加速
        "post_speed_enabled": False,  # 是否使用後處理加速
        "error": None  # 錯誤信息
    }

    try:
        # TTFB (首次響應)
        ttfb_match = re.search(r'\[⚡ First Token\].*?(\d+\.\d+)s', output)
        if ttfb_match:
            metrics["ttfb"] = float(ttfb_match.group(1))

        # 總耗時
        total_match = re.search(r'總耗時:\s*(\d+\.\d+)\s*s', output)
        if total_match:
            metrics["total_time"] = float(total_match.group(1))

        # 生成倍率 (Audio/Process Speed)
        avg_rate_match = re.search(r'Avg\s*:\s*(\d+\.\d+)\s*x', output)
        if avg_rate_match:
            metrics["avg_gen_rate"] = float(avg_rate_match.group(1))

        max_rate_match = re.search(r'Max\s*:\s*(\d+\.\d+)\s*x', output)
        if max_rate_match:
            metrics["max_gen_rate"] = float(max_rate_match.group(1))

        min_rate_match = re.search(r'Min\s*:\s*(\d+\.\d+)\s*x', output)
        if min_rate_match:
            metrics["min_gen_rate"] = float(min_rate_match.group(1))

        # 整體實時率 (RTF)
        rtf_match = re.search(r'整體實時率.*?RTF.*?(\d+\.\d+)', output)
        if rtf_match:
            metrics["overall_rtf"] = float(rtf_match.group(1))

        # 並行效率
        parallel_match = re.search(r'並行效率.*?(\d+\.\d+)%', output)
        if parallel_match:
            metrics["parallel_efficiency"] = float(parallel_match.group(1))

        # 預熱時間
        warmup_match = re.search(r'預熱完成.*?耗時:\s*(\d+\.\d+)s', output)
        if warmup_match:
            metrics["warmup_time"] = float(warmup_match.group(1))

        # 模型載入時間
        load_match = re.search(r'模型載入完成.*?耗時:\s*(\d+\.\d+)s', output)
        if load_match:
            metrics["model_load_time"] = float(load_match.group(1))

        # 參考音檔信息
        duration_match = re.search(r'秒數.*?Length.*?:\s*(\d+\.\d+)\s*s', output)
        if duration_match:
            metrics["ref_audio_duration"] = float(duration_match.group(1))

        size_match = re.search(r'大小.*?Size.*?:\s*(\d+\.\d+)\s*MB', output)
        if size_match:
            metrics["ref_audio_size_mb"] = float(size_match.group(1))

        # 檢測加速策略
        if "參考音檔加速" in output or "pre_speed_ref" in output:
            metrics["pre_speed_enabled"] = True

        if "播放後製加速" in output or "--speed" in output:
            metrics["post_speed_enabled"] = True

        # 片段計數
        chunk_matches = re.findall(r'\[Queue\] 片段\s*(\d+)', output)
        if chunk_matches:
            metrics["chunk_count"] = max([int(c) for c in chunk_matches])

    except Exception as e:
        metrics["error"] = f"解析錯誤: {str(e)}"

    return metrics


def run_single_test(test_config: Dict[str, Any], test_id: str) -> Dict[str, Any]:
    """
    執行單一測試並收集結果
    """
    print(f"\n{'='*80}")
    print(f"🧪 測試: {test_config['name']}")
    print(f"   {test_config['description']}")
    print(f"{'='*80}")

    result = {
        "test_id": test_id,
        "name": test_config["name"],
        "description": test_config["description"],
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "raw_output": "",
        "error": None,
        "memory_usage_mb": None
    }

    # 記憶體監控 (如果可用)
    initial_memory = None
    if HAS_PSUTIL:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # 執行測試
        cmd = [sys.executable, str(TEST_SCRIPT)] + test_config["args"]

        print(f"📝 執行命令:")
        print(f"   {' '.join(cmd)}\n")

        start_time = time.time()

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 分鐘超時
        )

        execution_time = time.time() - start_time

        # 記錄輸出
        result["raw_output"] = proc.stdout
        result["stderr"] = proc.stderr
        result["execution_time"] = execution_time
        result["return_code"] = proc.returncode

        # 記憶體使用
        if HAS_PSUTIL and initial_memory:
            final_memory = process.memory_info().rss / 1024 / 1024
            result["memory_usage_mb"] = final_memory - initial_memory

        # 解析指標
        if proc.returncode == 0:
            result["metrics"] = parse_test_output(proc.stdout)
            print(f"✅ 測試完成 (耗時: {execution_time:.2f}s)")

            # 顯示關鍵指標
            m = result["metrics"]
            if m.get("ttfb"):
                print(f"   ⚡ TTFB: {m['ttfb']:.2f}s")
            if m.get("total_time"):
                print(f"   ⏱️  總耗時: {m['total_time']:.2f}s")
            if m.get("avg_gen_rate"):
                print(f"   🚀 平均生成倍率: {m['avg_gen_rate']:.2f}x")
            if m.get("overall_rtf"):
                print(f"   📊 整體 RTF: {m['overall_rtf']:.3f}")
        else:
            result["error"] = f"測試失敗 (返回碼: {proc.returncode})"
            print(f"❌ {result['error']}")
            if proc.stderr:
                print(f"   錯誤輸出:\n{proc.stderr}")

    except subprocess.TimeoutExpired:
        result["error"] = "測試超時 (>600s)"
        print(f"❌ {result['error']}")
    except Exception as e:
        result["error"] = f"執行錯誤: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {result['error']}")

    return result




def save_results_csv(results: List[Dict[str, Any]], output_path: Path):
    """
    保存結果為 CSV 格式
    """
    if not results:
        return

    # 提取所有指標鍵
    all_metric_keys = set()
    for r in results:
        if r.get("metrics"):
            all_metric_keys.update(r["metrics"].keys())

    # 構建 CSV 表頭
    fieldnames = [
        "test_id", "name", "description", "timestamp",
        "execution_time", "memory_usage_mb", "error"
    ] + sorted(all_metric_keys)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "test_id": result.get("test_id"),
                "name": result.get("name"),
                "description": result.get("description"),
                "timestamp": result.get("timestamp"),
                "execution_time": result.get("execution_time"),
                "memory_usage_mb": result.get("memory_usage_mb"),
                "error": result.get("error")
            }

            # 添加指標
            if result.get("metrics"):
                for key in all_metric_keys:
                    row[key] = result["metrics"].get(key)

            writer.writerow(row)

    print(f"✅ CSV 已保存: {output_path}")


def save_results_json(results: List[Dict[str, Any]], output_path: Path):
    """
    保存完整結果為 JSON 格式
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON 已保存: {output_path}")


def generate_visualization_1(results: List[Dict[str, Any]], output_path: Path):
    """
    視覺化 1: TTFB vs 總耗時 vs 生成倍率 (柱狀圖)
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  跳過圖表生成 (matplotlib 未安裝)")
        return

    # 提取數據
    names = []
    ttfb_values = []
    total_time_values = []
    gen_rate_values = []

    for r in results:
        if r.get("error"):
            continue

        m = r.get("metrics", {})
        names.append(r["name"])
        ttfb_values.append(m.get("ttfb") or 0)
        total_time_values.append(m.get("total_time") or 0)
        gen_rate_values.append(m.get("avg_gen_rate") or 0)

    if not names:
        print("⚠️  無有效數據，跳過圖表生成")
        return

    # 創建圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('IndexTTS Performance Comparison', fontsize=16, fontweight='bold')

    x_pos = np.arange(len(names))

    # 子圖 1: TTFB (越低越好)
    axes[0].bar(x_pos, ttfb_values, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0].set_title('Time To First Byte (TTFB)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    # 添加數值標籤
    for i, v in enumerate(ttfb_values):
        axes[0].text(i, v + max(ttfb_values)*0.02, f'{v:.2f}s',
                    ha='center', va='bottom', fontsize=9)

    # 子圖 2: 總耗時 (越低越好)
    axes[1].bar(x_pos, total_time_values, color='coral', alpha=0.8)
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].set_title('Total Generation Time', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(total_time_values):
        axes[1].text(i, v + max(total_time_values)*0.02, f'{v:.2f}s',
                    ha='center', va='bottom', fontsize=9)

    # 子圖 3: 生成倍率 (越高越好)
    axes[2].bar(x_pos, gen_rate_values, color='mediumseagreen', alpha=0.8)
    axes[2].set_ylabel('Rate (x)', fontsize=12)
    axes[2].set_title('Average Generation Rate', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Real-time (1.0x)')
    axes[2].legend()

    for i, v in enumerate(gen_rate_values):
        axes[2].text(i, v + max(gen_rate_values)*0.02, f'{v:.2f}x',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 圖表 1 已保存: {output_path}")


def generate_visualization_2(results: List[Dict[str, Any]], output_path: Path):
    """
    視覺化 2: RTF vs 並行效率 vs 記憶體使用 (綜合雷達圖或多軸圖)
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  跳過圖表生成 (matplotlib 未安裝)")
        return

    # 提取數據
    names = []
    rtf_values = []
    parallel_values = []
    memory_values = []

    for r in results:
        if r.get("error"):
            continue

        m = r.get("metrics", {})
        names.append(r["name"])
        rtf_values.append(m.get("overall_rtf") or 0)
        parallel_values.append(m.get("parallel_efficiency") or 0)
        memory_values.append(r.get("memory_usage_mb") or 0)

    if not names:
        print("⚠️  無有效數據，跳過圖表生成")
        return

    # 創建圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('IndexTTS Efficiency & Resource Analysis', fontsize=16, fontweight='bold')

    x_pos = np.arange(len(names))

    # 子圖 1: RTF (越低越好)
    axes[0].bar(x_pos, rtf_values, color='orchid', alpha=0.8)
    axes[0].set_ylabel('RTF (lower is better)', fontsize=12)
    axes[0].set_title('Overall Real-Time Factor', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Real-time threshold')
    axes[0].legend()

    for i, v in enumerate(rtf_values):
        axes[0].text(i, v + max(rtf_values)*0.02 if rtf_values else 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # 子圖 2: 並行效率 (越高越好)
    axes[1].bar(x_pos, parallel_values, color='gold', alpha=0.8)
    axes[1].set_ylabel('Efficiency (%)', fontsize=12)
    axes[1].set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 100)

    for i, v in enumerate(parallel_values):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    # 子圖 3: 記憶體使用
    if any(memory_values):
        axes[2].bar(x_pos, memory_values, color='tomato', alpha=0.8)
        axes[2].set_ylabel('Memory (MB)', fontsize=12)
        axes[2].set_title('Memory Usage', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(names, rotation=45, ha='right')
        axes[2].grid(axis='y', alpha=0.3)

        for i, v in enumerate(memory_values):
            if v > 0:
                axes[2].text(i, v + max(memory_values)*0.02, f'{v:.1f}MB',
                            ha='center', va='bottom', fontsize=9)
    else:
        axes[2].text(0.5, 0.5, 'Memory data not available\n(psutil not installed)',
                    ha='center', va='center', fontsize=12, transform=axes[2].transAxes)
        axes[2].set_xticks([])
        axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 圖表 2 已保存: {output_path}")


def run_test_suite(suite_name: str, test_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    執行測試套件
    """
    print(f"\n{'#'*80}")
    print(f"# 測試套件: {suite_name}")
    print(f"# 測試數量: {len(test_configs)}")
    print(f"{'#'*80}\n")

    results = []

    for idx, config in enumerate(test_configs, 1):
        test_id = f"{suite_name}_{idx:02d}"
        result = run_single_test(config, test_id)
        results.append(result)

        # 延遲以釋放資源
        if idx < len(test_configs):
            print("\n⏸️  等待 5 秒後繼續下一個測試...\n")
            time.sleep(5)

    return results


def generate_summary_report(all_results: Dict[str, List[Dict[str, Any]]], output_path: Path):
    """
    生成綜合摘要報告
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("IndexTTS Streaming Performance Test - Summary Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for suite_name, results in all_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Test Suite: {suite_name}\n")
            f.write(f"{'='*80}\n\n")

            for result in results:
                f.write(f"Test: {result['name']}\n")
                f.write(f"Description: {result['description']}\n")

                if result.get("error"):
                    f.write(f"❌ Status: FAILED\n")
                    f.write(f"   Error: {result['error']}\n")
                else:
                    f.write(f"✅ Status: SUCCESS\n")
                    m = result.get("metrics", {})

                    if m.get("ttfb"):
                        f.write(f"   TTFB: {m['ttfb']:.2f}s\n")
                    if m.get("total_time"):
                        f.write(f"   Total Time: {m['total_time']:.2f}s\n")
                    if m.get("avg_gen_rate"):
                        f.write(f"   Avg Gen Rate: {m['avg_gen_rate']:.2f}x\n")
                    if m.get("overall_rtf"):
                        f.write(f"   Overall RTF: {m['overall_rtf']:.3f}\n")
                    if result.get("memory_usage_mb"):
                        f.write(f"   Memory Usage: {result['memory_usage_mb']:.1f} MB\n")

                f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

    print(f"✅ 摘要報告已保存: {output_path}")


# ==================== 主程式 ====================

def main():
    print("\n" + "=" * 80)
    print("IndexTTS Streaming Performance - Comprehensive Test Suite")
    print("=" * 80 + "\n")

    # 檢查測試腳本
    if not TEST_SCRIPT.exists():
        print(f"❌ 錯誤: 找不到測試腳本 {TEST_SCRIPT}")
        sys.exit(1)

    # 檢查參考音檔
    missing_audio = []
    for audio_file in ["voice_06.wav", "voice_07.wav"]:
        if not (REF_AUDIO_DIR / audio_file).exists():
            missing_audio.append(audio_file)

    if missing_audio:
        print(f"❌ 錯誤: 找不到參考音檔: {', '.join(missing_audio)}")
        print(f"   預期路徑: {REF_AUDIO_DIR}")
        sys.exit(1)

    # 執行所有測試套件
    all_results = {}

    print("\n📋 測試計劃:")
    print("   1️⃣  Voice Comparison (voice_06 vs voice_07)")
    print("   2️⃣  Speed Strategy Comparison (Pre/Post/Hybrid)")
    print("   3️⃣  Version & Mode Comparison (v1/v2, Streaming/Non-streaming)")
    print()

    # Test Suite 1
    print("\n" + "🔹" * 40)
    suite_1_results = run_test_suite("Suite1_Voice_Comparison", TEST_SUITE_1)
    all_results["Suite1_Voice_Comparison"] = suite_1_results

    # Test Suite 2
    print("\n" + "🔹" * 40)
    suite_2_results = run_test_suite("Suite2_Speed_Strategy", TEST_SUITE_2)
    all_results["Suite2_Speed_Strategy"] = suite_2_results

    # Test Suite 3
    print("\n" + "🔹" * 40)
    suite_3_results = run_test_suite("Suite3_Version_Mode", TEST_SUITE_3)
    all_results["Suite3_Version_Mode"] = suite_3_results

    # 生成報告
    print("\n" + "=" * 80)
    print("📊 生成報告中...")
    print("=" * 80 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 合併所有結果
    flat_results = []
    for results in all_results.values():
        flat_results.extend(results)

    # CSV
    csv_path = OUTPUT_DIR / f"benchmark_output_{timestamp}.csv"
    save_results_csv(flat_results, csv_path)

    # JSON
    json_path = OUTPUT_DIR / f"benchmark_output_{timestamp}.json"
    save_results_json(all_results, json_path)

    # 視覺化 1 (所有測試)
    viz1_path = OUTPUT_DIR / f"performance_comparison_{timestamp}.png"
    generate_visualization_1(flat_results, viz1_path)

    # 視覺化 2 (所有測試)
    viz2_path = OUTPUT_DIR / f"efficiency_analysis_{timestamp}.png"
    generate_visualization_2(flat_results, viz2_path)

    # 摘要報告
    summary_path = OUTPUT_DIR / f"summary_report_{timestamp}.txt"
    generate_summary_report(all_results, summary_path)

    # 最終總結
    print("\n" + "=" * 80)
    print("✅ 所有測試完成!")
    print("=" * 80)
    print(f"\n📂 輸出目錄: {OUTPUT_DIR}")
    print(f"\n生成的文件:")
    print(f"   • CSV 數據表格: {csv_path.name}")
    print(f"   • JSON 詳細日誌: {json_path.name}")
    print(f"   • 性能比較圖表: {viz1_path.name}")
    print(f"   • 效率分析圖表: {viz2_path.name}")
    print(f"   • 摘要報告: {summary_path.name}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

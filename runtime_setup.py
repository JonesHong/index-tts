import os
import sys
import site

# ==================== 內部輔助函數 ====================
def _print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def _set_env_var(key, value):
    """設定環境變數並打印 Log"""
    os.environ[key] = str(value)
    print(f"[ENV] Set {key:<25} = {value}")

def _ensure_dir(path, desc):
    """確保目錄存在並打印 Log"""
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
            print(f"[DIR] Created {desc:<21} : {path}")
        except Exception as e:
            print(f"[ERR] Failed to create {desc} : {e}")
    else:
        print(f"[DIR] Exists  {desc:<21} : {path}")

# ==================== 主功能函數 ====================
def initialize(script_path):
    """
    初始化 Index-TTS 執行環境
    :param script_path: 呼叫此函數的腳本路徑 (通常傳入 __file__)
    :return: 包含常用路徑的字典 (ROOT, INDEX_TTS_DIR, HF_HOME...)
    """
    
    # 1. 基礎路徑計算
    _print_section("Path Configuration")

    # 依據你的層級結構：Script -> index-tts -> libs -> root
    # 使用傳入的 script_path 來定位，確保無論在哪呼叫都能找到相對位置
    INDEX_TTS_DIR = os.path.dirname(os.path.abspath(script_path))
    LIBS_DIR = os.path.dirname(INDEX_TTS_DIR)
    ROOT = os.path.dirname(LIBS_DIR)

    # 虛擬環境與 FFMPEG
    VENV_PATH = os.path.join(ROOT, ".venv")
    FFMPEG_BIN = os.path.join(VENV_PATH, "ffmpeg", "bin")

    print(f"ROOT          : {ROOT}")
    print(f"LIBS_DIR      : {LIBS_DIR}")
    print(f"INDEX_TTS_DIR : {INDEX_TTS_DIR}")
    print(f"VENV_PATH     : {VENV_PATH}")
    print(f"FFMPEG_BIN    : {FFMPEG_BIN}")

    # 2. 基礎環境變數與緩存
    _print_section("Environment & Cache Setup")

    # UI 語言
    _set_env_var("UI_LANG", "zh_TW")

    # 緩存路徑定義
    HF_HOME = os.path.join(INDEX_TTS_DIR, "hf_cache")
    HUGGINGFACE_HUB_CACHE = os.path.join(HF_HOME, "hub")
    TRANSFORMERS_CACHE = os.path.join(HF_HOME, "transformers")
    TORCH_EXTENSIONS_DIR = os.path.join(INDEX_TTS_DIR, "_torch_ext")

    # 設定環境變數
    _set_env_var("HF_HOME", HF_HOME)
    _set_env_var("HUGGINGFACE_HUB_CACHE", HUGGINGFACE_HUB_CACHE)
    _set_env_var("TRANSFORMERS_CACHE", TRANSFORMERS_CACHE)
    _set_env_var("TORCH_EXTENSIONS_DIR", TORCH_EXTENSIONS_DIR)

    # 建立實體資料夾
    print("-" * 60)
    _ensure_dir(HF_HOME, "HF_HOME")
    _ensure_dir(HUGGINGFACE_HUB_CACHE, "HUGGINGFACE_HUB")
    _ensure_dir(TRANSFORMERS_CACHE, "TRANSFORMERS")
    _ensure_dir(TORCH_EXTENSIONS_DIR, "TORCH_EXT")

    # 3. DeepSpeed 與 CUDA 優化設定
    _print_section("DeepSpeed & CUDA Config")

    ds_vars = {
        "DEEPSPEED_DISABLE_AIO": "1",
        "DEEPSPEED_ENABLE_AIO": "0",
        "DS_BUILD_AIO": "0",
        "DS_BUILD_OPS": "0",
        "DS_BUILD_FUSED_ADAM": "0",
        "DS_BUILD_SPARSE_ATTN": "0",
        "DS_BUILD_QUANTIZER": "0",
        "DS_BUILD_CUFILE": "0",
        "DEEPSPEED_USE_AIO": "0",
        "DEEPSPEED_USE_CUFILE": "0"
    }

    for k, v in ds_vars.items():
        _set_env_var(k, v)

    print("-" * 60)
    # 限制 Torch CUDA 架構
    _set_env_var("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0;12.0+PTX")

    # 4. Python Path & System Path 注入
    _print_section("Path Injection")

    # —— BigVGAN CUDA 插件路徑 (sys.path) ——
    bigvgan_cuda_path = os.path.join(INDEX_TTS_DIR, "s2mel", "modules", "bigvgan", "alias_free_activation", "cuda", "build")
    if bigvgan_cuda_path not in sys.path:
        sys.path.insert(0, bigvgan_cuda_path)
        print(f"[SYS] Added BigVGAN to sys.path : {bigvgan_cuda_path}")
    else:
        print(f"[SYS] BigVGAN already in sys.path")

    # —— FFMPEG (System PATH) ——
    if os.path.exists(FFMPEG_BIN):
        if FFMPEG_BIN not in os.environ["PATH"]:
            os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]
            print(f"[PATH] Prepend FFMPEG            : {FFMPEG_BIN}")
        else:
            print(f"[PATH] FFMPEG already in PATH")
    else:
        print(f"[WARN] FFMPEG bin not found at   : {FFMPEG_BIN}")

    # —— Torch Lib DLLs (System PATH) ——
    torch_lib_found = False
    try:
        site_packages = site.getsitepackages()
        for sp in site_packages:
            torch_lib = os.path.join(sp, 'torch', 'lib')
            if os.path.exists(torch_lib):
                torch_lib_found = True
                if torch_lib not in os.environ["PATH"]:
                    os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
                    print(f"[PATH] Prepend Torch Lib         : {torch_lib}")
                else:
                    print(f"[PATH] Torch Lib already in PATH : {torch_lib}")
                break
    except Exception as e:
        print(f"[WARN] Error finding site-packages: {e}")

    if not torch_lib_found:
        print(f"[WARN] Could not locate 'torch/lib' in site-packages")

    _print_section("Boot Sequence Complete")
    
    # 回傳這些路徑，方便呼叫者使用 (例如設定 argparse default)
    return {
        "ROOT": ROOT,
        "INDEX_TTS_DIR": INDEX_TTS_DIR,
        "HF_HOME": HF_HOME
    }
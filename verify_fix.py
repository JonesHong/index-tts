import os
import sys

# Mimic startup environment
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

print("Attempting to import webui module...")
try:
    # We don't want to actually run the UI, just check imports
    # But webui.py has code at module level that runs.
    # However, we can just check if we can import it, or emulate the start.
    # Since webui.py has `if __name__ == "__main__":`, importing it is safe-ish
    # BUT it does a lot of heavy lifting at top level (loading models? no, that's in IndexTTS2 init)
    # It does `parser = argparse...` and `cmd_args = parser.parse_args()`
    # We need to mock sys.argv to avoid it parsing our script's args
    sys.argv = ["webui.py", "--help"]
    
    # Actually, just running it with --help is a good test.
    pass
except Exception as e:
    print(f"Setup failed: {e}")

print("Verification script ready.")

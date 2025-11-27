import os
import sys

# Mimic webui.py path setup
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

print("Importing wetext FIRST...")
try:
    import wetext
    print("wetext imported.")
except ImportError as e:
    print(f"Failed to import wetext: {e}")
    import traceback
    traceback.print_exc()

print("Importing kaldifst directly...")
try:
    import kaldifst
    print("kaldifst imported.")
except ImportError as e:
    print(f"Failed to import kaldifst: {e}")

print("Importing pandas...")
import pandas as pd
print("Importing gradio...")
import gradio as gr
print("Importing torch...")
import torch
print("All imports done.")

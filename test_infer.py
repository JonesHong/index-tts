import os
import sys

# 取得目前這個 py 檔案所在的目錄
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 組合出絕對路徑
v2_model_dir = os.path.join(ROOT_DIR, "checkpoints")
v2_config_path = os.path.join(v2_model_dir, "config.yaml")

# 檢查一下路徑是否存在
if not os.path.exists(v2_model_dir):
    print(f"❌ 錯誤：找不到模型資料夾：{v2_model_dir}")
    sys.exit(1)

print(f"✅ 模型路徑鎖定：{v2_model_dir}")

from indextts.infer_v2 import IndexTTS2




tts = IndexTTS2(cfg_path=v2_config_path, model_dir=v2_model_dir, use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "Translate for me, what is a surprise!"
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="output/gen_01.wav", verbose=True)

text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="output/gen_02.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)

text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="output/gen_03.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)

text = "哇塞！这个爆率也太高了！欧皇附体了！"
tts.infer(spk_audio_prompt='examples/voice_10.wav', text=text, output_path="output/gen_04.wav", emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0], use_random=False, verbose=True)

text = "快躲起来！是他要来了！他要来抓我们了！"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="output/gen_05.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)

text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="output/gen_06.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
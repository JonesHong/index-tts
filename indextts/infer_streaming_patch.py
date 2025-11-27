"""
IndexTTS 1.5 Streaming Patch
為 IndexTTS 1.5 添加 streaming 功能
"""

import time
import torch
import torchaudio
from indextts.utils.feature_extractors import MelSpectrogramFeatures


def infer_stream(self, audio_prompt, text, verbose=False, max_text_tokens_per_segment=120,
                 **generation_kwargs):
    """
    IndexTTS 1.5 的 streaming 版本
    每生成一個片段就立即 yield，而不是等待全部完成

    Args:
        self: IndexTTS 實例
        audio_prompt: 參考音頻路徑
        text: 要合成的文字
        verbose: 是否顯示詳細信息
        max_text_tokens_per_segment: 每個片段的最大 token 數
        **generation_kwargs: 額外的生成參數

    Yields:
        torch.Tensor: 音頻片段 (int16, shape: [1, samples])
    """
    print(">> starting streaming inference...")
    if verbose:
        print(f"origin text:{text}")
    start_time = time.perf_counter()

    # 如果参考音频改变了，才需要重新生成 cond_mel
    if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
        audio, sr = torchaudio.load(audio_prompt)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
        cond_mel_frame = cond_mel.shape[-1]
        if verbose:
            print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

        self.cache_audio_prompt = audio_prompt
        self.cache_cond_mel = cond_mel
    else:
        cond_mel = self.cache_cond_mel
        cond_mel_frame = cond_mel.shape[-1]

    auto_conditioning = cond_mel
    text_tokens_list = self.tokenizer.tokenize(text)
    segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment)

    if verbose:
        print("text token count:", len(text_tokens_list))
        print("segments count:", len(segments))
        print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
        print(*segments, sep="\n")

    # 生成參數
    do_sample = generation_kwargs.pop("do_sample", True)
    top_p = generation_kwargs.pop("top_p", 0.8)
    top_k = generation_kwargs.pop("top_k", 30)
    temperature = generation_kwargs.pop("temperature", 1.0)
    autoregressive_batch_size = 1
    length_penalty = generation_kwargs.pop("length_penalty", 0.0)
    num_beams = generation_kwargs.pop("num_beams", 3)
    repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
    max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
    sampling_rate = 24000

    gpt_gen_time = 0
    gpt_forward_time = 0
    bigvgan_time = 0
    has_warned = False

    # 🔥 關鍵修改：逐片段生成並立即 yield
    for idx, sent in enumerate(segments, 1):
        text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
        text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

        if verbose:
            print(f"\n>>> Processing segment {idx}/{len(segments)}")
            print(f"text_tokens shape: {text_tokens.shape}")

        m_start_time = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                # Step 1: GPT 生成 codes
                codes = self.gpt.inference_speech(
                    auto_conditioning, text_tokens,
                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=autoregressive_batch_size,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_generate_length=max_mel_tokens,
                    **generation_kwargs
                )
                gpt_gen_time += time.perf_counter() - m_start_time

                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    import warnings
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) "
                        f"or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)

                # Remove ultra-long silence
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)

                # Step 2: GPT forward 生成 latent
                m_start_time = time.perf_counter()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        auto_conditioning, text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        codes,
                        code_lens * self.gpt.mel_length_compression,
                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                        return_latent=True,
                        clip_inputs=False
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                    # Step 3: BigVGAN 生成音頻
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                # 正規化到 int16 範圍
                wav = torch.clamp(32767 * wav, -32767.0, 32767.0).type(torch.int16)

                if verbose:
                    duration = wav.shape[-1] / sampling_rate
                    print(f"✅ Generated segment {idx}: {duration:.2f}s, shape: {wav.shape}")

                # 🔥 立即 yield 這個片段
                yield wav.cpu()

    end_time = time.perf_counter()

    # 打印統計信息
    print(f"\n>> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
    print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
    print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
    print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
    print(f">> Total streaming time: {end_time - start_time:.2f} seconds")


def add_streaming_to_indextts(tts_instance):
    """
    動態添加 streaming 方法到 IndexTTS 實例

    Args:
        tts_instance: IndexTTS 實例

    Returns:
        修改後的 IndexTTS 實例（添加了 infer_stream 方法）
    """
    import types
    tts_instance.infer_stream = types.MethodType(infer_stream, tts_instance)
    return tts_instance

"""运行实验 3-3：SpeechT5 TTS 推理。"""
import os

import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
import soundfile as sf


def main():
    os.makedirs("outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = "语音信息处理实验三，基于深度学习的语音合成推理演示。"
    print(f"设备: {device}")
    print(f"合成文本: {text}")

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    inputs = processor(text=text, return_tensors="pt").to(device)
    # 使用固定维度的默认说话人嵌入，避免依赖旧版 datasets 脚本
    speaker_embedding = torch.zeros((1, 512), device=device)

    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

    out_path = "outputs/tts_speecht5.wav"
    sf.write(out_path, speech.cpu().numpy(), samplerate=16000)
    print(f"完成 3-3: {out_path}")


if __name__ == "__main__":
    main()

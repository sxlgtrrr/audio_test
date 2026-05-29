"""运行实验 3-3：SpeechT5 英文 TTS 推理。"""
import argparse
import os

# 必须在 import transformers / huggingface_hub 之前设置，否则会直连 huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

ROOT = Path(__file__).resolve().parent
SPEAKER_EMB_PATH = ROOT / "assets" / "speecht5_speaker_embedding.npy"
SPEAKER_INDEX = 7306
DEFAULT_TEXT = (
    "This is experiment three of speech information processing, "
    "demonstrating deep learning based text to speech synthesis."
)

def _from_pretrained(cls, model_id: str, device: str):
    try:
        obj = cls.from_pretrained(model_id, local_files_only=True)
    except OSError:
        print(f"本地无缓存，从镜像下载: {model_id}")
        obj = cls.from_pretrained(model_id)
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def load_speaker_embedding(device: str) -> torch.Tensor:
    if not SPEAKER_EMB_PATH.exists():
        SPEAKER_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
        zip_path = hf_hub_download(
            "Matthijs/cmu-arctic-xvectors",
            "spkrec-xvect.zip",
            repo_type="dataset",
        )
        with zipfile.ZipFile(zip_path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".npy")]
            xvector = np.load(archive.open(names[SPEAKER_INDEX]))
        np.save(SPEAKER_EMB_PATH, xvector)

    xvector = np.load(SPEAKER_EMB_PATH)
    return torch.tensor(xvector, device=device, dtype=torch.float32).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="SpeechT5 英文 TTS 推理")
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="要合成的英文文本（SpeechT5 不支持中文）",
    )
    parser.add_argument(
        "--output",
        default="outputs/tts_speecht5.wav",
        help="输出 wav 路径",
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = args.text
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    print(f"设备: {device}")
    print(f"合成文本: {text}")

    processor = _from_pretrained(SpeechT5Processor, "microsoft/speecht5_tts", device)
    model = _from_pretrained(SpeechT5ForTextToSpeech, "microsoft/speecht5_tts", device)
    vocoder = _from_pretrained(SpeechT5HifiGan, "microsoft/speecht5_hifigan", device)

    inputs = processor(text=text, return_tensors="pt").to(device)
    speaker_embedding = load_speaker_embedding(device)

    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

    out_path = args.output
    sf.write(out_path, speech.cpu().numpy(), samplerate=16000)
    duration = len(speech) / 16000
    print(f"完成 3-3: {out_path} ({duration:.2f}s, {len(speech)} samples)")

if __name__ == "__main__":
    main()

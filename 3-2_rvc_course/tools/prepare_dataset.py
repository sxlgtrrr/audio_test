"""准备 RVC 微调数据：将 demo.wav 切分为多段训练样本。"""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "course_voice"
SOURCE = ROOT.parent / "demo.wav"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        y, sr = librosa.load(librosa.example("trumpet"), sr=40000, mono=True)
        sf.write(SOURCE, y, sr)
        print(f"已生成源音频: {SOURCE}")

    audio, sr = librosa.load(SOURCE, sr=40000, mono=True)
    chunk_size = sr * 4
    hop = sr * 2
    count = 0
    for start in range(0, max(len(audio) - chunk_size, 1), hop):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < sr:
            continue
        out = DATA_DIR / f"sample_{count:03d}.wav"
        sf.write(out, chunk, sr)
        count += 1

    # 若源音频较短，通过循环与轻微扰动扩充样本数
    while count < 12:
        shift = (count * 800) % max(len(audio) // 4, 1)
        chunk = np.roll(audio, shift)[:chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunk = chunk * (0.95 + 0.01 * (count % 5))
        out = DATA_DIR / f"sample_{count:03d}.wav"
        sf.write(out, chunk.astype(np.float32), sr)
        count += 1

    if count == 0:
        out = DATA_DIR / "sample_000.wav"
        sf.write(out, audio, sr)
        count = 1

    print(f"已准备 {count} 条训练样本: {DATA_DIR}")


if __name__ == "__main__":
    main()

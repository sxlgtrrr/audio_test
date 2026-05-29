"""本地生成 RVC 训练所需的 mute 样本。"""
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MUTE_ROOT = ROOT / "logs" / "mute"


def write_silent_wav(path: Path, sr: int, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.zeros(int(sr * seconds), dtype=np.float32)
    sf.write(path, wav, sr)


def write_feature(path: Path, dim: int, frames: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((frames, dim), dtype=np.float32))


def write_f0(path: Path, frames: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros(frames, dtype=np.float32))


def main() -> None:
    for sr_name, sr in [("32k", 32000), ("40k", 40000), ("48k", 48000)]:
        write_silent_wav(MUTE_ROOT / "0_gt_wavs" / f"mute{sr_name}.wav", sr)

    write_feature(MUTE_ROOT / "3_feature256" / "mute.npy", 256)
    write_feature(MUTE_ROOT / "3_feature768" / "mute.npy", 768)
    write_f0(MUTE_ROOT / "2a_f0" / "mute.wav.npy")
    write_f0(MUTE_ROOT / "2b-f0nsf" / "mute.wav.npy")
    print(f"mute samples created under {MUTE_ROOT}")


if __name__ == "__main__":
    main()

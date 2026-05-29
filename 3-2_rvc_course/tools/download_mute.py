"""下载 RVC 训练所需的 mute 样本（通过镜像）。"""
from pathlib import Path
from urllib.request import urlopen
import os

BASE = os.environ.get(
    "RVC_WEIGHT_BASE_URL",
    "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main",
)
ROOT = Path(__file__).resolve().parents[1]
MUTE_ROOT = ROOT / "logs" / "mute"

FILES = [
    "logs/mute/0_gt_wavs/mute32000.wav",
    "logs/mute/0_gt_wavs/mute40000.wav",
    "logs/mute/0_gt_wavs/mute48000.wav",
    "logs/mute/3_feature256/mute.npy",
    "logs/mute/3_feature768/mute.npy",
    "logs/mute/2a_f0/mute.wav.npy",
    "logs/mute/2b-f0nsf/mute.wav.npy",
]


def download(relative: str, target: Path) -> None:
    if target.exists():
        print(f"skip {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{relative}"
    print(f"download {url}")
    with urlopen(url, timeout=120) as response, target.open("wb") as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
    print(f"saved {target}")


def main():
    for relative in FILES:
        rel_path = Path(relative)
        target = ROOT / rel_path
        download(relative, target)


if __name__ == "__main__":
    main()

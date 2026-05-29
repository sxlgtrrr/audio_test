from pathlib import Path
from urllib.request import urlopen
import os


BASE_URL = os.environ.get(
    "RVC_WEIGHT_BASE_URL",
    "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main",
)
ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = [
    ("hubert_base.pt", ROOT / "assets/hubert/hubert_base.pt"),
    ("pretrained_v2/G40k.pth", ROOT / "assets/pretrained_v2/G40k.pth"),
    ("pretrained_v2/D40k.pth", ROOT / "assets/pretrained_v2/D40k.pth"),
    ("pretrained_v2/f0G40k.pth", ROOT / "assets/pretrained_v2/f0G40k.pth"),
    ("pretrained_v2/f0D40k.pth", ROOT / "assets/pretrained_v2/f0D40k.pth"),
]


def download(relative_url: str, target: Path) -> None:
    if target.exists():
        print(f"skip {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{relative_url}"
    print(f"download {url}")

    with urlopen(url) as response, target.open("wb") as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)

    print(f"saved {target}")


def main() -> None:
    for relative_url, target in WEIGHTS:
        download(relative_url, target)


if __name__ == "__main__":
    main()

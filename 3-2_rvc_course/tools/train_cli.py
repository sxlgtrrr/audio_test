import argparse
import os
import shutil
import subprocess
import sys
from multiprocessing import cpu_count
from pathlib import Path
from random import shuffle


ROOT = Path(__file__).resolve().parents[1]
SR_DICT = {"32k": 32000, "40k": 40000, "48k": 48000}


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode not in (0, 2333333):
        raise subprocess.CalledProcessError(result.returncode, cmd)


def detect_device() -> tuple[str, bool]:
    import torch

    if torch.cuda.is_available():
        return "cuda:0", True
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def preprocess(dataset: Path, exp_dir: Path, sr: str, n_p: int, preprocess_per: float) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "preprocess.log").write_text("", encoding="utf-8")
    run(
        [
            sys.executable,
            "infer/modules/train/preprocess.py",
            str(dataset),
            str(SR_DICT[sr]),
            str(n_p),
            str(exp_dir),
            "False",
            f"{preprocess_per:.1f}",
        ]
    )


def extract_f0(exp_dir: Path, n_p: int, f0_method: str) -> None:
    run(
        [
            sys.executable,
            "infer/modules/train/extract/extract_f0_print.py",
            str(exp_dir),
            str(n_p),
            f0_method,
        ]
    )


def extract_features(exp_dir: Path, version: str, gpus: str) -> None:
    import torch

    device, is_half = detect_device()
    gpu_ids = [item for item in gpus.split("-") if item.strip()]
    if torch.cuda.is_available() and gpu_ids:
        for part, gpu_id in enumerate(gpu_ids):
            run(
                [
                    sys.executable,
                    "infer/modules/train/extract_feature_print.py",
                    device,
                    str(len(gpu_ids)),
                    str(part),
                    gpu_id,
                    str(exp_dir),
                    version,
                    str(is_half),
                ]
            )
    else:
        run(
            [
                sys.executable,
                "infer/modules/train/extract_feature_print.py",
                device,
                "1",
                "0",
                str(exp_dir),
                version,
                str(is_half),
            ]
        )


def config_key(sr: str, version: str) -> str:
    if version == "v1" or sr == "40k":
        return f"v1/{sr}.json"
    return f"v2/{sr}.json"


def write_config(exp_dir: Path, sr: str, version: str) -> None:
    src = ROOT / "configs" / config_key(sr, version)
    dst = exp_dir / "config.json"
    if not dst.exists():
        shutil.copy2(src, dst)


def _make_row(
    name: str,
    gt_wavs_dir: Path,
    feature_dir: Path,
    exp_dir: Path,
    if_f0: int,
    speaker_id: int,
) -> str:
    if if_f0:
        return (
            f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|"
            f"{exp_dir}/2a_f0/{name}.wav.npy|{exp_dir}/2b-f0nsf/{name}.wav.npy|{speaker_id}"
        )
    return f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{speaker_id}"


def write_filelist(exp_dir: Path, sr: str, version: str, if_f0: int, speaker_id: int) -> None:
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")

    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names = (
            {path.stem for path in gt_wavs_dir.glob("*.wav")}
            & {path.stem for path in feature_dir.glob("*.npy")}
            & {path.name.replace(".wav.npy", "") for path in f0_dir.glob("*.wav.npy")}
            & {path.name.replace(".wav.npy", "") for path in f0nsf_dir.glob("*.wav.npy")}
        )
    else:
        names = {path.stem for path in gt_wavs_dir.glob("*.wav")} & {
            path.stem for path in feature_dir.glob("*.npy")
        }

    sorted_names = sorted(names)
    n_total = len(sorted_names)
    n_val = max(1, int(n_total * 0.1))
    n_test = max(1, int(n_total * 0.1))
    if n_total <= 2:
        train_names = sorted_names
        val_names = sorted_names[:1]
        test_names = sorted_names[:1]
    else:
        val_names = sorted_names[:n_val]
        test_names = sorted_names[n_val : n_val + n_test]
        train_names = sorted_names[n_val + n_test :]

    train_rows = [
        _make_row(name, gt_wavs_dir, feature_dir, exp_dir, if_f0, speaker_id)
        for name in train_names
    ]
    val_rows = [
        _make_row(name, gt_wavs_dir, feature_dir, exp_dir, if_f0, speaker_id)
        for name in val_names
    ]
    test_rows = [
        _make_row(name, gt_wavs_dir, feature_dir, exp_dir, if_f0, speaker_id)
        for name in test_names
    ]

    fea_dim = 256 if version == "v1" else 768
    mute_root = ROOT / "logs" / "mute"
    for _ in range(2):
        if if_f0:
            train_rows.append(
                f"{mute_root}/0_gt_wavs/mute{sr}.wav|{mute_root}/3_feature{fea_dim}/mute.npy|"
                f"{mute_root}/2a_f0/mute.wav.npy|{mute_root}/2b-f0nsf/mute.wav.npy|{speaker_id}"
            )
        else:
            train_rows.append(
                f"{mute_root}/0_gt_wavs/mute{sr}.wav|{mute_root}/3_feature{fea_dim}/mute.npy|{speaker_id}"
            )

    shuffle(train_rows)
    (exp_dir / "filelist.txt").write_text("\n".join(train_rows), encoding="utf-8")
    (exp_dir / "filelist_val.txt").write_text("\n".join(val_rows), encoding="utf-8")
    (exp_dir / "filelist_test.txt").write_text("\n".join(test_rows), encoding="utf-8")
    print(
        f"filelist.txt rows: {len(train_rows)}, "
        f"filelist_val.txt rows: {len(val_rows)}, "
        f"filelist_test.txt rows: {len(test_rows)}"
    )


def train_model(args: argparse.Namespace, exp_dir: Path) -> None:
    write_config(exp_dir, args.sr, args.version)
    write_filelist(exp_dir, args.sr, args.version, args.if_f0, args.speaker_id)

    cmd = [
        sys.executable,
        "infer/modules/train/train.py",
        "-e",
        args.exp_name,
        "-sr",
        args.sr,
        "-f0",
        str(args.if_f0),
        "-bs",
        str(args.batch_size),
        "-te",
        str(args.epochs),
        "-se",
        str(args.save_every_epoch),
        "-l",
        str(args.save_latest),
        "-c",
        str(args.cache_gpu),
        "-sw",
        str(args.save_every_weights),
        "-v",
        args.version,
    ]
    if args.gpus:
        cmd.extend(["-g", args.gpus])
    if args.pretrained_g:
        cmd.extend(["-pg", args.pretrained_g])
    if args.pretrained_d:
        cmd.extend(["-pd", args.pretrained_d])
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RVC training pipeline for coursework.")
    parser.add_argument("--dataset", required=True, help="Directory containing raw speaker audio files")
    parser.add_argument("--exp-name", default="course_rvc", help="Experiment name under logs/")
    parser.add_argument("--sr", choices=sorted(SR_DICT), default="40k")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--if-f0", type=int, choices=[0, 1], default=1)
    parser.add_argument("--f0-method", choices=["pm", "harvest", "dio", "rmvpe"], default="harvest")
    parser.add_argument("--gpus", default="0", help="GPU ids split by '-', or empty string for CPU/MPS")
    parser.add_argument("--n-p", type=int, default=max(1, cpu_count() // 2))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save-every-epoch", type=int, default=1)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--pretrained-g", default="")
    parser.add_argument("--pretrained-d", default="")
    parser.add_argument("--save-latest", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cache-gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--save-every-weights", type=int, choices=[0, 1], default=1)
    parser.add_argument("--preprocess-per", type=float, default=3.7)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-feature", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    dataset = Path(args.dataset).expanduser().resolve()
    exp_dir = ROOT / "logs" / args.exp_name
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}")

    if not args.skip_preprocess:
        preprocess(dataset, exp_dir, args.sr, args.n_p, args.preprocess_per)
    if not args.skip_feature:
        if args.if_f0:
            extract_f0(exp_dir, args.n_p, args.f0_method)
        extract_features(exp_dir, args.version, args.gpus)
    if not args.skip_train:
        train_model(args, exp_dir)
    if not args.skip_index:
        run([sys.executable, "tools/build_index.py", "-e", args.exp_name, "-v", args.version])


if __name__ == "__main__":
    main()

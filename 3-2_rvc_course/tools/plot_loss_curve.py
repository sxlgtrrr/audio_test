"""从 train.log 绘制 train/eval loss 对比图。"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_train_loss(log_text: str):
    epochs = []
    mel_losses = []
    current_epoch = None
    for line in log_text.splitlines():
        epoch_match = re.search(r"Train Epoch: (\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        mel_match = re.search(r"loss_mel=([0-9.]+)", line)
        if mel_match and current_epoch is not None:
            epochs.append(current_epoch)
            mel_losses.append(float(mel_match.group(1)))
    return epochs, mel_losses


def parse_eval_loss(log_text: str):
    epochs = []
    mel_losses = []
    for line in log_text.splitlines():
        match = re.search(r"Eval Epoch: (\d+), val/loss_mel=([0-9.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            mel_losses.append(float(match.group(2)))
    return epochs, mel_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, help="logs/<exp-name> 目录")
    parser.add_argument("--output", default="outputs/rvc_loss_curve.png")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    log_path = exp_dir / "train.log"
    if not log_path.exists():
        raise FileNotFoundError(f"未找到 {log_path}")

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    train_epochs, train_mel = parse_train_loss(log_text)
    eval_epochs, eval_mel = parse_eval_loss(log_text)

    plt.figure(figsize=(10, 5))
    if train_epochs:
        plt.plot(train_epochs, train_mel, label="train loss_mel", alpha=0.7)
    if eval_epochs:
        plt.plot(eval_epochs, eval_mel, marker="o", label="eval val/loss_mel")
    plt.xlabel("Epoch")
    plt.ylabel("Mel L1 Loss")
    plt.title("RVC Train / Eval Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()

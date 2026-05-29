"""运行实验 3-1 Griffin-Lim，生成波形图与音频文件。"""
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

n_fft, hop_length, win_length = 720, 160, 720


def _stft(y):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _griffin_lim(S, gl_iters):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex128)
    y = _istft(S_complex * angles)
    for _ in range(gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def main():
    os.makedirs("outputs", exist_ok=True)
    input_path = "C7_2_y.wav"
    if not os.path.exists(input_path):
        input_path = "demo.wav"
    if not os.path.exists(input_path):
        y, sr = librosa.load(librosa.example("trumpet"), sr=16000, mono=True)
        sf.write("demo.wav", y, sr)
        input_path = "demo.wav"
        print("未找到课程音频，已用 librosa 示例生成 demo.wav")

    audio, sr = librosa.load(input_path, sr=16000, mono=True)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    gl_iters = 32
    re_audio = _griffin_lim(D, gl_iters)

    sf.write("outputs/original.wav", audio, sr)
    sf.write("outputs/reconstructed.wav", re_audio, sr)

    plt.rcParams["figure.figsize"] = (10.0, 8.0)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(audio)
    plt.xlabel("t/s")
    plt.title("original signal")
    plt.subplot(2, 1, 2)
    plt.plot(re_audio)
    plt.xlabel("t/s")
    plt.title("reconstructed signal")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("outputs/griffin_lim_waveform.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("完成 3-1: outputs/original.wav, outputs/reconstructed.wav, outputs/griffin_lim_waveform.png")


if __name__ == "__main__":
    main()

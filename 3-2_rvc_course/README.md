# RVC Course

本项目基于 [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)，保留语音转换模型、训练、特征检索索引和命令行推理等关键步骤，以供学习与实践。

如果算力有限，推荐使用学校算力平台或 [Colab](https://colab.research.google.com/) 完成本项目。

## 目录结构与文件说明

### 目录结构

```text
rvc-minimal-course/
├── assets/
│   ├── hubert/          # 存放 HuBERT 内容特征提取模型 hubert_base.pt
│   ├── pretrained/      # 存放 RVC v1 训练初始化用的 G/D 预训练权重。
│   ├── pretrained_v2/   # 存放 RVC v2 训练初始化用的 G/D 预训练权重。
│   ├── rmvpe/           # 存放 RMVPE F0 提取模型；本实验 if-f0=0 时不是必需
│   ├── indices/         # 存放训练后建立的 FAISS 检索索引
│   └── weights/         # 存放训练完成后导出的说话人模型
├── configs/             # RVC 原配置
├── infer/               # RVC 核心代码目录，虽然名字叫 infer，但这里同时包含训练和推理代码
│   ├── lib/infer_pack/  # 定义生成器、判别器等模型结构
│   ├── lib/train/       # 定义数据集读取、loss、checkpoint 保存加载等训练工具
│   └── modules/
│       ├── train/       # 包含预处理、F0 提取、HuBERT 特征提取和训练主流程
│       └── vc/          # 包含声音转换推理流程
├── logs/                # logs/<实验名>/ 会保存预处理结果、特征、checkpoint、TensorBoard 日志和索引
│   └── mute/            # RVC 训练 filelist 需要的静音样本
└── tools/
    ├── train_cli.py     # 训练启动脚本，串起预处理、特征提取、训练、建索引等过程
    ├── build_index.py   # 单独建立 FAISS 索引
    ├── infer_cli.py     # 用已有说话人模型进行推理
    └── download.py      # 下载最小实验所需权重
```

### 重点文件
其中，下列文件与本次实验核心工作相关性较高，需重点阅读

```text
tools/train_cli.py
训练实验的入口，串联数据预处理、特征提取、训练、建索引等流程，可以一键运行，也可以从中拆分出多个单独运行的步骤

infer/modules/train/train.py
训练主循环，包括生成器、判别器、loss 计算、日志记录、checkpoint 保存等步骤
```

下列文件与实验核心任务相关性较低，设计数据处理，模型结构，pipeline等，可作为进阶阅读

```text
infer/lib/train/data_utils.py
训练数据如何从 filelist.txt 组织成 batch，音频、HubERT 特征、F0 特征如何被读取和整理

infer/modules/vc/modules.py
说话人模型、HuBERT、索引等核心模型

infer/modules/vc/pipeline.py
声音转换pipeline，包括内容特征、F0、索引特征混合、生成器推理等
```

### 附件

附件包含了一个预训练说话人权重(rvc_voice.pt)和一个单说话人训练集(AISLEE3-SSB0011)，可以通过网盘下载
- 链接: https://yun.139.com/shareweb/#/w/i/2v3EiR3FnMgls  提取码:0f7r

也可以直接访问rvc官网与aishell官网下载
- https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/weights
- https://www.aishelltech.com/aishell_3

## 环境

建议使用 Python 3.10。推荐使用 `conda` 或在项目根目录中创建独立虚拟环境，避免和系统 Python 或其他课程项目的依赖冲突。

### 基于 pip 构建

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
或

```bash
conda create -n rvc python=3.10
conda activate rvc
pip install -r requirements.txt
```

### 基于 uv 构建

如果已经安装 `uv`，可以直接根据 `pyproject.toml` 创建环境并安装依赖：

```bash
uv sync
```

之后运行脚本时使用：

```bash
uv run tools/train_cli.py
uv run tools/infer_cli.py
```

### 其他依赖

系统还需要可执行的 `ffmpeg`。Linux/macOS 可以通过包管理器安装，Windows 可以下载 FFmpeg 后将其加入 `PATH`。安装完成后可用以下命令检查：

```bash
ffmpeg -version
```

首次训练前还需要下载 HuBERT 和 RVC 预训练权重：

```bash
python tools/download.py
```

该脚本会下载：

```text
assets/hubert/hubert_base.pt
assets/pretrained_v2/G40k.pth
assets/pretrained_v2/D40k.pth
```

## 训练

准备一个只包含目标说话人音频的目录，例如 `data/my_voice/`。一条命令可以完成数据预处理、F0 提取、Hubert 特征提取、训练和索引建立：

```bash
python tools/train_cli.py \
  --dataset data/my_voice \
  --exp-name demo_voice \
  --sr 40k \
  --version v2 \
  --if-f0 0 \
  --f0-method harvest \
  --gpus 0 \
  --epochs 20 \
  --batch-size 4
```

如果要使用预训练权重：

```bash
python tools/train_cli.py \
  --dataset data/my_voice \
  --exp-name demo_voice \
  --sr 48k \
  --version v2 \
  --pretrained-g assets/pretrained_v2/f0G48k.pth \
  --pretrained-d assets/pretrained_v2/f0D48k.pth
```

训练过程会生成：

```text
logs/<exp-name>/0_gt_wavs/       # 切分并重采样后的训练音频
logs/<exp-name>/1_16k_wavs/      # Hubert 输入音频
logs/<exp-name>/2a_f0/           # coarse F0
logs/<exp-name>/2b-f0nsf/        # continuous F0
logs/<exp-name>/3_feature768/    # v2 Hubert 特征，v1 为 3_feature256
logs/<exp-name>/G_*.pth,D_*.pth  # checkpoint
logs/<exp-name>/added_*.index    # FAISS 检索索引
assets/weights/<exp-name>.pth    # 导出的推理模型
```

## 推理

模型文件默认从 `.env` 中的 `weight_root=assets/weights` 查找：

```bash
python tools/infer_cli.py \
  --model_name demo_voice.pth \
  --input_path input.wav \
  --opt_path output.wav \
  --f0method harvest
```

## 任务

1. 依照上述说明完成环境部署，和基本的推理与微调，微调数据可用提供的数据集，也可自行搜集数据或自己录制，时长不少于10分钟；
2. 基于目前的训练代码，补充验证集划分代码
  - `tools/train_cli.py:107`
    - `write_filelist(...)`
    - 将样本划分为训练/验证/测试集，除 `filelist.txt` 外额外生成 `filelist_val.txt` 和 `filelist_test.txt`。

  - `infer/modules/train/train.py:299`
    - `build_eval_loader(hps)`
    - 从 `logs/<exp-name>/filelist_val.txt` 构造验证集 `DataLoader`。

  - `infer/modules/train/train.py:306`
    - `evaluate(epoch, hps, net_g, eval_loader, logger, writer_eval, rank)`
    - 在 `torch.no_grad()` 下运行验证集，计算并记录 `val/loss_mel`。
3. 训练并观察 train loss 与 eval loss 的差异，解释现象
4. 自行探索其他优化（可选）
# 语音信息处理实验三

实验内容：

- **3-1**：Griffin-Lim 声码器（`experiment_3.ipynb`）
- **3-2**：RVC 声音转换微调（`3-2_rvc_course/`，含验证集划分与 eval 代码补全）
- **3-3（选做）**：SpeechT5 TTS 推理（`run_exp3_3.py`）

## 目录

```text
experiment_3.ipynb      # 主实验 notebook
run_exp3_1.py           # 3-1 脚本
run_exp3_3.py           # 3-3 脚本
outputs/                # 实验输出（波形图、音频、loss 曲线）
3-2_rvc_course/         # RVC 课程项目（详见其 README.md）
```

## 运行说明

1. 打开 `experiment_3.ipynb` 按单元格顺序运行
2. RVC 训练与推理见 `3-2_rvc_course/README.md`
3. 大文件（预训练权重、AISHELL 数据集、模型 checkpoint）需自行下载，未纳入仓库

## 实验结果

`outputs/` 目录包含：

- Griffin-Lim 重建波形与对比图
- RVC Demo / AISHELL 微调 loss 曲线
- 声音转换与 TTS 推理音频

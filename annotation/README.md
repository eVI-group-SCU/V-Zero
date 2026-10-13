# 标注工具运行指南

以下命令均在仓库根目录运行。输入图片路径必须在运行机器上可访问。

## 环境与 API 配置

激活已安装项目依赖的 Python 环境，并安装标注所需依赖：

```bash
python -m pip install Pillow pyarrow openai tqdm
```

首次配置时复制模板；已有 `.seed.env` 时直接编辑该文件：

```bash
cp .seed.env.example .seed.env
```

在 `.seed.env` 中填写 `ARK_BASE_URL`、`ARK_MODEL` 和 `ARK_API_KEY`。脚本也支持其他兼容 OpenAI Chat Completions 的视觉模型服务。若密钥存放在其他环境变量中，设置 `ARK_API_KEY_ENV` 为该变量名。密钥文件不要提交到 Git。

命令行可通过 `--base-url`、`--model`、`--api-key-env` 覆盖对应配置。`--api-key-env` 接受环境变量名称，不接受密钥本身。

Random / nearby 启动脚本读取 `.project.env`、`.seed.env`，使用当前环境中的 Python；可以用 `PYTHON_BIN` 指定解释器。正 crop 的 `label.sh` 还会读取 `.ssh_server.env` 并激活 Conda 环境：使用 `LABEL_CONDA_ENV` 指定环境名，默认回退到 `SEED_CONDA_ENV`、`FILTER_CONDA_ENV` 或 `verl`；需要时在 `.project.env` 设置 `REMOTE_CONDA_INIT` 来初始化 Conda。

## Positive bbox / point

```bash
LABEL_CONDA_ENV=your-env bash annotation/label.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/positive-preview \
  --limit 10 --workers 4
```

输入需要原图及问题。先检查预览标注，再去掉 `--limit` 并使用新的输出目录进行全量标注。使用 `--help` 查看完整参数。

## Random / nearby negative crop

输入为一个 Parquet 文件，需要包含原图、问题和 `teacher_images` 中的一个 positive crop。负 crop 的宽高以 positive 图片的实际像素尺寸为准。

```bash
bash annotation/label_random_negative.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/random-preview \
  --limit 10 --workers 4

bash annotation/label_nearby_negative.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/nearby-preview \
  --limit 10 --workers 4
```

`--workers` 控制每个进程的 API 并发数。两个入口同时运行时，并发数相加。全量运行示例：

```bash
bash annotation/label_nearby_negative.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/nearby-full \
  --workers 64
```

Random 表示由模型选择无关区域，不是均匀随机采样。Nearby 要求模型选择 positive 周边区域。两者均向标注模型提供原图、positive crop 和问题，并要求输出同尺寸负框及框内 point。

默认保存裁剪图片，并在全部选中行成功后导出 `train.parquet`，仅替换或新增 `teacher_neg_images`。使用 `--limit` 时，导出的文件只包含选中行，不能当作全量数据。

断点续跑默认开启：使用完全相同的命令和输出目录，跳过成功且裁图存在的记录，重试失败记录。改变输入、参数或从预览切换到全量时使用新目录。存在失败行时不会导出完整训练文件，应查看 `grounding.jsonl` 的 `error` 和 `summary.json`。

只保存标注记录可加 `--no-write-train-parquet`；若还要保存裁图，再加 `--save-crops`。坐标约定、几何校验和输出字段见 [负 crop 说明](NEGATIVE_CROPS.md)。

## Black negative crop

此入口无需 API。输入必须包含 `images`、`teacher_images`、`teacher_neg_images`，且图片使用可访问的本地路径。每行要求一个 positive crop。

```bash
python -m annotation.black_crops \
  --dataset /path/to/train.parquet \
  --output-dir results/black-negative
```

生成与 positive 同尺寸的纯黑 RGB PNG；相同尺寸共享图片。输出 `train.parquet`、`crops/` 和 `manifest.json`，仅修改 `teacher_neg_images`，并校验其他列保持一致。此入口没有断点续跑参数，请使用独立输出目录。

# Annotation Guide

Run all commands from the repository root. Input image paths must be accessible on the machine running the scripts.

## Environment and API configuration

Activate a Python environment with the project dependencies installed, then install the annotation dependencies:

```bash
python -m pip install Pillow pyarrow openai tqdm
```

For initial setup, copy the template. If `.seed.env` already exists, edit it directly:

```bash
cp .seed.env.example .seed.env
```

Set `ARK_BASE_URL`, `ARK_MODEL`, and `ARK_API_KEY` in `.seed.env`. The scripts also support vision model services compatible with OpenAI Chat Completions. If the key is stored in another environment variable, set `ARK_API_KEY_ENV` to that variable's name. Do not commit credentials to Git.

Override these settings with `--base-url`, `--model`, and `--api-key-env`. The `--api-key-env` argument takes an environment variable name, not the key itself.

The random and nearby launchers read `.project.env` and `.seed.env` and use Python from the active environment. Set `PYTHON_BIN` to select an interpreter. The positive crop launcher, `label.sh`, also reads `.ssh_server.env` and activates a Conda environment. Set `LABEL_CONDA_ENV` to its name; the fallback order is `SEED_CONDA_ENV`, `FILTER_CONDA_ENV`, then `verl`. If necessary, set `REMOTE_CONDA_INIT` in `.project.env` to initialize Conda.

## Positive bbox / point

```bash
LABEL_CONDA_ENV=your-env bash annotation/label.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/positive-preview \
  --limit 10 --workers 4
```

Input records must contain the original image and question. Inspect the preview annotations, then remove `--limit` and select a new output directory for a full run. Use `--help` to list all arguments.

## Random / nearby negative crops

Provide one Parquet file containing the original image, question, and exactly one positive crop in `teacher_images` per record. The negative crop dimensions are determined by the positive image's actual pixel dimensions.

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

`--workers` sets API concurrency per process. Concurrent runs add to the total concurrency. For a full run:

```bash
bash annotation/label_nearby_negative.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/nearby-full \
  --workers 64
```

Random annotation asks the model to select an unrelated region; it does not sample regions uniformly at random. Nearby annotation asks for a region immediately beside the positive crop. Both provide the original image, positive crop, and question to the model and request a negative box with the same pixel dimensions and a point inside it.

By default, the scripts save crop images and export `train.parquet` once every selected record succeeds. Only `teacher_neg_images` is replaced or added. When using `--limit`, the exported file contains only the selected records and is not a full dataset.

Resume is enabled by default. Repeat the exact command with the same output directory to skip successful records whose crop files exist and retry failed records. Use a new directory when changing inputs or parameters, including switching from a preview to a full run. Failed records prevent training Parquet export; inspect `error` in `grounding.jsonl` and the counts in `summary.json`.

Use `--no-write-train-parquet` to save annotation records without exporting training data. Add `--save-crops` to also save crop images. See [negative crop details](NEGATIVE_CROPS.md) for coordinates, geometric validation, and output fields.

## Black negative crops

This entry point requires no API. Input must contain `images`, `teacher_images`, and `teacher_neg_images`, with accessible local image paths and exactly one positive crop per record.

```bash
python -m annotation.black_crops \
  --dataset /path/to/train.parquet \
  --output-dir results/black-negative
```

The script generates pure black RGB PNGs matching the positive crop dimensions. Records with the same dimensions share an image. Outputs are `train.parquet`, `crops/`, and `manifest.json`. Only `teacher_neg_images` changes; all other columns are checked for equality. This entry point has no resume option; use a separate output directory.

# Annotation Guide

Run all commands from the repository root. Input image paths must be accessible on the machine running the scripts.

## Choose an entry point

| Crop type | Entry point | Selection rule |
| --- | --- | --- |
| Positive | `python -m annotation.label.run` or `label.sh` | Question-relevant bbox and point |
| Nearby negative | `generate_nearby_negative.sh` | Nearest same-size box; prefer IoU 0, allow at most 0.05 only when necessary |
| Blank / black negative | `python -m annotation.black_crops` | Pure black RGB image, same size as positive |

Start with positive crops, inspect the results, then generate each negative
variant from the same validated positive dataset into a separate output directory.
Do not chain ablation variants by feeding one negative variant into another.
## Environment and positive annotation configuration

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

The positive crop launcher, `label.sh`, reads `.project.env`, `.seed.env`, and `.ssh_server.env` and activates a Conda environment. Set `PYTHON_BIN` to select an interpreter. Set `LABEL_CONDA_ENV` to its name; the fallback order is `SEED_CONDA_ENV`, `FILTER_CONDA_ENV`, then `verl`. If necessary, set `REMOTE_CONDA_INIT` in `.project.env` to initialize Conda.

## Positive bbox / point

With an active Python environment, export the API settings in your shell and run:

```bash
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export ARK_MODEL="doubao-seed-2-0-pro-260215"
# Set ARK_API_KEY securely in your shell, or choose an existing variable with --api-key-env.
python -m annotation.label.run \
  --dataset data/train.parquet --image-root data \
  --output-dir results/positive-preview --limit 10 --workers 4
```

The Python module does not load `.seed.env` automatically. Alternatively, use the
wrapper to load the environment files and activate Conda:

```bash
LABEL_CONDA_ENV=your-env bash annotation/label.sh \
  --dataset data/train.parquet --image-root data \
  --output-dir results/positive-preview --limit 10 --workers 4
```

Input records need an original image and a question (for example `images` and
`question`; the loader also supports prompt/message-based questions). Existing
`extra_info`, if present, should be a mapping. The model returns bbox/point
coordinates on a 0–999 canvas, converted to original-image pixels before cropping.

Outputs include `grounding.jsonl`, `summary.json`, `crops/`, and `train.parquet`.
The training output updates `images`, `teacher_images`, `teacher_raw_prompt`, and
positive crop fields in `extra_info`, including `bbox_pixel` and `point`.
**Only successful rows with saved crops are exported.** This entry point can
produce a partial Parquet when annotations fail; verify `failed == 0` and the
expected row count before using it as a full dataset. It does not promise to
preserve the original Arrow schema.

Positive resume skips every already recorded index, including failed records.
Repeating the same command therefore does not retry those failures. To rerun the
same selection, use a new output directory (or `--no-resume`, which overwrites the
annotation record file). Inspect the preview, then remove `--limit` and use a new
output directory for a full run. The geometric and black generators require an
existing `teacher_neg_images` column; retain it when preparing the source training
schema. Their output is not a substitute for missing required training fields.

## Nearby negative crops

Use this entry point for nearest same-size crops with IoU in **[0, 0.05]**:

```bash
python -m pip install Pillow pyarrow
bash annotation/generate_nearby_negative.sh \
  --dataset data/train.parquet \
  --image-root data \
  --output-dir results/nearby-geometric \
  --workers 4 --seed 42
```

The input must contain `images`, `teacher_images`, `teacher_neg_images` (each a
one-element list of image dictionaries), and `extra_info.bbox_pixel` in original
pixel coordinates `[left, top, right, bottom]`. `extra_info` can be a Parquet struct
or a JSON string. The positive bbox must be within the original image and match
the positive crop's actual width and height. Missing bboxes are errors; this tool
does not recover coordinates or call an annotation model. Embedded image bytes
are unsupported. `--image-root` resolves relative input image paths and defaults
to the input Parquet's directory.

The algorithm first finds the closest disjoint box with the same dimensions.
Only if no disjoint box fits does it allow up to 5% IoU. Within each stage, it
minimizes squared center distance; the seed and row index resolve ties. This is
not uniform sampling over IoU values. Geometry does not guarantee that the crop
excludes answer-bearing evidence.

Outputs are `crops/`, `grounding.jsonl`, `summary.json`, and `train.parquet`.
The script verifies saved PNG pixels and dimensions, replaces `teacher_neg_images`,
and updates the negative crop fields in `extra_info`; other fields and row order
are preserved. New metadata fields may extend the `extra_info` struct schema.
Negative image paths in the exported Parquet are absolute paths resolved at run
time. Original image references are preserved, so keep their original dataset
root configured when training. To publish or relocate the dataset, package the
images and rewrite path references first.

Use a new output directory for every run: existing directories are refused.
If any row fails (including geometrically impossible boxes), errors are saved in
`grounding.jsonl` and no final training Parquet is exported. `--workers` controls
CPU/image processing concurrency.

## Blank / black negative crops

Input must contain `images`, `teacher_images`, and `teacher_neg_images`, with accessible local image paths and exactly one positive crop per record.

```bash
python -m annotation.black_crops \
  --dataset /path/to/train.parquet \
  --output-dir results/black-negative
```

Here, **blank means pure black**, not white, blurred, or transparent. No bbox or
point is annotated because the negative image is synthetic.

This script has no `--image-root` option: source paths must be absolute or resolve
relative to your current working directory, not relative to the Parquet file.

The script generates pure black RGB PNGs matching the positive crop dimensions. Records with the same dimensions share an image. Outputs are `train.parquet`, `crops/`, and `manifest.json`. Only `teacher_neg_images` changes; all other columns are checked for equality. This entry point has no resume option; use a separate output directory.

## Checks before training

- Verify the selected row count and inspect `summary.json` or `manifest.json`.
- Inspect positive and nearby crops visually; valid coordinates
  do not establish semantic correctness.
- Keep the positive bbox available for geometric nearby and overlap validation.
- Check that every exported image path resolves in the training environment.
  Generated crop paths are local filesystem references, not portable image bytes.
- Black crops preserve `extra_info`; existing negative bbox metadata is not
  refreshed and does not describe the synthetic image. Geometric nearby updates
  its negative metadata. Training should read the new `teacher_neg_images`.
- Use separate output directories and retain the original input dataset.

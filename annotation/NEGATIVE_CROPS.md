# Negative Bounding Box and Point Annotation

Configure the environment and API as described in the [annotation guide](README.md), then use the separate random and nearby entry points:

```bash
bash annotation/label_random_negative.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/random-negative-preview --limit 10 --workers 4

bash annotation/label_nearby_negative.sh \
  --dataset /path/to/train.parquet \
  --output-dir results/nearby-negative-preview --limit 10 --workers 4
```

Activate the required Python environment or set `PYTHON_BIN`. Dependencies are Pillow, PyArrow, OpenAI, and tqdm. API configuration uses `ARK_MODEL`, `ARK_BASE_URL`, and `ARK_API_KEY_ENV`, as in positive crop annotation; override them with `--model`, `--base-url`, and `--api-key-env`. The scripts read `.project.env` and `.seed.env` and do not write API keys to outputs. For a full run, remove `--limit` and use a new output directory.

Each request includes two images: **the original image first, followed by the positive crop from `teacher_images`**, plus the question. The model directly annotates a negative bounding box and point. The request does not include the reference answer. The complete prompt is defined in `build_prompt` in `annotation/label/negative_grounding.py`.

- **Random:** Select an arbitrary region unrelated to the question and containing no answer evidence, without specifically seeking the positive target's neighborhood. This is model-selected annotation, not uniform random sampling.
- **Nearby:** Locate the positive crop in the original image, then select the local context immediately outside it. Exclude the target and answer evidence, and avoid distant regions.

The model uses the same output format as positive crop annotation:

```text
<bbox>100 200 300 400</bbox>
<point>200 300</point>
```

Coordinates are integers normalized to 0–999 relative to the original image. The model selects the box position; its width and height must exactly match the positive crop's pixel dimensions. The prompt supplies the original image dimensions, positive crop dimensions, and corresponding normalized width and height. Proposals with substantially different dimensions are rejected. Only rounding errors caused by coordinate quantization are corrected; images are never resized to match the positive crop.

The final crop uses `bbox_pixel`. The raw response and proposed pixel box are preserved in `raw_response` and `proposed_bbox_pixel`. The model's point must lie inside the box, and both normalized and pixel coordinates are saved. A zero-area box indicates that no valid region was found and is recorded as a failure. Out-of-range coordinates, reversed boxes, and points outside the box are rejected rather than silently corrected.

When `extra_info.bbox_pixel` is available, the script also rejects overlap with the positive box. Without this metadata, it cannot perform that geometric check; localization depends on the model interpreting both images and subsequent manual inspection.

Outputs include `grounding.jsonl` (raw responses, prompts, boxes, points, crop paths, and errors), `crops/*.png`, `summary.json`, and `manifest.json`. By default, `train.parquet` is exported only after all selected records succeed. Only `teacher_neg_images` is replaced or added, and every other source column is checked for equality. Negative boxes and points stay in the separate annotation records and do not overwrite positive box metadata.

Repeat the same command to resume. Successful records with existing crop files are skipped, and failures are retried. `--limit` is applied before skipping successful records. Changes to configuration or input require a new output directory. Each entry point accepts one Parquet file and does not support `--random` row shuffling. Use `--no-write-train-parquet` to save annotations without training data export; add `--save-crops` to also save crop images. Inspect samples before training, since geometric validation does not guarantee semantic correctness.

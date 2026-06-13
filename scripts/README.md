# V-Zero Training

This directory contains the public V-Zero on-policy distillation launchers. The main launcher is `run_vzero_qwen35_vl_fsdp.sh`. It configures Qwen3.5-VL on-policy rollouts, teacher replay on positive and negative evidence crops, and token-level evidence-gated distillation.

The complete training implementation lives in the repository-local `verl/` package:

- teacher-crop multimodal data fields
- teacher log-prob replay
- positive/negative evidence weighting
- distillation losses under `distillation.*`
- FSDP rollout/training integration

## Requirements

Set up an environment with compatible versions of PyTorch, Transformers, vLLM, FlashAttention, causal-conv1d, FLA, qwen-vl-utils, and Ray. CUDA toolkit/runtime versions should match the PyTorch/vLLM build. If CUDA extensions need a non-default toolkit, export `CUDA_HOME` before launching.

Install the bundled training package in editable mode when setting up a fresh environment:

```bash
cd V-Zero
uv pip install -e .
```

## Data Format

The training parquet is expected to contain these columns:

| Column | Purpose |
| --- | --- |
| `prompt` | Student full-image prompt messages. |
| `images` | Full image payloads used by the student rollout. |
| `teacher_raw_prompt` | Positive teacher prompt messages containing the crop marker. |
| `teacher_images` | Positive evidence crop payloads. |
| `teacher_neg_raw_prompt` | Negative teacher prompt messages. |
| `teacher_neg_images` | Negative or irrelevant evidence crop payloads. |
| `reward_model` | Ground-truth metadata used by the verl reward path. |
| `extra_info` | Optional metadata for logging and analysis. |

For ablations, the same schema can include extra image columns such as `teacher_random_images`; pass the corresponding key with `TEACHER_CROP_IMAGE_KEY`.

## Launch

```bash
cd V-Zero

STUDENT_MODEL=<student-model-path> \
TEACHER_MODEL=<teacher-model-path> \
TRAIN_FILE=<train-parquet-path> \
VAL_FILE=<validation-parquet-path> \
RUN_TAG=vzero_qwen35_27b_4b \
bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Qwen3.5 thinking mode is disabled by default through `APPLY_CHAT_TEMPLATE_ENABLE_THINKING=False`.

## Common Overrides

Use a local model root:

```bash
MODEL_ROOT=<model-root> \
TRAIN_FILE=<train-parquet-path> \
VAL_FILE=<validation-parquet-path> \
bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Run the random-positive-crop ablation:

```bash
TEACHER_CROP_IMAGE_KEY=teacher_random_images \
RUN_TAG=vzero_random_positive_crop \
bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Change rollout count:

```bash
ROLLOUT_N=12 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Reduce vLLM rollout memory reservation:

```bash
ROLLOUT_GPU_MEM_UTIL=0.35 TEACHER_GPU_MEM_UTIL=0.45 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Use a specific CUDA toolkit for JIT or extension builds:

```bash
export CUDA_HOME=<cuda-toolkit-path>
bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

## Outputs

By default, logs and rollout generations are written under:

- `./final_logs/<RUN_TAG>/rollouts`
- `./final_logs/<RUN_TAG>/val_generations`

verl checkpoints are controlled by the underlying trainer config and experiment name. Set `PROJECT_NAME`, `EXPERIMENT_NAME`, `SAVE_FREQ`, and `TEST_FREQ` to match the desired experiment layout.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>`
- `distillation.evidence_weighting.enabled=True`
- `distillation.evidence_weighting.negative_mode=hard_negative_crop`
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`

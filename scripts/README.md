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
uv pip install -e . --no-deps
```

## FA3 training through the main launcher

`run_vzero_qwen35_vl_fsdp.sh` defaults to the FA3 training runtime,
including teacher scheduler compatibility, FLA/TileLang dispatch, isolated Ray
and tmux processes, and per-actor FA3 forward/backward verification.

Select a profile before preparation:

| `FA3_PROFILE` | Default student | Default nodes | Packed-sequence mode |
| --- | --- | --- | --- |
| `four_b` (default) | Qwen3.5-4B | 2 | Baseline |
| `nine_b` | Qwen3.5-9B | 4 | Baseline |
| `four_b_boundary` | Qwen3.5-4B | 2 | Explicit boundary variant |

Each node needs eight GPUs: four for the student and four for the teacher. All
profiles share `fa3/`. `NNODES` may override the node count to 2 or 4 while keeping
this GPU split. Set `CPUS_PER_NODE` to the allocated CPU count (default 184), with
sufficient CPUs for both worker placement groups and rollout tasks.

Defaults are 100 steps, batch 32, mini-batch 16, eight rollouts, constant learning
rate `1e-6`, no warmup and SP=1. Set `TOTAL_TRAINING_STEPS` and `SAVE_FREQ` to change
the duration; the final step is also saved. Batch size, rollout count, optimizer
and SP settings are guarded to preserve the recipe. Both model sizes use baseline
behavior; only `four_b_boundary` enables the separate boundary variant.

Checkpoints are synchronous and **model-only**, and all scheduled checkpoints are
retained. Use a fresh output directory for each run. This runtime does not restore
optimizer state or provide full training-state resume.

The pinned training backend is:

```text
kernels-community/flash-attn3@7cb368cf8278b583132eb72cbf312d54586df2e2
```

Provide an existing compatible shared runtime containing `.venv`, its Python base,
the downloaded model weights and populated kernel/Hugging Face caches. Installing
`flash-attn` alone is insufficient. This entry does not provision cloud machines,
install packages or download weights. The runtime must provide Ray and tmux and
meet the [runtime compatibility requirements](fa3/README.md#supported-recipe-and-compatibility). The default FA3 call targets belong to
the pinned provider; a different kernel requires corresponding verified targets.

Export the same settings on all nodes. Example paths below are placeholders:

```bash
export FA3_PROFILE=four_b
export RUN_TAG=vzero-fa3-4b-run1
export TOTAL_TRAINING_STEPS=100
export SAVE_FREQ=10
export CPUS_PER_NODE=184
export VZERO_FA3_SHARED_MOUNT=/shared
export VZERO_FA3_ENV_ROOT=/shared/work/fa3-runtime
export VZERO_FA3_PYTHON_BASE=/shared/work/python-base
export VZERO_FA3_WANDB_ENV_FILE=/shared/private/wandb.env
export STUDENT_MODEL=/shared/models/Qwen3.5-4B
export TEACHER_MODEL=/shared/models/Qwen3.5-27B
export TRAIN_FILE=/shared/data/train.parquet
export VAL_FILE=/shared/data/val.parquet
```

Run commands from the repository root. For the 9B profile, set `FA3_PROFILE=nine_b`
and `STUDENT_MODEL` to the Qwen3.5-9B directory.

The repository, runtime, Python base, data, model weights and caches must be on the
actual shared mount. `VZERO_FA3_TRAIN_PYTHON` defaults to
`$VZERO_FA3_ENV_ROOT/.venv/bin/python`; caches default to its `.cache/huggingface`
and `.cache/kernels` directories. The private W&B file must contain literal
`export WANDB_BASE_URL=...` and `export WANDB_API_KEY=...` assignments; optional
`WANDB_ENTITY` and `WANDB_PROJECT` are supported. Do not commit this file.
For hosts that require CUDA compatibility libraries, set `VZERO_FA3_CUDA_COMPAT_DIR`
to the verified shared compatibility directory. The runtime checks the driver
and library selection; do not change the driver globally.

First run the full input/configuration preparation once, before launching nodes:

```bash
bash scripts/run_vzero_qwen35_vl_fsdp.sh --prepare
```

Preparation verifies image paths, dataset fields, model shards, dependencies and
checkpoint space, then writes the resolved configuration and source fingerprints
under `results/$RUN_TAG`. Use a fresh run name containing lowercase letters, digits and hyphens, starting with a letter. Do not edit code/configuration
between preparation and launch.

Then run on **each allocated node**, using its distinct zero-based node rank:

```bash
export MASTER_ADDR=head-node-address
export MASTER_PORT=6379
export RANK=0  # 0..1 for the two-node profiles; 0..3 for nine_b
bash scripts/run_vzero_qwen35_vl_fsdp.sh --launch
```

Omitting the mode is equivalent to `--launch`. The shell sets the node count, role environment, FLA flags, FA3 provider and
patch paths, then calls the shared runtime. It does not launch peers over SSH; every allocated node
must execute the command. Only use nodes assigned to this job. Completion requires
successful node results and actual FA3 calls on all actor ranks, not just startup.
The evidence is saved in the run directory. `precision_pass=false` in the runtime
receipt means no numerical-equivalence claim is made by this training run.

## Direct FA2 training

The configurable direct training route remains available explicitly:

```bash
VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

The environment options and Hydra overrides below apply to that direct route.
FA3 profiles use the same shell recipe with the guarded settings described above,
and accept environment settings rather than positional Hydra overrides. Training
attention selection is separate from vLLM rollout/teacher attention selection.

See [`fa3/README.md`](fa3/README.md) for the patch inventory,
compatibility requirements and runtime evidence files.

## Data Format

The training parquet is expected to contain these columns:

| Column | Purpose |
| --- | --- |
| `prompt` | Student full-image prompt messages. |
| `images` | Full image payloads used by the student rollout. |
| `teacher_raw_prompt` | Positive teacher prompt messages containing the crop marker. |
| `teacher_images` | Positive evidence crop payloads. |
| `teacher_neg_raw_prompt` | Optional negative teacher messages. If absent, the positive teacher messages are reused with negative crop images substituted. |
| `teacher_neg_images` | Negative or irrelevant evidence crop payloads. |
| `data_source` | Non-empty dataset identifier required by the default reward manager. |
| `reward_model` | Metadata containing `ground_truth` for the default reward manager; preserve genuine null labels. |
| `extra_info` | Optional metadata for logging and analysis. |

The FA3 preflight requires image payloads to resolve to existing **absolute local
paths on the shared mount**; embedded image bytes and remote URLs are not accepted.
Provide one training Parquet and one non-empty validation Parquet. Preserve the
original metadata fields; do not invent answer labels to satisfy the schema.

For ablations, the same schema can include extra image columns such as `teacher_random_images`; pass the corresponding key with `TEACHER_CROP_IMAGE_KEY`.

## Direct FA2 launch

```bash
cd V-Zero

STUDENT_MODEL=/path/to/student \
TEACHER_MODEL=/path/to/teacher \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
RUN_TAG=vzero_qwen35_27b_4b \
VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Qwen3.5 thinking mode is disabled by default through `APPLY_CHAT_TEMPLATE_ENABLE_THINKING=False`.

## Direct FA2 overrides

Use a local model root:

```bash
MODEL_ROOT=/path/to/models \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Run the random-positive-crop ablation:

```bash
TEACHER_CROP_IMAGE_KEY=teacher_random_images \
RUN_TAG=vzero_random_positive_crop \
VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Change rollout count:

```bash
ROLLOUT_N=12 VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Reduce vLLM rollout memory reservation:

```bash
ROLLOUT_GPU_MEM_UTIL=0.35 TEACHER_GPU_MEM_UTIL=0.45 VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Use a specific CUDA toolkit for JIT or extension builds:

```bash
export CUDA_HOME=/path/to/cuda
VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

## Outputs

FA3 outputs default to `results/$RUN_TAG/` (override with
`VZERO_FA3_TRAIN_OUTPUT_DIR` before preparation). This includes resolved configs,
per-node execution records, per-actor FA3 call records, update receipts,
`checkpoints/`, and `training_complete.json`. See the
[output inventory](fa3/README.md#evidence-and-failures) for verification details.

For the direct FA2 route, logs and rollout generations default to:

- `./final_logs/<RUN_TAG>/rollouts`
- `./final_logs/<RUN_TAG>/val_generations`

Direct FA2 checkpoints are controlled by the underlying trainer config and experiment name. Set `PROJECT_NAME`, `EXPERIMENT_NAME`, `SAVE_FREQ`, and `TEST_FREQ` to match the desired experiment layout.

## Direct FA2 Hydra flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>`
- `distillation.evidence_weighting.enabled=True`
- `distillation.evidence_weighting.negative_mode=hard_negative_crop`
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`

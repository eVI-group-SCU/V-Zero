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

## FA3 training through the main launcher

`run_vzero_qwen35_vl_fsdp.sh` defaults to the complete FA3 retraining runtime,
including teacher scheduler compatibility, FLA/TileLang dispatch, isolated Ray
and tmux processes, and per-actor FA3 forward/backward verification.

Choose `FA3_PROFILE=four_b` (two nodes), `four_b_boundary` (two nodes with the
packed-sequence boundary correction), or `nine_b` (four nodes). Each node needs
eight GPUs: four for the student and four for the teacher. All profiles use one shared implementation in `fa3/`. Defaults are
100 steps, batch 32, mini-batch 16, eight rollouts, constant learning rate 1e-6,
and model-only checkpoints every ten steps. Set `TOTAL_TRAINING_STEPS` and
`SAVE_FREQ` to change the duration. The final step is always saved. `NNODES`
may be 2 or 4; each node retains the four-student/four-teacher split.
Set `CPUS_PER_NODE` to the allocated CPU count (default 184, minimum 16).
Batch size, rollout count, optimizer, and SP=1 remain guarded to preserve this
recipe. Incompatible settings fail before training. Both model sizes default to
baseline; only `four_b_boundary` enables the separate packed-boundary correction.

The pinned training backend is:

```text
kernels-community/flash-attn3@7cb368cf8278b583132eb72cbf312d54586df2e2
```

Provide an existing compatible shared runtime containing `.venv`, its Python base,
the downloaded model weights and populated kernel/Hugging Face caches. Installing
`flash-attn` alone is insufficient. This entry does not provision cloud machines,
install packages or download weights. The runtime must provide Ray and tmux and
match the checks in the selected harness. The default FA3 call targets belong to
the pinned provider; a different kernel requires corresponding verified targets.

Export the same settings on all nodes. Example paths below are placeholders:

```bash
export FA3_PROFILE=four_b
export RUN_TAG=vzero-fa3-4b-run1
export VZERO_FA3_SHARED_MOUNT=/shared
export VZERO_FA3_ENV_ROOT=/shared/work/fa3-runtime
export VZERO_FA3_PYTHON_BASE=/shared/work/python-base
export VZERO_FA3_WANDB_ENV_FILE=/shared/private/wandb.env
export STUDENT_MODEL=/shared/models/Qwen3.5-4B
export TEACHER_MODEL=/shared/models/Qwen3.5-27B
export TRAIN_FILE=/shared/data/train.parquet
export VAL_FILE=/shared/data/val.parquet
```

The repository and runtime must be on the actual shared mount. `VZERO_FA3_TRAIN_PYTHON` defaults to
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
under `results/$RUN_TAG`. Use a fresh run name. Do not edit code/configuration
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
FA3 profiles use the same shell recipe with the guarded settings described above.
They require fresh output directories and save model weights only: they do not
support resuming optimizer state. Do not use a model-only checkpoint as a complete
training-state checkpoint. Training attention selection is
separate from vLLM rollout/teacher attention selection.

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
VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
```

Qwen3.5 thinking mode is disabled by default through `APPLY_CHAT_TEMPLATE_ENABLE_THINKING=False`.

## Common Overrides

Use a local model root:

```bash
MODEL_ROOT=<model-root> \
TRAIN_FILE=<train-parquet-path> \
VAL_FILE=<validation-parquet-path> \
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
export CUDA_HOME=<cuda-toolkit-path>
VZERO_TRAIN_BACKEND=fa2 bash scripts/run_vzero_qwen35_vl_fsdp.sh
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

# FA3 training runtime

Use the public entry in [`../run_vzero_qwen35_vl_fsdp.sh`](../run_vzero_qwen35_vl_fsdp.sh).
See [`../README.md`](../README.md#fa3-training-through-the-main-launcher) for preparation
and per-node launch commands. `--prepare` performs CPU checks; `--launch` starts the
allocated training job. Preparation alone is not evidence of successful FA3 training.

## Components

| File | Responsibility |
| --- | --- |
| `run.sh` | Translate public environment settings into a single FA3 runtime configuration; set the immutable provider, cache paths, FLA dispatch and profile defaults. |
| `launch.py` | Capture the public shell's Hydra recipe, validate inputs and source fingerprints, select CUDA libraries per node, and start isolated Ray/tmux processes. |
| `runtime_hooks.py` | Reserve student/teacher placement groups on each node, check actual actor FA3 forward/backward calls, bound teacher waits, verify checkpoints and write completion evidence. |
| `teacher_scheduler.py` | Teacher-only vLLM Mamba/vision scheduling compatibility patch. Align before encoder scheduling and recover an unaligned resume boundary. Reject unsupported scheduler source layouts instead of guessing. |
| `packed_boundary.py` | Audit packed Qwen3.5 positions in baseline mode. In explicit boundary mode, also pass sequence boundaries to the native convolution and Gated DeltaNet kernels, including checkpoint recomputation. |

The profiles share all source files. `four_b` defaults to two nodes, `nine_b` to
four; `four_b_boundary` is the explicit two-node packed-boundary variant. Baseline
mode does **not** apply that numerical correction. FA3 attention and Gated DeltaNet
are different kernels: selecting FA3 does not itself repair recurrent packed
sequence boundaries.

## Training framework fixes

These belong to the common V-Zero implementation, not to a launch-time FA3 patch:

- `verl/trainer/distillation/teacher_alignment.py` and `teacher_manager.py` align
  replayed teacher tokens with the student's response positions.
- `verl/trainer/distillation/evidence_weighting.py` starts the response prediction
  slice at `prompt_width - 1`, retaining the first response-token likelihood.
- `verl/trainer/distillation/losses.py` runs PPO metadata initialization before
  distillation aggregation so global batch normalization is available.

The runtime uses the existing relative-evidence objective. It does not install a
second distillation method or replace the teacher/student loss.

## Supported recipe and compatibility

- Two or four exclusive nodes, eight GPUs per node, four student and four teacher
  GPUs on each node. SP=1; FSDP spans the student ranks.
- Batch 32, mini-batch 16, eight rollouts, constant learning rate `1e-6`, no warmup.
- Defaults: 100 updates and save every ten updates. Override with
  `TOTAL_TRAINING_STEPS` and `SAVE_FREQ`; non-divisible final steps are also saved.
- Synchronous **model-only** checkpoints; all scheduled checkpoints are retained.
  Runs start fresh. Optimizer-state resume is not implemented by this runtime.
- Online W&B is required by the current run-evidence checks. Provide its settings
  in a private literal-export file, never in tracked scripts.
- Ray `2.55.1`; the placement adapter rejects other versions. The compatible
  runtime stack is Python 3.12, PyTorch `2.11.0+cu130`, Transformers `5.9.0`,
  vLLM `0.22.1rc1.dev58+g68dafcca7`, FLA `0.5`, Triton `3.6`, and TileLang.
  The scheduler patch additionally checks the installed implementation structurally.
- Training uses the immutable provider
  `kernels-community/flash-attn3@7cb368cf8278b583132eb72cbf312d54586df2e2`.
  Installing the `flash-attn` distribution alone does not configure this provider.
- The provider and model caches must already exist on the shared mount. Downloads
  are disabled during training. Runtime and Python base paths are user-supplied;
  no personal directory is built into the launcher.
- `FLA_DISABLE_BACKEND_DISPATCH=0` and `FLA_TILELANG=1` preserve the supported
  backward dispatch. The runtime does not bypass kernel correctness guards.

These compatibility requirements apply to the training actor. vLLM has its own
attention backend selection for rollout and teacher inference.

## Evidence and failures

`results/$RUN_TAG/` contains:

- `baseline_config.json`, `resolved_config.json`, `config_diff.json`;
- `prepared.json`: input inventory, source hashes, package versions and run controls;
- `execution/`: node readiness, CUDA probe/selection, node results and failure records;
- `actor-runtime-rank-*.json`: actual FA3 call counts for every student rank;
- packed-boundary audit records, `teacher-wait.json`, and placement records;
- `updates/step-*.json`, checkpoint receipts and `training_complete.json`.

Check actual update and completion receipts. A configured backend string, a live
process, or successful preparation alone is insufficient. `precision_pass=false`
means that numerical equivalence with another backend has not been established;
it is not a failed training status. Failed execution preserves evidence and does
not automatically retry, delete checkpoints, or submit another job.

CPU regressions (no Ray cluster or GPU workload):

```bash
python3 -m unittest discover -s tests/scripts -v
```

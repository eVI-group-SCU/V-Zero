# VA OPD implementation in V Zero

[Visual-Advantage On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2605.21924v1), sections 2 and 3, defines the training objective implemented in this fork. It uses the student full-image rollout and scores those exact response tokens under a frozen teacher with the original and pixelated images. No crop annotations or answer labels enter the objective.

## Objective mapping

- Visual advantage is `relu(logp_original - logp_pixelated)` on valid response tokens.
- Pixelation downsamples each spatial dimension to 10% with bilinear interpolation, then restores the original size with nearest interpolation. This runs after the existing multimodal image preprocessing.
- Rollout scores are masked means of VA. Standardization uses the population standard deviation within each prompt UID, followed by softmax at temperature 1.
- The highest 20% of valid response positions form the visual group. The rest form the language group. Each group's KL is averaged independently; the two averages have weights 0.5 and 0.5.
- The teacher conditioned on the original image supplies the distillation target. Pixelated scores only supply detached weights.

The controller computes these signals on the complete sibling batch before actor DP and microbatch splits. The actor uses `seq-mean-token-sum` aggregation. Rollout weights stored in `opd_weights` are K times the paper's probability weights, compensating for the actor's mean over K rollouts; their sibling mean is 1. Token coefficients already divide by each group's size.

## Implementation conventions and limits

The paper does not specify integer rounding or tied VA ranks. This implementation uses `ceil(p*T)`, keeps both groups nonempty when T >= 2, and breaks ties by response order. One-token responses use coefficient 1. Empty responses contribute zero and retain a sibling slot. Unequal sibling counts fail explicitly.

The paper v1 source references an implementation appendix that is omitted from its main document and source archive. Learning rate, sequence lengths, and the numerical KL estimator are not specified there. This implementation reuses V-Zero's sampled reverse-KL `k1` policy-gradient/PPO estimator by default; it does **not** compute an exact full-vocabulary KL. PPO clipping away from the rollout policy and sampled-token estimates are approximations. The raw KL/logprob clamps are disabled in the VA launcher. The grouped weights and aggregation implement the paper formulas independently of this estimator choice.

The supplied recipe selects Qwen3-VL-8B-Instruct -> Qwen3-VL-2B-Instruct, 4 rollouts, batch size 16, 5 epochs, and the paper VA parameters. Learning rate defaults to V-Zero's 1e-6; token limits and GPU allocations also inherit its launcher and should be chosen for the available hardware. This is a method implementation, not a claimed reproduction of the reported benchmark scores.

## Run

Use the Linux CUDA environment described in the repository README. Supply ordinary veRL multimodal Parquet files with `prompt` and `images` columns (the existing `raw_prompt` conversion handles image placeholders), plus the metadata expected by the existing trainer/reward adapter. No `teacher_images` or `teacher_raw_prompt` columns are needed.

```bash
TRAIN_FILE=/data/geometry3k/train.parquet \
VAL_FILE=/data/geometry3k/validation.parquet \
STUDENT_MODEL=/models/Qwen3-VL-2B-Instruct \
TEACHER_MODEL=/models/Qwen3-VL-8B-Instruct \
bash scripts/run_va_opd_qwen3_vl_fsdp.sh
```

For ViRL39K or a different teacher size, change the input paths/model. Data downloading, preprocessing those two datasets, and official eight-benchmark avg@8 reproduction are not supplied by this change. Existing evaluation helpers remain available, but their default protocol is not asserted to match the paper's official scoring.

Ablations: `VA_ROLLOUT_REWEIGHTING=False` disables rollout weights; `VA_TOKEN_GROUPING=False` uses a uniform token mean. Disable both for an OPD control with the same teacher scoring overhead.

## Validation

CPU tests in `tests/distillation/test_visual_advantage.py` cover the paper objective and gradients, shuffled sibling grouping, microbatch partitioning, padding, negative differences, tied scores, empty/short responses, ablations, and pixel interpolation. `test_va_opd_integration.py` covers actual loss aggregation and launcher configuration. GPU teacher scoring, distributed backward, and benchmark gains require a Linux CUDA run.


On 2026-10-13, all 16 CPU tests passed using `python tests/distillation/run_cpu_tests.py`; this runner bypasses the package initializers that register Ray workers, while loading the actual loss reducers and configuration classes. The launcher test also composes the generated overrides through Hydra. Lint, Python syntax, and shell syntax checks passed. No CUDA model forward or distributed training was executed.

In a Linux training environment, run `python -m pytest tests/distillation`. For a CPU development environment with the dependencies listed at the top of the runner installed, use `python tests/distillation/run_cpu_tests.py`.

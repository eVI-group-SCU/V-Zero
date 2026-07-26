# V-Zero

**V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating for Fine-Grained Visual Reasoning**

[[Paper]](https://arxiv.org/abs/2606.25319) [[Model]](https://huggingface.co/hao05/v-zero-4b)

## Overview

V-Zero improves fine-grained visual reasoning without annotated answer labels. The student model samples on-policy reasoning trajectories from the full image, while a teacher model replays the same trajectories with paired positive and negative visual evidence views. By contrasting teacher support under the task-relevant crop and an irrelevant crop, V-Zero estimates how well each trajectory is grounded in visual evidence and uses this signal to gate dense token-level distillation. The resulting training objective keeps standard full-image inference unchanged while providing answer-label-free supervision for localized visual reasoning.

<p align="center">
  <img src="resource/method.png" alt="V-Zero Method Overview" width="100%">
</p>

## Training

The V-Zero training implementation is included under [`verl/`](verl/), with the public launcher in [`scripts/run_vzero_qwen35_vl_fsdp.sh`](scripts/run_vzero_qwen35_vl_fsdp.sh). It configures the on-policy distillation recipe with teacher replay, positive/negative evidence crops, and evidence-gated token distillation.

Install the training package from the repository root:

```bash
uv pip install -e .
```

See [`scripts/README.md`](scripts/README.md) for data schema, environment variables, and launch examples.

## Evaluation

The evaluation framework is included under [`eval/`](eval/), with launch helpers in [`scripts/start_vllm_server.sh`](scripts/start_vllm_server.sh) and [`scripts/eval_benchmarks.sh`](scripts/eval_benchmarks.sh). Copy the Seed judge template and fill local credentials before running judge-based evaluation:

```bash
cp .seed.env.example .seed.env
```

Start the vLLM OpenAI-compatible server:

```bash
bash scripts/start_vllm_server.sh \
  --model <model-checkpoint-path-or-hf-id> \
  --served-model-name <served-model-name> \
  --port 8000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8
```

Run benchmark evaluation from another terminal:

```bash
bash scripts/eval_benchmarks.sh \
  --dataset mmvp \
  --model <model-checkpoint-path-or-hf-id> \
  --served-model-name <served-model-name> \
  --data-root <benchmark-data-root> \
  --output-dir results/<run-name> \
  --engine vllm_server \
  --vllm-server-base-url http://127.0.0.1:8000/v1 \
  --vllm-server-request-concurrency 512 \
  --eval-batch-size 512 \
  --vllm-server-timeout 1200 \
  --vllm-server-max-retries 0 \
  --enable-thinking \
  --no-resume
```

## TODO

- [x] Release training code
- [ ] Release data preparation scripts
- [x] Release evaluation scripts
- [x] Release model checkpoints
- [ ] Add detailed reproduction instructions

## License

This project is released under the Apache License 2.0.

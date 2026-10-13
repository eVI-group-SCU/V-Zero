import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.distillation.losses import distillation_loss
from verl.workers.config import ActorConfig
from verl.workers.config.distillation import DistillationEvidenceWeightingConfig

ROOT = Path(__file__).resolve().parents[2]


def test_actual_grouped_loss_with_global_batch_scaling(monkeypatch):
    # Exercise the real distillation reducer, with explicit per-token KL values.
    import verl.trainer.distillation.losses as module

    raw = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], requires_grad=True)
    monkeypatch.setattr(module, "get_distillation_loss_fn", lambda _: lambda **kwargs: (raw, {}))
    cfg = ActorConfig(strategy="fsdp", rollout_n=2, use_dynamic_bsz=True, loss_agg_mode="seq-mean-token-sum")
    cfg.global_batch_info.update(dp_size=2, global_batch_size=4)
    loss_cfg = SimpleNamespace(loss_mode="k3", loss_max_clamp=None, use_policy_gradient=False)
    distill_cfg = SimpleNamespace(
        distillation_loss=loss_cfg, evidence_weighting=SimpleNamespace(enabled=True, objective="va_opd")
    )
    c = torch.tensor([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    w = torch.tensor([1.5, 0.5])
    data = TensorDict({"response_mask": torch.ones(2, 3), "va_token_weights": c, "opd_weights": w}, [2])
    actual, _ = distillation_loss(cfg, distill_cfg, {}, data)
    expected = (
        (1.5 * (0.5 * raw[0, 0] + 0.5 * raw[0, 1:].mean()) + 0.5 * (0.5 * raw[1, 0] + 0.5 * raw[1, 1:].mean())) / 4 * 2
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, raw, retain_graph=True)[0], torch.autograd.grad(expected, raw)[0]
    )
    cfg = replace(cfg, loss_agg_mode="token-mean")
    with pytest.raises(ValueError, match="seq-mean-token-sum"):
        distillation_loss(cfg, distill_cfg, {}, data)


def test_evidence_config_preserves_legacy_and_validates_va():
    DistillationEvidenceWeightingConfig(enabled=True)
    DistillationEvidenceWeightingConfig(enabled=True, objective="va_opd", negative_mode="pixelate_teacher_image")
    with pytest.raises(ValueError, match="pixelate"):
        DistillationEvidenceWeightingConfig(enabled=True, objective="va_opd")


def test_launcher_builds_full_image_va_config(tmp_path):
    # Capture the exact Python argv without starting Ray or allocating GPUs.
    stub = tmp_path / "python3"
    out = tmp_path / "argv.json"
    stub.write_text("#!/usr/bin/env python3\n")
    # Avoid recursion through the PATH stub.
    import sys

    stub.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        'open(os.environ["ARGV_OUTPUT"], "w").write(json.dumps(sys.argv[1:]))\n'
    )
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        ARGV_OUTPUT=str(out),
        TRAIN_FILE="/tmp/train.parquet",
        VAL_FILE="/tmp/val.parquet",
    )
    for key in (
        "STUDENT_MODEL",
        "TEACHER_MODEL",
        "TRAIN_BATCH_SIZE",
        "PPO_MINI_BATCH_SIZE",
        "TOTAL_EPOCHS",
        "ROLLOUT_N",
        "VERL_REPO_ROOT",
    ):
        env.pop(key, None)
    subprocess.run(
        ["bash", str(ROOT / "scripts/run_va_opd_qwen3_vl_fsdp.sh")], env=env, check=True, capture_output=True, text=True
    )
    args = json.loads(out.read_text())
    overrides = dict(arg.lstrip("+").split("=", 1) for arg in args[3:])
    assert overrides["data.teacher_crop.enabled"] == "False"
    assert overrides["distillation.teacher_crop.enabled"] == "True"
    assert overrides["distillation.teacher_crop.prompt_key"] == "raw_prompt"
    assert overrides["distillation.evidence_weighting.objective"] == "va_opd"
    assert overrides["distillation.evidence_weighting.negative_mode"] == "pixelate_teacher_image"
    assert overrides["actor_rollout_ref.actor.loss_agg_mode"] == "seq-mean-token-sum"
    assert overrides["actor_rollout_ref.rollout.n"] == "4"
    assert overrides["data.train_batch_size"] == "16"
    assert overrides["trainer.total_epochs"] == "5"
    assert overrides["actor_rollout_ref.model.path"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert overrides["distillation.distillation_loss.loss_max_clamp"] == "null"
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=args[3:])
    assert composed.distillation.evidence_weighting.objective == "va_opd"
    assert not composed.data.teacher_crop.enabled
    assert composed.distillation.teacher_crop.prompt_key == "raw_prompt"
    # Every new evidence override must exist in the YAML schema.
    schema = OmegaConf.load(ROOT / "verl/trainer/config/distillation/distillation.yaml")
    for key, value in overrides.items():
        if key.startswith("distillation.evidence_weighting."):
            field = key.rsplit(".", 1)[-1]
            assert field in schema.evidence_weighting


def test_default_policy_gradient_uses_grouped_weights(monkeypatch):
    import verl.trainer.distillation.losses as module
    from verl.workers.config.distillation import DistillationLossConfig

    raw = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    monkeypatch.setattr(module, "get_distillation_loss_fn", lambda _: lambda **kwargs: (raw, {}))
    monkeypatch.setattr(module, "no_padding_2_padding", lambda tensor, _: tensor)
    cfg = ActorConfig(strategy="fsdp", rollout_n=2, use_dynamic_bsz=True, loss_agg_mode="seq-mean-token-sum")
    cfg.global_batch_info.update(dp_size=2, global_batch_size=4)
    dc = SimpleNamespace(
        distillation_loss=DistillationLossConfig(loss_mode="k1", use_policy_gradient=True, use_task_rewards=False),
        evidence_weighting=SimpleNamespace(enabled=True, objective="va_opd"),
    )
    c = torch.tensor([[0.5, 0.25, 0.25]] * 2)
    w = torch.tensor([1.5, 0.5])
    data = TensorDict(
        {
            "response_mask": torch.ones(2, 3),
            "va_token_weights": c,
            "opd_weights": w,
            "old_log_probs": torch.zeros(2, 3),
        },
        [2],
    )
    logp = torch.zeros(2, 3, requires_grad=True)
    actual, _ = distillation_loss(cfg, dc, {"log_probs": logp}, data)
    expected_coefficients = raw * c * w[:, None] / 4 * 2
    torch.testing.assert_close(actual, expected_coefficients.sum())
    torch.testing.assert_close(torch.autograd.grad(actual, logp)[0], expected_coefficients)

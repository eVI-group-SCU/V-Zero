# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def _squeeze_single_logprob(name: str, logprobs: torch.Tensor) -> torch.Tensor:
    if logprobs.dim() == 2:
        return logprobs
    if logprobs.dim() == 3 and logprobs.shape[-1] == 1:
        return logprobs.squeeze(-1)
    raise ValueError(
        f"{name} must contain sampled-token logprobs with shape (bsz, seq_len) or "
        f"(bsz, seq_len, 1), got {tuple(logprobs.shape)}."
    )


def compute_relative_evidence_opd_weights(
    *,
    teacher_logprobs: torch.Tensor,
    teacher_neg_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_width: int,
    uids: Sequence[Any] | np.ndarray,
    gamma: float,
    alpha: float,
    w_min: float,
    w_max: float,
    eps: float,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, float]] | tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Compute per-sequence relative evidence weights for OPD.

    `teacher_logprobs` and `teacher_neg_logprobs` are padded to the full
    prompt+response sequence. The returned weights have shape `(bsz,)`.
    """

    if len(uids) != response_mask.shape[0]:
        raise ValueError(f"uid count ({len(uids)}) must match batch size ({response_mask.shape[0]}).")

    plus_full = _squeeze_single_logprob("teacher_logprobs", teacher_logprobs).float()
    minus_full = _squeeze_single_logprob("teacher_neg_logprobs", teacher_neg_logprobs).float()
    response_len = response_mask.shape[1]
    response_end = prompt_width + response_len
    if plus_full.shape != minus_full.shape:
        raise ValueError(f"teacher positive/negative logprob shapes differ: {plus_full.shape} vs {minus_full.shape}.")
    if response_end > plus_full.shape[1]:
        raise ValueError(
            f"Teacher logprob sequence is too short for prompt_width+response_len: "
            f"{plus_full.shape[1]=}, {prompt_width=}, {response_len=}."
        )

    plus = plus_full[:, prompt_width:response_end]
    minus = minus_full[:, prompt_width:response_end]
    mask = response_mask.to(device=plus.device, dtype=plus.dtype)
    mask_bool = mask.bool()

    delta = plus - minus
    token_scores = torch.relu(delta) - float(gamma) * torch.relu(-delta)
    lengths = mask.sum(dim=-1).clamp_min(float(eps))
    scores = (token_scores * mask).sum(dim=-1) / lengths

    advantages = torch.zeros_like(scores)
    uid_values = np.asarray(uids, dtype=object)
    for uid in np.unique(uid_values):
        indices = np.nonzero(uid_values == uid)[0]
        if indices.size <= 1:
            continue
        group_idx = torch.as_tensor(indices, device=scores.device, dtype=torch.long)
        group_scores = scores.index_select(0, group_idx)
        group_mean = group_scores.mean()
        group_std = torch.sqrt(torch.mean((group_scores - group_mean).square()))
        advantages[group_idx] = (group_scores - group_mean) / (group_std + float(eps))

    weights = torch.clamp(1.0 + float(alpha) * advantages, min=float(w_min), max=float(w_max)).detach()

    valid_delta = delta[mask_bool]
    if valid_delta.numel() == 0:
        delta_mean = delta_pos_frac = 0.0
    else:
        delta_mean = valid_delta.mean().detach().item()
        delta_pos_frac = (valid_delta > 0).float().mean().detach().item()

    metrics = {
        "rew_opd/score_mean": scores.mean().detach().item(),
        "rew_opd/score_std": scores.std(unbiased=False).detach().item(),
        "rew_opd/adv_mean": advantages.mean().detach().item(),
        "rew_opd/adv_std": advantages.std(unbiased=False).detach().item(),
        "rew_opd/weight_mean": weights.mean().detach().item(),
        "rew_opd/weight_min": weights.min().detach().item(),
        "rew_opd/weight_max": weights.max().detach().item(),
        "rew_opd/weight_zero_frac": (weights <= float(w_min) + float(eps)).float().mean().detach().item(),
        "rew_opd/delta_mean": delta_mean,
        "rew_opd/delta_pos_frac": delta_pos_frac,
    }
    weights = weights.to(dtype=torch.float32)
    if return_diagnostics:
        diagnostics = {
            "scores": scores.detach().to(dtype=torch.float32),
            "advantages": advantages.detach().to(dtype=torch.float32),
            "weights": weights.detach(),
        }
        return weights, metrics, diagnostics
    return weights, metrics

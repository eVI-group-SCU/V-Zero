"""Visual-Advantage OPD, arXiv:2605.21924v1, equations (2)--(6).

Signals are computed over the complete sibling batch before DP/microbatch splits.
The rollout weights include K because the actor averages over sequences.
"""

import math
from collections import defaultdict

import torch
from PIL import Image

from verl.trainer.distillation.evidence_weighting import _squeeze_single_logprob


def pixelate_images(images, ratio=0.1):
    """Bilinear downsample followed by nearest upsample; preserve image size."""
    if not 0 < ratio < 1:
        raise ValueError("pixelation ratio must be between 0 and 1")
    images = list(images) if isinstance(images, (list, tuple)) else [images]
    if not images:
        raise ValueError("VA-OPD requires at least one image")
    result = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError("pixelation requires PIL images")
        width, height = image.size
        small = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        result.append(image.resize(small, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.NEAREST))
    return result


@torch.no_grad()
def compute_va_opd_weights(
    *,
    teacher_logprobs,
    teacher_neg_logprobs,
    response_mask,
    prompt_width,
    uids,
    temperature=1.0,
    high_va_fraction=0.2,
    high_va_weight=0.5,
    eps=1e-6,
    rollout_reweighting=True,
    token_grouping=True,
):
    """Return K*softmax(z/tau), within-group token coefficients, diagnostics.

    For fixed K, mean_sequences(sum_tokens(K*w*c*KL)) equals the paper's
    mean_prompts(sum_siblings(w*grouped_KL)). Reject unequal sibling counts.
    Top fraction uses ceil(p*T), bounded to leave one low token for T >= 2.
    A one-token response gets coefficient 1; an empty response contributes 0.
    Ties are broken by response position using a stable sort.
    """
    if temperature <= 0 or eps <= 0:
        raise ValueError("temperature and eps must be positive")
    if not 0 < high_va_fraction < 1 or not 0 < high_va_weight < 1:
        raise ValueError("VA fraction and weight must be between 0 and 1")
    plus = _squeeze_single_logprob("teacher_logprobs", teacher_logprobs).float()
    minus = _squeeze_single_logprob("teacher_neg_logprobs", teacher_neg_logprobs).float()
    if response_mask.ndim != 2 or len(uids) != response_mask.shape[0] or not len(uids):
        raise ValueError("expected a nonempty response batch and one uid per rollout")
    end = prompt_width + response_mask.shape[1]
    if plus.shape != minus.shape or plus.shape[0] != len(uids) or prompt_width < 0 or end > plus.shape[1]:
        raise ValueError("teacher outputs do not align with prompt and response")
    mask = response_mask.to(device=plus.device, dtype=torch.bool)
    plus, minus = plus[:, prompt_width:end], minus[:, prompt_width:end]
    if not torch.isfinite(plus[mask]).all() or not torch.isfinite(minus[mask]).all():
        raise ValueError("nonfinite teacher logprobs on valid response tokens")
    va = torch.where(mask, (plus - minus).relu(), 0.0)
    lengths = mask.sum(-1)
    scores = va.sum(-1) / lengths.clamp_min(1)
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    if len({len(indices) for indices in groups.values()}) != 1:
        raise ValueError("VA-OPD requires equal K sibling counts per prompt")
    weights = torch.ones_like(scores)
    advantages = torch.zeros_like(scores)
    for indices in groups.values():
        idx = torch.tensor(indices, device=scores.device)
        values = scores[idx]
        z = (values - values.mean()) / (values.std(unbiased=False) + eps)
        advantages[idx] = z
        if rollout_reweighting:
            weights[idx] = len(indices) * torch.softmax(z / temperature, dim=0)
    coefficients = torch.zeros_like(va)
    high_mask = torch.zeros_like(mask)
    for row in range(len(uids)):
        positions = mask[row].nonzero(as_tuple=True)[0]
        n = positions.numel()
        if not n:
            continue
        if not token_grouping or n == 1:
            coefficients[row, positions] = 1.0 / n
            continue
        n_high = min(n - 1, max(1, math.ceil(high_va_fraction * n)))
        order = torch.argsort(va[row, positions], descending=True, stable=True)
        high, low = positions[order[:n_high]], positions[order[n_high:]]
        high_mask[row, high] = True
        coefficients[row, high] = high_va_weight / n_high
        coefficients[row, low] = (1.0 - high_va_weight) / (n - n_high)
    metrics = {
        "va_opd/score_mean": scores.mean().item(),
        "va_opd/weight_min": weights.min().item(),
        "va_opd/weight_max": weights.max().item(),
        "va_opd/high_token_fraction": (high_mask.sum() / mask.sum().clamp_min(1)).item(),
        "va_opd/empty_response_fraction": (lengths == 0).float().mean().item(),
    }
    return weights, coefficients, metrics, {"scores": scores, "advantages": advantages, "high_mask": high_mask}

import numpy as np
import pytest
import torch
from PIL import Image

from verl.trainer.distillation.visual_advantage import compute_va_opd_weights, pixelate_images


def signals(delta, mask, uids, **kwargs):
    delta = torch.as_tensor(delta, dtype=torch.float32)
    # Prompt slots must never influence visual advantage.
    plus = torch.cat([torch.full((delta.shape[0], 2), 1000.0), delta - 10], 1).unsqueeze(-1)
    minus = torch.full_like(plus, -10)
    return compute_va_opd_weights(
        teacher_logprobs=plus,
        teacher_neg_logprobs=minus,
        response_mask=torch.as_tensor(mask),
        prompt_width=2,
        uids=uids,
        **kwargs,
    )


def test_paper_objective_and_gradient_with_interleaved_siblings():
    # Two prompts, two siblings each; manually evaluate equations (3)--(6).
    w, c, _, d = signals(
        [[4, 0, 0, 0, 0], [0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[1] * 5] * 4, ["a", "b", "a", "b"]
    )
    expected_a = torch.softmax(torch.tensor([1.0, -1.0]) / (1 + 1e-6 / 0.2), 0)
    torch.testing.assert_close(w[[0, 2]] / 2, expected_a)
    torch.testing.assert_close(w[[1, 3]], torch.ones(2))
    torch.testing.assert_close(c, torch.tensor([[0.5, 0.125, 0.125, 0.125, 0.125]] * 4))
    assert d["high_mask"][:, 0].all()  # deterministic ties
    loss = torch.arange(20, dtype=torch.float32).reshape(4, 5).requires_grad_()
    actor = (loss * c * w[:, None]).sum() / 4
    per_sequence = 0.5 * loss[:, 0] + 0.5 * loss[:, 1:].mean(-1)
    paper = (expected_a @ per_sequence[[0, 2]] + per_sequence[[1, 3]].mean()) / 2
    torch.testing.assert_close(actor, paper)
    torch.testing.assert_close(
        torch.autograd.grad(actor, loss, retain_graph=True)[0], torch.autograd.grad(paper, loss)[0]
    )
    assert not w.requires_grad and not c.requires_grad


def test_mask_negative_va_short_empty_and_nonfinite_padding():
    w, c, _, d = signals(
        [[1, -3, 8, float("nan")], [0, 4, 0, 0], [0, 0, 0, 0]], [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], ["x"] * 3
    )
    torch.testing.assert_close(d["scores"], torch.tensor([0.5, 4, 0]))
    torch.testing.assert_close(c, torch.tensor([[0.5, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]))
    assert torch.isfinite(w).all() and w.sum().item() == pytest.approx(3)


def test_reordering_and_microbatch_partition_preserve_objective():
    torch.manual_seed(1)
    delta = torch.rand(8, 17)
    mask = torch.arange(17)[None, :] < torch.tensor([17, 5, 12, 9, 2, 16, 8, 10])[:, None]
    uids = np.array(["a"] * 4 + ["b"] * 4)
    w, c, _, _ = signals(delta, mask, uids)
    order = torch.tensor([7, 1, 6, 3, 4, 0, 2, 5])
    wr, cr, _, _ = signals(delta[order], mask[order], uids[order])
    torch.testing.assert_close(wr, w[order])
    torch.testing.assert_close(cr, c[order])
    losses = torch.rand_like(delta)
    full = (losses * c * w[:, None]).sum() / 8
    pieces = sum((losses[idx] * c[idx] * w[idx, None]).sum() / 8 for idx in order.split(2))
    torch.testing.assert_close(full, pieces)


@pytest.mark.parametrize("rollout,token", [(False, True), (True, False), (False, False)])
def test_ablation_switches(rollout, token):
    w, c, _, _ = signals(
        [[3, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[1] * 5] * 2, ["a"] * 2, rollout_reweighting=rollout, token_grouping=token
    )
    if not rollout:
        torch.testing.assert_close(w, torch.ones(2))
    if not token:
        torch.testing.assert_close(c, torch.full((2, 5), 0.2))


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"high_va_fraction": 1}, {"high_va_weight": 0}, {"eps": 0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        signals([[1]], [[1]], ["x"], **kwargs)


def test_reject_unequal_siblings_and_nonfinite_response():
    with pytest.raises(ValueError, match="equal K"):
        signals([[1]] * 3, [[1]] * 3, ["a", "a", "b"])
    with pytest.raises(ValueError, match="nonfinite"):
        signals([[float("nan")]], [[1]], ["a"])


def test_pixelation_preserves_size_and_matches_paper_interpolation():
    arr = np.random.default_rng(7).integers(0, 256, (41, 73, 3), dtype=np.uint8)
    image = Image.fromarray(arr)
    result = pixelate_images([image])[0]
    expected = image.resize((7, 4), Image.Resampling.BILINEAR).resize((73, 41), Image.Resampling.NEAREST)
    assert result.size == image.size and result.mode == image.mode
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(image), arr)
    assert pixelate_images(Image.new("RGB", (1, 1)))[0].size == (1, 1)

"""Prediction-aligned teacher replay helpers, shared by vLLM and the actor.

vLLM extraction removes the first None and appends a final dummy. Row i
therefore scores input token i+1, matching causal model logits.
"""

import torch
from torch.nn import functional as F


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


def _align_teacher_outputs_to_student_sequence(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    student_prompt_length: int,
    student_response_length: int,
    teacher_prompt_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align teacher-crop replay outputs back to the student prompt/response layout.

    Inputs are teacher outputs for:
        teacher_crop_prompt + student_response

    Output tensors are shaped like:
        student_prompt_length - 1 placeholders + response predictions + final dummy

    Example:
        student_prompt_length = 3
        teacher_prompt_length = 5
        student_response_length = 2

        teacher input tokens:         T0 T1 T2 T3 T4 R0 R1
        teacher prediction rows:      T1 T2 T3 T4 R0 R1  P
        returned prediction rows:      P  P R0 R1  P

    The P positions are placeholders only. Downstream distillation uses the
    response positions, but the full tensor still has to match the student
    sequence layout. teacher_ids/logprobs may have a trailing top-k dimension,
    e.g. (seq_len, 1) or (seq_len, K), so we preserve shape[1:].
    """
    if student_prompt_length < 1 or teacher_prompt_length < 1 or student_response_length < 0:
        raise ValueError("Teacher replay requires nonempty prompts and a nonnegative response length.")
    expected_length = teacher_prompt_length + student_response_length
    if teacher_ids.shape[0] != expected_length or teacher_logprobs.shape[0] != expected_length:
        raise ValueError(
            f"Teacher output length must match teacher prompt plus student response, got "
            f"{teacher_ids.shape[0]=}, {teacher_logprobs.shape[0]=}, {expected_length=}."
        )

    response_start = teacher_prompt_length - 1
    response_end = response_start + student_response_length
    teacher_response_ids = teacher_ids[response_start:response_end]
    teacher_response_logprobs = teacher_logprobs[response_start:response_end]

    prompt_ids_pad = torch.full(
        (student_prompt_length - 1, *teacher_ids.shape[1:]),
        fill_value=pad_token_id,
        dtype=teacher_ids.dtype,
        device=teacher_ids.device,
    )
    prompt_logprobs_pad = torch.zeros(
        (student_prompt_length - 1, *teacher_logprobs.shape[1:]),
        dtype=teacher_logprobs.dtype,
        device=teacher_logprobs.device,
    )
    return (
        torch.cat([prompt_ids_pad, teacher_response_ids, torch.full_like(teacher_ids[:1], pad_token_id)], dim=0),
        torch.cat([prompt_logprobs_pad, teacher_response_logprobs, torch.zeros_like(teacher_logprobs[:1])], dim=0),
    )

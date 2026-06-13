# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn import functional as F

from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    # vLLM rejects non-default temperature for prompt_logprobs requests. Teacher
    # logprob replay is deterministic scoring, so it must not inherit rollout
    # generation sampling temperature.
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
    }


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
        student_prompt_length placeholder + student_response

    Example:
        student_prompt_length = 3
        teacher_prompt_length = 5
        student_response_length = 2

        teacher input/output layout:  T0 T1 T2 T3 T4 R0 R1
        returned aligned layout:      P  P  P  R0 R1

    The P positions are placeholders only. Downstream distillation uses the
    response positions, but the full tensor still has to match the student
    sequence layout. teacher_ids/logprobs may have a trailing top-k dimension,
    e.g. (seq_len, 1) or (seq_len, K), so we preserve shape[1:].
    """
    expected_length = teacher_prompt_length + student_response_length
    if teacher_ids.shape[0] != expected_length or teacher_logprobs.shape[0] != expected_length:
        raise ValueError(
            f"Teacher output length must match teacher prompt plus student response, got "
            f"{teacher_ids.shape[0]=}, {teacher_logprobs.shape[0]=}, {expected_length=}."
        )

    response_start = teacher_prompt_length
    response_end = teacher_prompt_length + student_response_length
    teacher_response_ids = teacher_ids[response_start:response_end]
    teacher_response_logprobs = teacher_logprobs[response_start:response_end]

    prompt_ids_pad = torch.full(
        (student_prompt_length, *teacher_ids.shape[1:]),
        fill_value=pad_token_id,
        dtype=teacher_ids.dtype,
        device=teacher_ids.device,
    )
    prompt_logprobs_pad = torch.zeros(
        (student_prompt_length, *teacher_logprobs.shape[1:]),
        dtype=teacher_logprobs.dtype,
        device=teacher_logprobs.device,
    )
    return (
        torch.cat([prompt_ids_pad, teacher_response_ids], dim=0),
        torch.cat([prompt_logprobs_pad, teacher_response_logprobs], dim=0),
    )


@dataclass
class TeacherLogprobRequest:
    """Unpadded per-sample teacher replay request plus final padding metadata."""

    sequence_ids: list[int]
    prompt_width: int
    response_width: int
    prompt_length: int
    response_length: int
    pad_token_id: int
    multi_modal_data: Optional[dict[str, Any]] = None
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    routing_key: Optional[str] = None
    student_prompt_length: Optional[int] = None
    student_response_length: Optional[int] = None
    teacher_prompt_length: Optional[int] = None
    negative_sequence_ids: Optional[list[int]] = None
    negative_multi_modal_data: Optional[dict[str, Any]] = None
    negative_mm_processor_kwargs: Optional[dict[str, Any]] = None
    negative_teacher_prompt_length: Optional[int] = None


@dataclass
class TeacherLogprobOutput:
    """Teacher logprob replay output for positive evidence and optional negative evidence."""

    teacher_ids: torch.Tensor
    teacher_logprobs: torch.Tensor
    teacher_neg_ids: Optional[torch.Tensor] = None
    teacher_neg_logprobs: Optional[torch.Tensor] = None


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def _score_teacher_sequence(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        student_prompt_length: Optional[int] = None,
        student_response_length: Optional[int] = None,
        teacher_prompt_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one unpadded teacher replay sequence."""
        alignment_args = (student_prompt_length, student_response_length, teacher_prompt_length, pad_token_id)
        if any(value is not None for value in alignment_args) and any(value is None for value in alignment_args):
            raise ValueError("Teacher-crop alignment requires complete alignment metadata.")

        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        if all(value is not None for value in alignment_args):
            teacher_ids, teacher_logprobs = _align_teacher_outputs_to_student_sequence(
                teacher_ids,
                teacher_logprobs,
                student_prompt_length=student_prompt_length,
                student_response_length=student_response_length,
                teacher_prompt_length=teacher_prompt_length,
                pad_token_id=pad_token_id,
            )
        return teacher_ids, teacher_logprobs

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        student_prompt_length: Optional[int] = None,
        student_response_length: Optional[int] = None,
        teacher_prompt_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        negative_sequence_ids: Optional[list[int]] = None,
        negative_multi_modal_data: Optional[dict[str, Any]] = None,
        negative_mm_processor_kwargs: Optional[dict[str, Any]] = None,
        negative_teacher_prompt_length: Optional[int] = None,
    ) -> TeacherLogprobOutput:
        """Compute teacher log probabilities for a single unpadded sequence."""
        if negative_sequence_ids is not None:
            if negative_teacher_prompt_length is None:
                raise ValueError("negative_teacher_prompt_length is required when negative_sequence_ids is set.")
            positive_output, negative_output = await asyncio.gather(
                self._score_teacher_sequence(
                    sequence_ids=sequence_ids,
                    multi_modal_data=multi_modal_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                    routing_key=routing_key,
                    student_prompt_length=student_prompt_length,
                    student_response_length=student_response_length,
                    teacher_prompt_length=teacher_prompt_length,
                    pad_token_id=pad_token_id,
                ),
                self._score_teacher_sequence(
                    sequence_ids=negative_sequence_ids,
                    multi_modal_data=negative_multi_modal_data,
                    mm_processor_kwargs=negative_mm_processor_kwargs,
                    routing_key=routing_key,
                    student_prompt_length=student_prompt_length,
                    student_response_length=student_response_length,
                    teacher_prompt_length=negative_teacher_prompt_length,
                    pad_token_id=pad_token_id,
                ),
            )
            teacher_ids, teacher_logprobs = positive_output
            teacher_neg_ids, teacher_neg_logprobs = negative_output
        else:
            teacher_ids, teacher_logprobs = await self._score_teacher_sequence(
                sequence_ids=sequence_ids,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                routing_key=routing_key,
                student_prompt_length=student_prompt_length,
                student_response_length=student_response_length,
                teacher_prompt_length=teacher_prompt_length,
                pad_token_id=pad_token_id,
            )
            teacher_neg_ids = None
            teacher_neg_logprobs = None

        return TeacherLogprobOutput(
            teacher_ids=teacher_ids,
            teacher_logprobs=teacher_logprobs,
            teacher_neg_ids=teacher_neg_ids,
            teacher_neg_logprobs=teacher_neg_logprobs,
        )

    async def compute_teacher_logprobs(
        self,
        requests: list[TeacherLogprobRequest],
    ) -> DataProto:
        """Compute teacher log probabilities for a batch of prepared replay requests."""
        if not requests:
            raise ValueError("Teacher logprob requests must be non-empty.")
        if any(request is None for request in requests):
            raise ValueError("Teacher logprob request is missing for at least one sample.")

        async def compute_one(request: TeacherLogprobRequest) -> TeacherLogprobOutput:
            return await self.compute_teacher_logprobs_single(
                sequence_ids=request.sequence_ids,
                multi_modal_data=request.multi_modal_data,
                mm_processor_kwargs=request.mm_processor_kwargs,
                routing_key=request.routing_key,
                student_prompt_length=request.student_prompt_length,
                student_response_length=request.student_response_length,
                teacher_prompt_length=request.teacher_prompt_length,
                pad_token_id=request.pad_token_id,
                negative_sequence_ids=request.negative_sequence_ids,
                negative_multi_modal_data=request.negative_multi_modal_data,
                negative_mm_processor_kwargs=request.negative_mm_processor_kwargs,
                negative_teacher_prompt_length=request.negative_teacher_prompt_length,
            )

        outputs = await asyncio.gather(*[compute_one(request) for request in requests])

        teacher_ids_list = []
        teacher_logprobs_list = []
        teacher_neg_ids_list = []
        teacher_neg_logprobs_list = []
        has_negative = any(output.teacher_neg_logprobs is not None for output in outputs)
        if has_negative and any(output.teacher_neg_logprobs is None or output.teacher_neg_ids is None for output in outputs):
            raise ValueError("Negative teacher ids/logprobs must be present for every teacher logprob output.")
        for request, output in zip(requests, outputs, strict=True):
            padded_teacher_ids, padded_teacher_logprobs = _pad_teacher_outputs(
                output.teacher_ids,
                output.teacher_logprobs,
                prompt_width=request.prompt_width,
                response_width=request.response_width,
                prompt_length=request.prompt_length,
                response_length=request.response_length,
                pad_token_id=request.pad_token_id,
            )
            teacher_ids_list.append(padded_teacher_ids)
            teacher_logprobs_list.append(padded_teacher_logprobs)
            if has_negative:
                padded_teacher_neg_ids, padded_teacher_neg_logprobs = _pad_teacher_outputs(
                    output.teacher_neg_ids,
                    output.teacher_neg_logprobs,
                    prompt_width=request.prompt_width,
                    response_width=request.response_width,
                    prompt_length=request.prompt_length,
                    response_length=request.response_length,
                    pad_token_id=request.pad_token_id,
                )
                teacher_neg_ids_list.append(padded_teacher_neg_ids)
                teacher_neg_logprobs_list.append(padded_teacher_neg_logprobs)

        batch_dict = {
            "teacher_ids": torch.cat(teacher_ids_list, dim=0),
            "teacher_logprobs": torch.cat(teacher_logprobs_list, dim=0),
        }
        if has_negative:
            batch_dict["teacher_neg_ids"] = torch.cat(teacher_neg_ids_list, dim=0)
            batch_dict["teacher_neg_logprobs"] = torch.cat(teacher_neg_logprobs_list, dim=0)

        batch = TensorDict(batch_dict, batch_size=len(requests))
        return DataProto(batch=batch)

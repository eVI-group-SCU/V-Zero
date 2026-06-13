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
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import contextlib
import copy
import functools
import logging
import math
import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.utils import resolve_config_path
from verl.experimental.teacher_loop.teacher_manager import TeacherLogprobOutput, TeacherLogprobRequest
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    model_config = ConfigDict(extra="allow")

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        teacher_ids, teacher_logprobs = (
            output["extra_fields"].pop("teacher_ids", None),
            output["extra_fields"].pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        teacher_neg_ids, teacher_neg_logprobs = (
            output["extra_fields"].pop("teacher_neg_ids", None),
            output["extra_fields"].pop("teacher_neg_logprobs", None),
        )
        if teacher_neg_ids is not None:
            output["teacher_neg_ids"] = teacher_neg_ids
        if teacher_neg_logprobs is not None:
            output["teacher_neg_logprobs"] = teacher_neg_logprobs
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    teacher_neg_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from negative teacher evidence."""
    teacher_neg_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the negative teacher log probabilities."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


@dataclass
class TeacherReplayContext:
    """Prepared teacher-only context for replaying a student response."""

    teacher_prompt_ids: list[int]
    teacher_multi_modal_data: dict[str, Any]
    teacher_mm_processor_kwargs: dict[str, Any]
    teacher_prompt_length: int
    pad_token_id: int
    negative_teacher_prompt_ids: Optional[list[int]] = None
    negative_teacher_multi_modal_data: Optional[dict[str, Any]] = None
    negative_teacher_mm_processor_kwargs: Optional[dict[str, Any]] = None
    negative_teacher_prompt_length: Optional[int] = None


@dataclass
class TeacherLogprobQueueItem:
    """Per-sample teacher logprob work item produced by an agent loop task."""

    sample_index: int
    request: TeacherLogprobRequest


@dataclass
class TeacherLogprobQueueResult:
    """Per-sample teacher logprob result consumed before final batch assembly."""

    teacher_ids: torch.Tensor
    teacher_logprobs: torch.Tensor
    teacher_neg_ids: Optional[torch.Tensor] = None
    teacher_neg_logprobs: Optional[torch.Tensor] = None
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class TeacherLogprobStreamHandle:
    """Trainer-side handle for streamed teacher logprob work owned by an AgentLoopWorker."""

    worker: Any
    stream_id: str
    length: int


@dataclass
class ValidationRewardQueueItem:
    """Per-sample validation reward work item produced by an agent loop task."""

    sample_index: int
    data: Optional[DataProto]
    response_length: int
    reward_score: Optional[float] = None
    reward_extra_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRewardQueueResult:
    """Per-sample validation reward result consumed before final metrics assembly."""

    reward_score: float
    reward_extra_info: dict[str, Any] = field(default_factory=dict)
    response_length: int = 0


@dataclass
class ValidationRewardStreamHandle:
    """Trainer-side handle for streamed validation reward work owned by an AgentLoopWorker."""

    worker: Any
    stream_id: str
    length: int


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        processing_class = self.processor if self.processor is not None else self.tokenizer
        self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.teacher_crop_config = config.distillation.get("teacher_crop", {}) or {}
        self.teacher_crop_enabled = self.teacher_crop_config.get("enabled", False)
        self.teacher_crop_prompt_key = self.teacher_crop_config.get("prompt_key", "teacher_raw_prompt")
        self.teacher_crop_image_key = self.teacher_crop_config.get("image_key", "teacher_images")
        self.evidence_weighting_config = config.distillation.get("evidence_weighting", {}) or {}
        self.evidence_weighting_enabled = self.evidence_weighting_config.get("enabled", False)
        self.evidence_negative_mode = self.evidence_weighting_config.get("negative_mode", "downsample_teacher_crop")
        self.evidence_negative_image_key = self.evidence_weighting_config.get("negative_image_key", "teacher_neg_images")
        self.evidence_negative_prompt_key = self.evidence_weighting_config.get(
            "negative_prompt_key", "teacher_neg_raw_prompt"
        )
        self.evidence_downsample_factor = int(self.evidence_weighting_config.get("downsample_factor", 2))
        self.teacher_crop_async_prepare_config = self.teacher_crop_config.get("async_prepare", {}) or {}
        self.teacher_crop_async_prepare_enabled = (
            self.teacher_crop_enabled and self.teacher_crop_async_prepare_config.get("enabled", False)
        )
        self.teacher_crop_async_prepare_total_concurrency = int(
            self.teacher_crop_async_prepare_config.get("concurrency", 8)
        )
        if self.teacher_crop_async_prepare_total_concurrency <= 0:
            raise ValueError("distillation.teacher_crop.async_prepare.concurrency must be positive.")
        agent_num_workers = int(getattr(self.rollout_config.agent, "num_workers", 1) or 1)
        self.teacher_crop_async_prepare_concurrency = min(
            self.teacher_crop_async_prepare_total_concurrency,
            max(1, math.ceil(self.teacher_crop_async_prepare_total_concurrency / agent_num_workers)),
        )
        self.teacher_crop_prepare_semaphore = (
            asyncio.Semaphore(self.teacher_crop_async_prepare_concurrency)
            if self.teacher_crop_async_prepare_enabled
            else None
        )
        self.teacher_crop_prepare_executor = (
            ThreadPoolExecutor(
                max_workers=self.teacher_crop_async_prepare_concurrency,
                thread_name_prefix="teacher-crop-prepare",
            )
            if self.teacher_crop_async_prepare_enabled
            else None
        )
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        self._pending_teacher_logprob_streams: dict[str, asyncio.Task[DataProto]] = {}
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    def _unwrap_sample_value(self, value: Any) -> Any:
        # Normalize batched numpy values before per-sample validation.
        if isinstance(value, np.ndarray):
            return value.item() if value.shape == () else value.tolist()
        if hasattr(value, "item") and not isinstance(value, str | bytes | dict | list | tuple):
            return value.item()
        return value

    def _require_teacher_crop_field(self, sample_kwargs: Optional[dict[str, Any]], field_name: str) -> Any:
        # Fail immediately when teacher-crop data is not present in the sample.
        if sample_kwargs is None or field_name not in sample_kwargs:
            raise ValueError(f"teacher_crop.enabled=True but sample is missing {field_name!r}")
        value = self._unwrap_sample_value(sample_kwargs[field_name])
        if value is None:
            raise ValueError(f"teacher_crop.enabled=True but sample field {field_name!r} is None")
        if hasattr(value, "__len__") and not isinstance(value, str | bytes) and len(value) == 0:
            raise ValueError(f"teacher_crop.enabled=True but sample field {field_name!r} is empty")
        return value

    async def _extract_teacher_crop_multi_modal_data(self, teacher_messages: list[dict]) -> dict[str, Any]:
        # Extract the teacher-only crop inputs from the teacher prompt.
        multi_modal_data = {}
        if self.processor is None:
            return multi_modal_data

        image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
        if hasattr(self.dataset_cls, "process_multi_modal_info"):
            images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                teacher_messages,
                image_patch_size=image_patch_size,
                config=self.config.data,
            )
        else:
            images, videos = await self.dataset_cls.process_vision_info(
                teacher_messages,
                image_patch_size=image_patch_size,
                config=self.config.data,
            )
            audios = None
        if images is not None:
            multi_modal_data["images"] = images
        if videos is not None:
            multi_modal_data["videos"] = videos
        if audios is not None:
            multi_modal_data["audios"] = audios
        return multi_modal_data

    def _extract_teacher_crop_multi_modal_data_sync(self, teacher_messages: list[dict]) -> dict[str, Any]:
        # Synchronous fast path used by the teacher-crop prepare thread pool.
        multi_modal_data = {}
        if self.processor is None:
            return multi_modal_data
        if not hasattr(self.dataset_cls, "_process_multi_modal_info"):
            raise NotImplementedError(
                f"{self.dataset_cls.__name__} does not expose a synchronous _process_multi_modal_info method"
            )

        image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
        images, videos, audios = self.dataset_cls._process_multi_modal_info(
            teacher_messages,
            image_patch_size=image_patch_size,
            config=self.config.data,
        )
        if images is not None:
            multi_modal_data["images"] = images
        if videos is not None:
            multi_modal_data["videos"] = videos
        if audios is not None:
            multi_modal_data["audios"] = audios
        return multi_modal_data

    def _downsample_teacher_images(self, images: Any) -> list[Image.Image]:
        if self.evidence_negative_mode != "downsample_teacher_crop":
            raise ValueError(f"Unsupported evidence negative mode: {self.evidence_negative_mode!r}.")
        image_list = list(images) if isinstance(images, (list, tuple)) else [images]
        if not image_list:
            raise ValueError("Evidence weighting requires at least one teacher crop image.")

        resample_filter = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        downsampled_images = []
        for image in image_list:
            if not isinstance(image, Image.Image):
                raise TypeError(f"Evidence downsampling expects PIL.Image inputs, got {type(image)}.")
            width, height = image.size
            new_size = (
                max(1, width // self.evidence_downsample_factor),
                max(1, height // self.evidence_downsample_factor),
            )
            downsampled_images.append(image.resize(new_size, resample=resample_filter))
        return downsampled_images

    def _coerce_teacher_image_list(self, value: Any, field_name: str) -> list[Any]:
        value = self._unwrap_sample_value(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        image_list = list(value) if isinstance(value, (list, tuple)) else [value]
        if not image_list:
            raise ValueError(f"Evidence weighting requires non-empty {field_name!r}.")
        return image_list

    def _build_teacher_image_content(self, image: Any, template_content: Optional[dict[str, Any]] = None) -> dict:
        template_content = template_content or {}
        image = self._unwrap_sample_value(image)

        if isinstance(image, Image.Image):
            payload = {"image": image if image.mode == "RGB" else image.convert("RGB")}
        elif isinstance(image, dict):
            payload = dict(image)
            if "bytes" in payload and "image" not in payload:
                payload["image"] = Image.open(BytesIO(payload["bytes"]))
        elif isinstance(image, str):
            payload = {"image": image}
        else:
            raise TypeError(
                f"Evidence negative image must be a dict, path string, or PIL.Image, got {type(image)}."
            )

        payload.pop("type", None)
        content = {"type": "image", **payload}
        for bound_key in ("min_pixels", "max_pixels"):
            if bound_key in template_content and bound_key not in content:
                content[bound_key] = template_content[bound_key]
        return content

    def _replace_teacher_crop_images(
        self,
        teacher_messages: list[dict],
        negative_images: list[Any],
    ) -> list[dict]:
        negative_messages = copy.deepcopy(teacher_messages)
        image_slots: list[tuple[list[Any], int, dict[str, Any]]] = []
        for message in negative_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for index, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image":
                    image_slots.append((content, index, part))

        if len(image_slots) < len(negative_images):
            raise ValueError(
                f"Evidence weighting got {len(negative_images)} negative image(s) from "
                f"{self.evidence_negative_image_key!r}, but the teacher prompt has only "
                f"{len(image_slots)} image placeholder(s)."
            )

        # The teacher prompt is built from <image> first and <crop> after it, so
        # replacing the trailing image slots swaps the crop evidence while
        # preserving the original full-image context.
        for negative_image, (content, index, template_content) in zip(
            negative_images,
            image_slots[-len(negative_images) :],
            strict=True,
        ):
            content[index] = self._build_teacher_image_content(negative_image, template_content)
        return negative_messages

    def _get_hard_negative_teacher_messages(
        self,
        sample_kwargs: Optional[dict[str, Any]],
        teacher_messages: list[dict],
    ) -> list[dict]:
        negative_prompt_key = self.evidence_negative_prompt_key
        if sample_kwargs is not None and negative_prompt_key in sample_kwargs:
            negative_messages = self._unwrap_sample_value(sample_kwargs[negative_prompt_key])
            if negative_messages is None:
                raise ValueError(f"Evidence weighting sample field {negative_prompt_key!r} is None")
            return list(negative_messages)

        negative_images = self._require_teacher_crop_field(sample_kwargs, self.evidence_negative_image_key)
        negative_images = self._coerce_teacher_image_list(negative_images, self.evidence_negative_image_key)
        return self._replace_teacher_crop_images(teacher_messages, negative_images)

    def _build_negative_teacher_multi_modal_data(self, teacher_multi_modal_data: dict[str, Any]) -> dict[str, Any]:
        negative_multi_modal_data = dict(teacher_multi_modal_data)
        images = teacher_multi_modal_data.get("images")
        if images is None:
            raise ValueError("Evidence weighting requires teacher crop images for negative evidence.")
        negative_multi_modal_data["images"] = self._downsample_teacher_images(images)
        return negative_multi_modal_data

    def _prepare_hard_negative_teacher_context_sync(
        self,
        sample_kwargs: Optional[dict[str, Any]],
        teacher_messages: list[dict],
    ) -> tuple[list[int], dict[str, Any], dict[str, Any], int]:
        negative_teacher_messages = self._get_hard_negative_teacher_messages(sample_kwargs, teacher_messages)
        negative_teacher_multi_modal_data = self._extract_teacher_crop_multi_modal_data_sync(negative_teacher_messages)
        if not negative_teacher_multi_modal_data.get("images"):
            raise ValueError(
                f"Evidence weighting hard_negative_crop but {self.evidence_negative_image_key!r} "
                "produced no negative teacher images"
            )
        negative_teacher_mm_processor_kwargs = self._get_mm_processor_kwargs(
            negative_teacher_multi_modal_data.get("audios")
        )
        negative_teacher_prompt_ids = self._tokenize_teacher_crop_prompt_sync(
            negative_teacher_messages,
            negative_teacher_multi_modal_data,
            negative_teacher_mm_processor_kwargs,
        )
        return (
            negative_teacher_prompt_ids,
            negative_teacher_multi_modal_data,
            negative_teacher_mm_processor_kwargs,
            len(negative_teacher_prompt_ids),
        )

    async def _prepare_hard_negative_teacher_context(
        self,
        sample_kwargs: Optional[dict[str, Any]],
        teacher_messages: list[dict],
    ) -> tuple[list[int], dict[str, Any], dict[str, Any], int]:
        negative_teacher_messages = self._get_hard_negative_teacher_messages(sample_kwargs, teacher_messages)
        negative_teacher_multi_modal_data = await self._extract_teacher_crop_multi_modal_data(negative_teacher_messages)
        if not negative_teacher_multi_modal_data.get("images"):
            raise ValueError(
                f"Evidence weighting hard_negative_crop but {self.evidence_negative_image_key!r} "
                "produced no negative teacher images"
            )
        negative_teacher_mm_processor_kwargs = self._get_mm_processor_kwargs(
            negative_teacher_multi_modal_data.get("audios")
        )
        negative_teacher_prompt_ids = await self._tokenize_teacher_crop_prompt(
            negative_teacher_messages,
            negative_teacher_multi_modal_data,
            negative_teacher_mm_processor_kwargs,
        )
        return (
            negative_teacher_prompt_ids,
            negative_teacher_multi_modal_data,
            negative_teacher_mm_processor_kwargs,
            len(negative_teacher_prompt_ids),
        )

    def _tokenize_teacher_crop_prompt_sync(
        self,
        teacher_messages: list[dict],
        teacher_multi_modal_data: dict[str, Any],
        mm_processor_kwargs: dict[str, Any],
    ) -> list[int]:
        # Tokenize the teacher prompt without changing the student rollout prompt.
        images = teacher_multi_modal_data.get("images")
        videos = teacher_multi_modal_data.get("videos")
        audios = teacher_multi_modal_data.get("audios")

        if self.processor is not None:
            raw_prompt = apply_chat_template(
                self.processor,
                teacher_messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.config.data.get("apply_chat_template_kwargs", {}),
            )
            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            return normalize_token_ids(model_inputs.pop("input_ids"))

        tokenized_prompt = apply_chat_template(
            self.tokenizer,
            teacher_messages,
            add_generation_prompt=True,
            tokenize=True,
            **self.config.data.get("apply_chat_template_kwargs", {}),
        )
        return normalize_token_ids(tokenized_prompt)

    async def _tokenize_teacher_crop_prompt(
        self,
        teacher_messages: list[dict],
        teacher_multi_modal_data: dict[str, Any],
        mm_processor_kwargs: dict[str, Any],
    ) -> list[int]:
        if not getattr(self, "teacher_crop_async_prepare_enabled", False):
            return self._tokenize_teacher_crop_prompt_sync(
                teacher_messages,
                teacher_multi_modal_data,
                mm_processor_kwargs,
            )
        executor = getattr(self, "teacher_crop_prepare_executor", None)
        return await get_event_loop().run_in_executor(
            executor,
            functools.partial(
                self._tokenize_teacher_crop_prompt_sync,
                teacher_messages,
                teacher_multi_modal_data,
                mm_processor_kwargs,
            ),
        )

    async def _prepare_teacher_crop_context(self, sample_kwargs: Optional[dict[str, Any]]) -> TeacherReplayContext:
        semaphore = getattr(self, "teacher_crop_prepare_semaphore", None)
        if semaphore is None:
            return await self._prepare_teacher_crop_context_unlimited(sample_kwargs)
        async with semaphore:
            if hasattr(self.dataset_cls, "_process_multi_modal_info"):
                return await get_event_loop().run_in_executor(
                    self.teacher_crop_prepare_executor,
                    functools.partial(self._prepare_teacher_crop_context_sync, sample_kwargs),
                )
            return await self._prepare_teacher_crop_context_unlimited(sample_kwargs)

    def _prepare_teacher_crop_context_sync(self, sample_kwargs: Optional[dict[str, Any]]) -> TeacherReplayContext:
        teacher_messages = self._require_teacher_crop_field(sample_kwargs, self.teacher_crop_prompt_key)
        teacher_messages = list(teacher_messages)
        teacher_multi_modal_data = self._extract_teacher_crop_multi_modal_data_sync(teacher_messages)
        if not teacher_multi_modal_data.get("images"):
            raise ValueError(
                f"teacher_crop.enabled=True but {self.teacher_crop_image_key!r} produced no teacher images"
            )

        teacher_mm_processor_kwargs = self._get_mm_processor_kwargs(teacher_multi_modal_data.get("audios"))
        teacher_prompt_ids = self._tokenize_teacher_crop_prompt_sync(
            teacher_messages,
            teacher_multi_modal_data,
            teacher_mm_processor_kwargs,
        )
        negative_teacher_multi_modal_data = None
        negative_teacher_mm_processor_kwargs = None
        negative_teacher_prompt_ids = None
        negative_teacher_prompt_length = None
        if self.evidence_weighting_enabled:
            if self.evidence_negative_mode == "downsample_teacher_crop":
                negative_teacher_multi_modal_data = self._build_negative_teacher_multi_modal_data(
                    teacher_multi_modal_data
                )
                negative_teacher_mm_processor_kwargs = self._get_mm_processor_kwargs(
                    negative_teacher_multi_modal_data.get("audios")
                )
                negative_teacher_prompt_ids = self._tokenize_teacher_crop_prompt_sync(
                    teacher_messages,
                    negative_teacher_multi_modal_data,
                    negative_teacher_mm_processor_kwargs,
                )
                negative_teacher_prompt_length = len(negative_teacher_prompt_ids)
            elif self.evidence_negative_mode == "hard_negative_crop":
                (
                    negative_teacher_prompt_ids,
                    negative_teacher_multi_modal_data,
                    negative_teacher_mm_processor_kwargs,
                    negative_teacher_prompt_length,
                ) = self._prepare_hard_negative_teacher_context_sync(sample_kwargs, teacher_messages)
            else:
                raise ValueError(f"Unsupported evidence negative mode: {self.evidence_negative_mode!r}.")

        return TeacherReplayContext(
            teacher_prompt_ids=teacher_prompt_ids,
            teacher_multi_modal_data=teacher_multi_modal_data,
            teacher_mm_processor_kwargs=teacher_mm_processor_kwargs,
            teacher_prompt_length=len(teacher_prompt_ids),
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            negative_teacher_prompt_ids=negative_teacher_prompt_ids,
            negative_teacher_multi_modal_data=negative_teacher_multi_modal_data,
            negative_teacher_mm_processor_kwargs=negative_teacher_mm_processor_kwargs,
            negative_teacher_prompt_length=negative_teacher_prompt_length,
        )

    async def _prepare_teacher_crop_context_unlimited(
        self, sample_kwargs: Optional[dict[str, Any]]
    ) -> TeacherReplayContext:
        teacher_messages = self._require_teacher_crop_field(sample_kwargs, self.teacher_crop_prompt_key)
        teacher_messages = list(teacher_messages)
        teacher_multi_modal_data = await self._extract_teacher_crop_multi_modal_data(teacher_messages)
        if not teacher_multi_modal_data.get("images"):
            raise ValueError(
                f"teacher_crop.enabled=True but {self.teacher_crop_image_key!r} produced no teacher images"
            )

        teacher_mm_processor_kwargs = self._get_mm_processor_kwargs(teacher_multi_modal_data.get("audios"))
        teacher_prompt_ids = await self._tokenize_teacher_crop_prompt(
            teacher_messages,
            teacher_multi_modal_data,
            teacher_mm_processor_kwargs,
        )
        negative_teacher_multi_modal_data = None
        negative_teacher_mm_processor_kwargs = None
        negative_teacher_prompt_ids = None
        negative_teacher_prompt_length = None
        if self.evidence_weighting_enabled:
            if self.evidence_negative_mode == "downsample_teacher_crop":
                negative_teacher_multi_modal_data = self._build_negative_teacher_multi_modal_data(
                    teacher_multi_modal_data
                )
                negative_teacher_mm_processor_kwargs = self._get_mm_processor_kwargs(
                    negative_teacher_multi_modal_data.get("audios")
                )
                negative_teacher_prompt_ids = await self._tokenize_teacher_crop_prompt(
                    teacher_messages,
                    negative_teacher_multi_modal_data,
                    negative_teacher_mm_processor_kwargs,
                )
                negative_teacher_prompt_length = len(negative_teacher_prompt_ids)
            elif self.evidence_negative_mode == "hard_negative_crop":
                (
                    negative_teacher_prompt_ids,
                    negative_teacher_multi_modal_data,
                    negative_teacher_mm_processor_kwargs,
                    negative_teacher_prompt_length,
                ) = await self._prepare_hard_negative_teacher_context(sample_kwargs, teacher_messages)
            else:
                raise ValueError(f"Unsupported evidence negative mode: {self.evidence_negative_mode!r}.")

        return TeacherReplayContext(
            teacher_prompt_ids=teacher_prompt_ids,
            teacher_multi_modal_data=teacher_multi_modal_data,
            teacher_mm_processor_kwargs=teacher_mm_processor_kwargs,
            teacher_prompt_length=len(teacher_prompt_ids),
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            negative_teacher_prompt_ids=negative_teacher_prompt_ids,
            negative_teacher_multi_modal_data=negative_teacher_multi_modal_data,
            negative_teacher_mm_processor_kwargs=negative_teacher_mm_processor_kwargs,
            negative_teacher_prompt_length=negative_teacher_prompt_length,
        )

    def _maybe_start_teacher_crop_prepare_task(
        self,
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]],
    ) -> Optional[asyncio.Task]:
        if not (
            self.distillation_enabled
            and self.teacher_crop_enabled
            and getattr(self, "teacher_crop_async_prepare_enabled", False)
            and not validate
        ):
            return None
        return asyncio.create_task(self._prepare_teacher_crop_context(sample_kwargs))

    @staticmethod
    def _normalize_teacher_crop_cache_key(value: Any) -> Any:
        if hasattr(value, "item"):
            value = value.item()
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    async def _cancel_teacher_crop_prepare_task(self, task: Optional[asyncio.Task]) -> None:
        if task is None:
            return
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()

    def _record_agent_loop_timing(self, output: AgentLoopOutput, timing: dict[str, float]) -> None:
        metrics = output.metrics
        for key, value in timing.items():
            if hasattr(metrics, key):
                setattr(metrics, key, float(value))
            elif isinstance(metrics, dict):
                metrics[key] = float(value)
            else:
                setattr(metrics, key, float(value))

    def _should_stream_teacher_logprobs(self, validate: bool, global_steps: int) -> bool:
        if validate or not self.distillation_enabled:
            return False
        if getattr(self, "teacher_server_manager", None) is None:
            return False
        critic_warmup = int(self.config.trainer.get("critic_warmup", 0))
        return global_steps >= critic_warmup and self._teacher_logprob_streaming_mode() != "trainer"

    def _teacher_logprob_streaming_mode(self) -> str:
        configured = os.getenv("VERL_TEACHER_LOGPROB_STREAMING_MODE", "nonblocking").strip().lower()
        aliases = {
            "1": "nonblocking",
            "true": "nonblocking",
            "yes": "nonblocking",
            "on": "nonblocking",
            "async": "nonblocking",
            "agent": "nonblocking",
            "agent_nonblocking": "nonblocking",
            "0": "trainer",
            "false": "trainer",
            "no": "trainer",
            "off": "trainer",
            "none": "trainer",
            "disable": "trainer",
            "disabled": "trainer",
            "sync": "blocking",
            "agent_blocking": "blocking",
        }
        mode = aliases.get(configured, configured)
        if mode not in {"nonblocking", "blocking", "trainer"}:
            raise ValueError(
                "VERL_TEACHER_LOGPROB_STREAMING_MODE must be one of "
                "nonblocking, blocking, or trainer."
            )
        return mode

    def _teacher_logprob_streaming_concurrency(self, batch_size: int) -> int:
        if batch_size <= 0:
            return 0
        configured = os.getenv("VERL_TEACHER_LOGPROB_STREAMING_CONCURRENCY")
        if configured is not None:
            concurrency = int(configured)
            if concurrency <= 0:
                raise ValueError("VERL_TEACHER_LOGPROB_STREAMING_CONCURRENCY must be positive.")
            return min(batch_size, concurrency)
        return min(batch_size, 8)

    async def _compute_teacher_logprob_request(
        self,
        request: TeacherLogprobRequest,
    ) -> tuple[TeacherLogprobOutput, dict[str, float]]:
        timing = {}
        with simple_timer("teacher_logprob_vllm", timing):
            teacher_output = await self.teacher_server_manager.compute_teacher_logprobs_single(
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
        return teacher_output, timing

    async def _consume_teacher_logprob_requests(
        self,
        queue: asyncio.Queue,
        results: list[Optional[TeacherLogprobQueueResult]],
    ) -> None:
        from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

        while True:
            item = await queue.get()
            if item is None:
                return
            teacher_output, timing = await self._compute_teacher_logprob_request(item.request)
            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_output.teacher_ids,
                teacher_output.teacher_logprobs,
                prompt_width=item.request.prompt_width,
                response_width=item.request.response_width,
                prompt_length=item.request.prompt_length,
                response_length=item.request.response_length,
                pad_token_id=item.request.pad_token_id,
            )
            teacher_neg_ids = None
            teacher_neg_logprobs = None
            if teacher_output.teacher_neg_ids is not None and teacher_output.teacher_neg_logprobs is not None:
                teacher_neg_ids, teacher_neg_logprobs = _pad_teacher_outputs(
                    teacher_output.teacher_neg_ids,
                    teacher_output.teacher_neg_logprobs,
                    prompt_width=item.request.prompt_width,
                    response_width=item.request.response_width,
                    prompt_length=item.request.prompt_length,
                    response_length=item.request.response_length,
                    pad_token_id=item.request.pad_token_id,
                )
            results[item.sample_index] = TeacherLogprobQueueResult(
                teacher_ids=teacher_ids,
                teacher_logprobs=teacher_logprobs,
                teacher_neg_ids=teacher_neg_ids,
                teacher_neg_logprobs=teacher_neg_logprobs,
                timing=timing,
            )

    def _attach_streamed_teacher_logprob_results(
        self,
        outputs: list[_InternalAgentLoopOutput],
        results: list[Optional[TeacherLogprobQueueResult]],
    ) -> None:
        if len(outputs) != len(results):
            raise ValueError(f"Teacher logprob result count mismatch: {len(outputs)=}, {len(results)=}.")
        missing = [idx for idx, result in enumerate(results) if result is None]
        if missing:
            raise ValueError(f"Missing streamed teacher logprob results for sample indices: {missing}.")
        for output, result in zip(outputs, results, strict=True):
            output.teacher_ids = result.teacher_ids
            output.teacher_logprobs = result.teacher_logprobs
            output.teacher_neg_ids = result.teacher_neg_ids
            output.teacher_neg_logprobs = result.teacher_neg_logprobs
            self._record_agent_loop_timing(output, result.timing)

    async def _finalize_teacher_logprob_stream(
        self,
        queue: asyncio.Queue,
        consumers: list[asyncio.Task],
        results: list[Optional[TeacherLogprobQueueResult]],
    ) -> DataProto:
        try:
            for _ in consumers:
                await queue.put(None)
            await asyncio.gather(*consumers)
        except BaseException:
            for consumer in consumers:
                consumer.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.gather(*consumers, return_exceptions=True)
            raise

        missing = [idx for idx, result in enumerate(results) if result is None]
        if missing:
            raise ValueError(f"Missing streamed teacher logprob results for sample indices: {missing}.")

        teacher_ids = torch.cat([result.teacher_ids for result in results if result is not None], dim=0)
        teacher_logprobs = torch.cat([result.teacher_logprobs for result in results if result is not None], dim=0)
        batch_dict = {
            "teacher_ids": teacher_ids,
            "teacher_logprobs": teacher_logprobs,
        }
        has_negative = any(result.teacher_neg_logprobs is not None for result in results if result is not None)
        if has_negative:
            if any(
                result.teacher_neg_ids is None or result.teacher_neg_logprobs is None
                for result in results
                if result is not None
            ):
                raise ValueError("Negative streamed teacher ids/logprobs must be present for every result.")
            batch_dict["teacher_neg_ids"] = torch.cat(
                [result.teacher_neg_ids for result in results if result is not None], dim=0
            )
            batch_dict["teacher_neg_logprobs"] = torch.cat(
                [result.teacher_neg_logprobs for result in results if result is not None], dim=0
            )
        batch = TensorDict(batch_dict, batch_size=teacher_ids.shape[0])
        return DataProto(batch=batch)

    def _log_teacher_logprob_stream_failure(self, stream_id: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        with contextlib.suppress(Exception):
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Teacher logprob stream %s failed.",
                    stream_id,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    async def get_teacher_logprob_stream(self, stream_id: str) -> DataProto:
        pending_streams = getattr(self, "_pending_teacher_logprob_streams", None)
        if pending_streams is None or stream_id not in pending_streams:
            raise KeyError(f"Unknown teacher logprob stream id: {stream_id}")
        task = pending_streams.pop(stream_id)
        return await task

    def _should_stream_validation_rewards(self, validate: bool) -> bool:
        if not validate or self.reward_loop_worker_handles is None:
            return False
        value = os.getenv("VERL_VALIDATION_REWARD_STREAMING", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _validation_reward_streaming_mode(self) -> str:
        mode = os.getenv("VERL_VALIDATION_REWARD_STREAMING_MODE", "nonblocking").strip().lower()
        if mode not in {"blocking", "nonblocking"}:
            raise ValueError(
                "VERL_VALIDATION_REWARD_STREAMING_MODE must be 'blocking' or 'nonblocking', "
                f"got {mode!r}."
            )
        return mode

    @staticmethod
    def _validation_reward_streaming_concurrency(num_samples: int) -> int:
        raw = os.getenv("VERL_VALIDATION_REWARD_STREAMING_CONCURRENCY")
        if raw is None:
            return max(1, min(num_samples, 32))
        concurrency = int(raw)
        if concurrency <= 0:
            raise ValueError("VERL_VALIDATION_REWARD_STREAMING_CONCURRENCY must be positive.")
        return min(concurrency, max(1, num_samples))

    async def _compute_score_data(self, data: DataProto) -> dict:
        if self.reward_loop_worker_handles is None:
            raise RuntimeError("Validation reward streaming requires reward loop worker handles.")
        selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
        return await selected_reward_loop_worker_handle.compute_score.remote(data)

    async def _consume_validation_reward_requests(
        self,
        queue: asyncio.Queue,
        results: list[Optional[ValidationRewardQueueResult]],
    ) -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            if item.data is None:
                results[item.sample_index] = ValidationRewardQueueResult(
                    reward_score=float(item.reward_score),
                    reward_extra_info=dict(item.reward_extra_info),
                    response_length=item.response_length,
                )
                continue
            result = await self._compute_score_data(item.data)
            results[item.sample_index] = ValidationRewardQueueResult(
                reward_score=float(result["reward_score"]),
                reward_extra_info=dict(result.get("reward_extra_info", {})),
                response_length=item.response_length,
            )

    def _attach_streamed_validation_reward_results(
        self,
        outputs: list[AgentLoopOutput],
        results: list[Optional[ValidationRewardQueueResult]],
    ) -> None:
        if len(outputs) != len(results):
            raise ValueError(f"Validation reward result count mismatch: {len(outputs)=}, {len(results)=}.")
        missing = [idx for idx, result in enumerate(results) if result is None]
        if missing:
            raise ValueError(f"Missing streamed validation reward results for sample indices: {missing}.")
        for output, result in zip(outputs, results, strict=True):
            output.reward_score = result.reward_score
            output.extra_fields["reward_extra_info"] = result.reward_extra_info

    def _validation_reward_results_to_dataproto(
        self,
        results: list[Optional[ValidationRewardQueueResult]],
    ) -> DataProto:
        missing = [idx for idx, result in enumerate(results) if result is None]
        if missing:
            raise ValueError(f"Missing streamed validation reward results for sample indices: {missing}.")

        response_width = self.rollout_config.response_length
        rm_scores = torch.zeros((len(results), response_width), dtype=torch.float32)
        reward_extra_infos = []
        for idx, result in enumerate(results):
            assert result is not None
            response_length = max(1, min(int(result.response_length), response_width))
            rm_scores[idx, response_length - 1] = float(result.reward_score)
            reward_extra_infos.append(result.reward_extra_info)

        reward_extra_keys = sorted({key for info in reward_extra_infos for key in info})
        non_tensor_batch = {
            key: np.array([info.get(key) for info in reward_extra_infos], dtype=object) for key in reward_extra_keys
        }
        return DataProto(
            batch=TensorDict({"rm_scores": rm_scores}, batch_size=len(results)),
            non_tensor_batch=non_tensor_batch,
            meta_info={"reward_extra_keys": reward_extra_keys},
        )

    async def _finalize_validation_reward_stream(
        self,
        queue: asyncio.Queue,
        consumers: list[asyncio.Task],
        results: list[Optional[ValidationRewardQueueResult]],
    ) -> DataProto:
        try:
            for _ in consumers:
                await queue.put(None)
            await asyncio.gather(*consumers)
        except BaseException:
            for consumer in consumers:
                consumer.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.gather(*consumers, return_exceptions=True)
            raise
        return self._validation_reward_results_to_dataproto(results)

    def _log_validation_reward_stream_failure(self, stream_id: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        with contextlib.suppress(Exception):
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Validation reward stream %s failed.",
                    stream_id,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    async def get_validation_reward_stream(self, stream_id: str) -> DataProto:
        pending_streams = getattr(self, "_pending_validation_reward_streams", None)
        if pending_streams is None or stream_id not in pending_streams:
            raise KeyError(f"Unknown validation reward stream id: {stream_id}")
        task = pending_streams.pop(stream_id)
        return await task

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            presence_penalty=config.presence_penalty,
            repetition_penalty=config.repetition_penalty,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
            sampling_params["presence_penalty"] = config.val_kwargs.presence_penalty
            sampling_params["repetition_penalty"] = config.val_kwargs.repetition_penalty

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0
            params["presence_penalty"] = 0.0
            params["repetition_penalty"] = 1.0

        if validate:
            if not config.val_kwargs.do_sample:
                apply_greedy_sampling_params(sampling_params)
        elif not config.do_sample:
            apply_greedy_sampling_params(sampling_params)

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        teacher_logprob_queue = None
        teacher_logprob_results = None
        teacher_logprob_consumers = []
        teacher_logprob_streaming_mode = "trainer"
        if self._should_stream_teacher_logprobs(validate, batch.meta_info.get("global_steps", -1)):
            teacher_logprob_streaming_mode = self._teacher_logprob_streaming_mode()
            teacher_logprob_queue = asyncio.Queue()
            teacher_logprob_results = [None] * len(batch)
            concurrency = self._teacher_logprob_streaming_concurrency(len(batch))
            teacher_logprob_consumers = [
                asyncio.create_task(
                    self._consume_teacher_logprob_requests(teacher_logprob_queue, teacher_logprob_results)
                )
                for _ in range(concurrency)
            ]

        validation_reward_queue = None
        validation_reward_results = None
        validation_reward_consumers = []
        validation_reward_streaming_mode = "blocking"
        if self._should_stream_validation_rewards(validate):
            validation_reward_streaming_mode = self._validation_reward_streaming_mode()
            validation_reward_queue = asyncio.Queue()
            validation_reward_results = [None] * len(batch)
            concurrency = self._validation_reward_streaming_concurrency(len(batch))
            validation_reward_consumers = [
                asyncio.create_task(
                    self._consume_validation_reward_requests(validation_reward_queue, validation_reward_results)
                )
                for _ in range(concurrency)
            ]

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        shared_teacher_context_tasks: dict[Any, Optional[asyncio.Task]] = {}
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            teacher_context_task = None
            if self.distillation_enabled and self.teacher_crop_enabled:
                cache_value = kwargs.get("uid", index[i])
                cache_key = self._normalize_teacher_crop_cache_key(cache_value)
                if cache_key not in shared_teacher_context_tasks:
                    shared_teacher_context_tasks[cache_key] = self._maybe_start_teacher_crop_prepare_task(
                        validate,
                        kwargs,
                    )
                teacher_context_task = shared_teacher_context_tasks[cache_key]
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(
                        sample_sampling_params,
                        trajectory_info[i],
                        trace=trace_this_sample,
                        teacher_logprob_queue=teacher_logprob_queue,
                        validation_reward_queue=validation_reward_queue,
                        sample_position=i,
                        teacher_context_task=teacher_context_task,
                        owns_teacher_context_task=False,
                        **kwargs,
                    )
                )
            )

        try:
            outputs = await asyncio.gather(*tasks)
            teacher_logprob_stream_id = None
            validation_reward_stream_id = None
            if teacher_logprob_queue is not None:
                if teacher_logprob_streaming_mode == "blocking":
                    for _ in teacher_logprob_consumers:
                        await teacher_logprob_queue.put(None)
                    await asyncio.gather(*teacher_logprob_consumers)
                    self._attach_streamed_teacher_logprob_results(outputs, teacher_logprob_results)
                else:
                    teacher_logprob_stream_id = uuid4().hex
                    finalize_task = asyncio.create_task(
                        self._finalize_teacher_logprob_stream(
                            teacher_logprob_queue,
                            teacher_logprob_consumers,
                            teacher_logprob_results,
                        )
                    )
                    pending_streams = getattr(self, "_pending_teacher_logprob_streams", None)
                    if pending_streams is None:
                        self._pending_teacher_logprob_streams = {}
                        pending_streams = self._pending_teacher_logprob_streams
                    pending_streams[teacher_logprob_stream_id] = finalize_task
                    finalize_task.add_done_callback(
                        functools.partial(self._log_teacher_logprob_stream_failure, teacher_logprob_stream_id)
                    )
            if validation_reward_queue is not None:
                if validation_reward_streaming_mode == "blocking":
                    for _ in validation_reward_consumers:
                        await validation_reward_queue.put(None)
                    await asyncio.gather(*validation_reward_consumers)
                    self._attach_streamed_validation_reward_results(outputs, validation_reward_results)
                else:
                    validation_reward_stream_id = uuid4().hex
                    finalize_task = asyncio.create_task(
                        self._finalize_validation_reward_stream(
                            validation_reward_queue,
                            validation_reward_consumers,
                            validation_reward_results,
                        )
                    )
                    pending_streams = getattr(self, "_pending_validation_reward_streams", None)
                    if pending_streams is None:
                        self._pending_validation_reward_streams = {}
                        pending_streams = self._pending_validation_reward_streams
                    pending_streams[validation_reward_stream_id] = finalize_task
                    finalize_task.add_done_callback(
                        functools.partial(self._log_validation_reward_stream_failure, validation_reward_stream_id)
                    )
        except BaseException:
            for task in tasks:
                task.cancel()
            for teacher_context_task in shared_teacher_context_tasks.values():
                await self._cancel_teacher_crop_prepare_task(teacher_context_task)
            for consumer in teacher_logprob_consumers:
                consumer.cancel()
            for consumer in validation_reward_consumers:
                consumer.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.gather(*teacher_logprob_consumers, return_exceptions=True)
            with contextlib.suppress(Exception):
                await asyncio.gather(*validation_reward_consumers, return_exceptions=True)
            raise

        output = self._postprocess(
            outputs,
            input_non_tensor_batch=batch.non_tensor_batch,
            validate=batch.meta_info.get("validate", False),
        )
        if teacher_logprob_stream_id is not None:
            output.meta_info["teacher_logprob_stream_id"] = teacher_logprob_stream_id
        if validation_reward_stream_id is not None:
            output.meta_info["validation_reward_stream_id"] = validation_reward_stream_id
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        teacher_logprob_queue: Optional[asyncio.Queue] = None,
        validation_reward_queue: Optional[asyncio.Queue] = None,
        sample_position: Optional[int] = None,
        teacher_context_task: Optional[asyncio.Task] = None,
        owns_teacher_context_task: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            if teacher_context_task is None:
                teacher_context_task = self._maybe_start_teacher_crop_prepare_task(trajectory["validate"], kwargs)
                owns_teacher_context_task = True
            try:
                output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
                processed = await self._agent_loop_postprocess(
                    output,
                    trajectory["validate"],
                    teacher_context_task=teacher_context_task,
                    teacher_logprob_queue=teacher_logprob_queue,
                    validation_reward_queue=validation_reward_queue,
                    sample_position=sample_position,
                    **kwargs,
                )
                return processed
            except BaseException:
                if owns_teacher_context_task:
                    await self._cancel_teacher_crop_prepare_task(teacher_context_task)
                raise

    async def _agent_loop_postprocess(
        self,
        output,
        validate,
        teacher_context_task: Optional[asyncio.Task] = None,
        teacher_logprob_queue: Optional[asyncio.Queue] = None,
        validation_reward_queue: Optional[asyncio.Queue] = None,
        sample_position: Optional[int] = None,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, "
                    f"total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )

        if validation_reward_queue is not None:
            if sample_position is None:
                raise ValueError("Streaming validation reward requires sample_position.")
            if output.reward_score is None:
                reward_data = self._prepare_score_data([output], kwargs=kwargs)
                reward_score = None
                reward_extra_info = {}
            else:
                reward_data = None
                reward_score = output.reward_score
                reward_extra_info = dict(output.extra_fields.get("reward_extra_info") or {})
                output.reward_score = None
                output.extra_fields.pop("reward_extra_info", None)
            await validation_reward_queue.put(
                ValidationRewardQueueItem(
                    sample_index=sample_position,
                    data=reward_data,
                    response_length=len(output.response_ids),
                    reward_score=reward_score,
                    reward_extra_info=reward_extra_info,
                )
            )
        else:
            await self._compute_score([output], kwargs=kwargs)
        teacher_logprob_request = None
        if self.distillation_enabled and not validate:
            teacher_logprob_request = await self._prepare_teacher_logprob_request(
                output,
                prompt_ids=output.prompt_ids,
                response_ids=output.response_ids,
                validate=validate,
                sample_kwargs=kwargs,
                teacher_context_task=teacher_context_task,
                prompt_width=self.rollout_config.prompt_length,
                response_width=self.rollout_config.response_length,
            )
        if teacher_logprob_queue is not None and teacher_logprob_request is not None:
            if sample_position is None:
                raise ValueError("Streaming teacher logprob requires sample_position.")
            output.extra_fields.pop("teacher_logprob_request", None)
            await teacher_logprob_queue.put(
                TeacherLogprobQueueItem(sample_index=sample_position, request=teacher_logprob_request)
            )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        teacher_neg_ids, teacher_neg_logprobs = (
            output.extra_fields.pop("teacher_neg_ids", None),
            output.extra_fields.pop("teacher_neg_logprobs", None),
        )
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if teacher_neg_ids is not None and teacher_neg_logprobs is not None:
                teacher_neg_ids, teacher_neg_logprobs = _pad_teacher_outputs(
                    teacher_neg_ids,
                    teacher_neg_logprobs,
                    prompt_width=prompt_output["input_ids"].shape[1],
                    response_width=response_output["input_ids"].shape[1],
                    prompt_length=len(output.prompt_ids),
                    response_length=len(output.response_ids),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        processed = _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            teacher_neg_logprobs=teacher_neg_logprobs,
            teacher_neg_ids=teacher_neg_ids,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )
        return processed

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    def _prepare_score_data(self, outputs: list[AgentLoopOutput], kwargs: dict) -> DataProto:
        all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
        for output in outputs:
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                multi_modal_inputs,
                output.mm_processor_kwargs
                if output.mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(
                    output.multi_modal_data.get("audios") if output.multi_modal_data else None
                ),
            ).squeeze(0)
            all_prompts.append(prompts)
            all_responses.append(responses)
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)

        n = len(outputs)
        batch = TensorDict(
            {
                "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    all_attention_mask, batch_first=True, padding_value=0
                ),
                "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                "position_ids": torch.nn.utils.rnn.pad_sequence(
                    all_position_ids, batch_first=True, padding_value=0
                ),
            },
            batch_size=n,
        )
        non_tensor_batch = {
            **{k: np.array([v] * n) for k, v in kwargs.items()},
            "__num_turns__": np.array([o.num_turns for o in outputs]),
            "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
            "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
            "response_len": np.array([len(o.response_ids) for o in outputs]),
        }

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            result = await self._compute_score_data(self._prepare_score_data(outputs, kwargs=kwargs))
            final_output.reward_score = result["reward_score"]
            final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

    async def _prepare_teacher_logprob_request(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
        teacher_context_task: Optional[asyncio.Task] = None,
        prompt_width: Optional[int] = None,
        response_width: Optional[int] = None,
    ) -> Optional[TeacherLogprobRequest]:
        """Prepare teacher logprob request for asynchronous trainer-side execution."""
        if not (self.distillation_enabled and not validate):
            return None

        timing = {
            "teacher_prepare_wait": 0.0,
            "teacher_prepare_tokenize_processor": 0.0,
        }
        with simple_timer("teacher_prepare_total", timing):
            if prompt_width is None or response_width is None:
                raise ValueError("Teacher logprob request requires prompt_width and response_width.")
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
            multi_modal_data = output.multi_modal_data
            mm_processor_kwargs = output.mm_processor_kwargs
            sequence_ids = prompt_ids + response_ids
            teacher_prompt_length = None
            student_prompt_length = None
            student_response_length = None
            negative_sequence_ids = None
            negative_multi_modal_data = None
            negative_mm_processor_kwargs = None
            negative_teacher_prompt_length = None
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            if self.teacher_crop_enabled:
                if teacher_context_task is not None:
                    with simple_timer("teacher_prepare_wait", timing):
                        teacher_context = await teacher_context_task
                else:
                    with simple_timer("teacher_prepare_tokenize_processor", timing):
                        teacher_context = await self._prepare_teacher_crop_context(sample_kwargs)
                sequence_ids = teacher_context.teacher_prompt_ids + response_ids
                multi_modal_data = teacher_context.teacher_multi_modal_data
                mm_processor_kwargs = teacher_context.teacher_mm_processor_kwargs
                teacher_prompt_length = teacher_context.teacher_prompt_length
                student_prompt_length = len(prompt_ids)
                student_response_length = len(response_ids)
                pad_token_id = teacher_context.pad_token_id
                if teacher_context.negative_teacher_prompt_ids is not None:
                    negative_sequence_ids = teacher_context.negative_teacher_prompt_ids + response_ids
                    negative_multi_modal_data = teacher_context.negative_teacher_multi_modal_data
                    negative_mm_processor_kwargs = teacher_context.negative_teacher_mm_processor_kwargs
                    negative_teacher_prompt_length = teacher_context.negative_teacher_prompt_length
            request = TeacherLogprobRequest(
                sequence_ids=sequence_ids,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                routing_key=routing_key,
                student_prompt_length=student_prompt_length,
                student_response_length=student_response_length,
                teacher_prompt_length=teacher_prompt_length,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=len(prompt_ids),
                response_length=len(response_ids),
                pad_token_id=pad_token_id,
                negative_sequence_ids=negative_sequence_ids,
                negative_multi_modal_data=negative_multi_modal_data,
                negative_mm_processor_kwargs=negative_mm_processor_kwargs,
                negative_teacher_prompt_length=negative_teacher_prompt_length,
            )
        output.extra_fields["teacher_logprob_request"] = request
        self._record_agent_loop_timing(output, timing)
        return request

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
        teacher_context_task: Optional[asyncio.Task] = None,
        prompt_width: Optional[int] = None,
        response_width: Optional[int] = None,
    ) -> None:
        """Compute teacher logprobs immediately for a single sample."""
        request = await self._prepare_teacher_logprob_request(
            output=output,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            validate=validate,
            sample_kwargs=sample_kwargs,
            teacher_context_task=teacher_context_task,
            prompt_width=prompt_width if prompt_width is not None else len(prompt_ids),
            response_width=response_width if response_width is not None else len(response_ids),
        )
        if request is None:
            return
        output.extra_fields.pop("teacher_logprob_request", None)
        teacher_output, timing = await self._compute_teacher_logprob_request(request)
        output.extra_fields["teacher_ids"] = teacher_output.teacher_ids
        output.extra_fields["teacher_logprobs"] = teacher_output.teacher_logprobs
        if teacher_output.teacher_neg_ids is not None and teacher_output.teacher_neg_logprobs is not None:
            output.extra_fields["teacher_neg_ids"] = teacher_output.teacher_neg_ids
            output.extra_fields["teacher_neg_logprobs"] = teacher_output.teacher_neg_logprobs
        self._record_agent_loop_timing(output, timing)

    def _postprocess(
        self,
        inputs: list[AgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Pad raw agent-loop outputs at worker-chunk granularity and combine them into a batch."""
        if not inputs:
            raise ValueError("Agent loop postprocess requires at least one output.")

        if isinstance(inputs[0].prompt_ids, torch.Tensor):
            prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
            response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
            response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
            attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
            input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
            position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
            optional_outputs = {}
            if inputs[0].response_logprobs is not None:
                optional_outputs["rollout_log_probs"] = torch.cat(
                    [input.response_logprobs for input in inputs], dim=0
                )
            if inputs[0].routed_experts is not None:
                optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
            if inputs[0].teacher_logprobs is not None and inputs[0].teacher_ids is not None:
                optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
                optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)
            if inputs[0].teacher_neg_logprobs is not None and inputs[0].teacher_neg_ids is not None:
                optional_outputs["teacher_neg_logprobs"] = torch.cat(
                    [input.teacher_neg_logprobs for input in inputs], dim=0
                )
                optional_outputs["teacher_neg_ids"] = torch.cat([input.teacher_neg_ids for input in inputs], dim=0)
            batch = TensorDict(
                {
                    "prompts": prompt_ids,
                    "responses": response_ids,
                    "response_mask": response_mask,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    **optional_outputs,
                },
                batch_size=len(inputs),
            )

            scores = [input.reward_score for input in inputs]
            if all(score is not None for score in scores):
                prompt_length = prompt_ids.size(1)
                response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
                rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
                rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(
                    scores, dtype=torch.float32
                )
                batch["rm_scores"] = rm_scores

            non_tensor_batch = {
                "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
            }
            if self.reward_loop_worker_handles is None and input_non_tensor_batch:
                non_tensor_batch.update(input_non_tensor_batch)

            reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
            reward_extra_keys = list(reward_extra_infos[0].keys())
            for key in reward_extra_keys:
                non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

            multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
            if any(mmi is not None for mmi in multi_modal_inputs_list):
                non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

            metrics = [input.metrics.model_dump() for input in inputs]
            extra_fields = {}
            default_extra_keys = {
                "turn_scores",
                "tool_rewards",
                "min_global_steps",
                "max_global_steps",
                "extras",
            }
            all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
            for key in all_keys:
                temp_arr = np.empty(len(inputs), dtype=object)
                temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
                extra_fields[key] = temp_arr

            non_tensor_batch.update(extra_fields)

            if "rm_scores" in batch.keys():
                meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
            else:
                meta_info = {"metrics": metrics}

            return DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                meta_info=meta_info,
            )

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        def as_token_list(values: Any) -> list[int]:
            if hasattr(values, "tolist"):
                values = values.tolist()
            return [int(value.item() if hasattr(value, "item") else value) for value in values]

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids are left padded; response_ids and response_mask are right padded.
        batch_size = len(inputs)
        prompt_width = self.rollout_config.prompt_length
        response_width = self.rollout_config.response_length
        total_width = prompt_width + response_width
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        prompt_ids = torch.full((batch_size, prompt_width), pad_token_id, dtype=torch.long)
        prompt_attention_mask = torch.zeros((batch_size, prompt_width), dtype=torch.long)
        response_ids = torch.full((batch_size, response_width), pad_token_id, dtype=torch.long)
        response_attention_mask = torch.zeros((batch_size, response_width), dtype=torch.long)
        response_mask = torch.zeros((batch_size, response_width), dtype=torch.long)

        prompt_lengths = []
        response_lengths = []
        for idx, output in enumerate(inputs):
            prompt = as_token_list(output.prompt_ids)
            response = as_token_list(output.response_ids)
            output_response_mask = as_token_list(output.response_mask)
            if len(prompt) > prompt_width:
                raise ValueError(f"Prompt length {len(prompt)} exceeds configured prompt width {prompt_width}.")
            if len(response) > response_width:
                raise ValueError(f"Response length {len(response)} exceeds configured response width {response_width}.")

            prompt_lengths.append(len(prompt))
            response_lengths.append(len(response))
            prompt_start = prompt_width - len(prompt)
            if prompt:
                prompt_ids[idx, prompt_start:] = torch.tensor(prompt, dtype=torch.long)
                prompt_attention_mask[idx, prompt_start:] = 1
            if response:
                response_ids[idx, : len(response)] = torch.tensor(response, dtype=torch.long)
                response_attention_mask[idx, : len(response)] = 1
            if output_response_mask:
                mask_width = min(len(output_response_mask), response_width)
                response_mask[idx, :mask_width] = torch.tensor(output_response_mask[:mask_width], dtype=torch.long)

        response_mask = response_mask * response_attention_mask
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)

        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            rollout_log_probs = torch.zeros((batch_size, response_width), dtype=torch.float32)
            for idx, output in enumerate(inputs):
                if output.response_logprobs is None:
                    raise ValueError("Missing response_logprobs for at least one agent-loop output.")
                values = (
                    output.response_logprobs.tolist()
                    if hasattr(output.response_logprobs, "tolist")
                    else output.response_logprobs
                )
                values = [float(value.item() if hasattr(value, "item") else value) for value in values]
                logprob_width = min(len(values), response_width)
                if logprob_width:
                    rollout_log_probs[idx, :logprob_width] = torch.tensor(values[:logprob_width], dtype=torch.float32)
            optional_outputs["rollout_log_probs"] = rollout_log_probs

        routed_outputs = [output.routed_experts for output in inputs]
        if any(routed is not None for routed in routed_outputs):
            if any(routed is None for routed in routed_outputs):
                raise ValueError("Missing routed_experts for at least one agent-loop output.")
            first_routed = routed_outputs[0]
            if isinstance(first_routed, np.ndarray):
                first_routed_array = first_routed.copy() if not first_routed.flags.writeable else first_routed
                first_routed = torch.from_numpy(first_routed_array)
            if not isinstance(first_routed, torch.Tensor):
                raise TypeError(f"Unsupported type for routed_experts: {type(first_routed)}")
            _, layer_num, topk_num = first_routed.shape
            routed_experts = torch.zeros(batch_size, total_width, layer_num, topk_num, dtype=first_routed.dtype)
            for idx, routed in enumerate(routed_outputs):
                if isinstance(routed, np.ndarray):
                    routed = torch.from_numpy(routed.copy() if not routed.flags.writeable else routed)
                elif not isinstance(routed, torch.Tensor):
                    raise TypeError(f"Unsupported type for routed_experts: {type(routed)}")
                length = routed.shape[0]
                start_pos = prompt_width - prompt_lengths[idx]
                end_pos = min(start_pos + length, total_width)
                if start_pos < 0 or end_pos > total_width:
                    raise ValueError(
                        f"Invalid routed_experts position range: {start_pos=}, {end_pos=}, {total_width=}."
                    )
                routed_experts[idx, start_pos:end_pos] = routed[: end_pos - start_pos]
            optional_outputs["routed_experts"] = routed_experts

        teacher_pairs = [
            (output.extra_fields.pop("teacher_ids", None), output.extra_fields.pop("teacher_logprobs", None))
            for output in inputs
        ]
        has_teacher_outputs = any(
            teacher_ids is not None or teacher_logprobs is not None for teacher_ids, teacher_logprobs in teacher_pairs
        )
        if has_teacher_outputs:
            if any(teacher_ids is None or teacher_logprobs is None for teacher_ids, teacher_logprobs in teacher_pairs):
                raise ValueError("Teacher ids/logprobs must be present together for every agent-loop output.")
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            padded_teacher_ids = []
            padded_teacher_logprobs = []
            for idx, (teacher_ids, teacher_logprobs) in enumerate(teacher_pairs):
                if not isinstance(teacher_ids, torch.Tensor):
                    teacher_ids = torch.tensor(teacher_ids, dtype=torch.int32)
                if not isinstance(teacher_logprobs, torch.Tensor):
                    teacher_logprobs = torch.tensor(teacher_logprobs, dtype=torch.float32)
                if teacher_ids.dim() == 3:
                    padded_ids, padded_logprobs = teacher_ids, teacher_logprobs
                else:
                    padded_ids, padded_logprobs = _pad_teacher_outputs(
                        teacher_ids,
                        teacher_logprobs,
                        prompt_width=prompt_width,
                        response_width=response_width,
                        prompt_length=prompt_lengths[idx],
                        response_length=response_lengths[idx],
                        pad_token_id=pad_token_id,
                    )
                padded_teacher_ids.append(padded_ids)
                padded_teacher_logprobs.append(padded_logprobs)
            optional_outputs["teacher_ids"] = torch.cat(padded_teacher_ids, dim=0)
            optional_outputs["teacher_logprobs"] = torch.cat(padded_teacher_logprobs, dim=0)

        teacher_neg_pairs = [
            (output.extra_fields.pop("teacher_neg_ids", None), output.extra_fields.pop("teacher_neg_logprobs", None))
            for output in inputs
        ]
        has_teacher_neg_outputs = any(
            teacher_ids is not None or teacher_logprobs is not None
            for teacher_ids, teacher_logprobs in teacher_neg_pairs
        )
        if has_teacher_neg_outputs:
            if any(
                teacher_ids is None or teacher_logprobs is None
                for teacher_ids, teacher_logprobs in teacher_neg_pairs
            ):
                raise ValueError("Negative teacher ids/logprobs must be present together for every agent-loop output.")
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            padded_teacher_neg_ids = []
            padded_teacher_neg_logprobs = []
            for idx, (teacher_ids, teacher_logprobs) in enumerate(teacher_neg_pairs):
                if not isinstance(teacher_ids, torch.Tensor):
                    teacher_ids = torch.tensor(teacher_ids, dtype=torch.int32)
                if not isinstance(teacher_logprobs, torch.Tensor):
                    teacher_logprobs = torch.tensor(teacher_logprobs, dtype=torch.float32)
                if teacher_ids.dim() == 3:
                    padded_ids, padded_logprobs = teacher_ids, teacher_logprobs
                else:
                    padded_ids, padded_logprobs = _pad_teacher_outputs(
                        teacher_ids,
                        teacher_logprobs,
                        prompt_width=prompt_width,
                        response_width=response_width,
                        prompt_length=prompt_lengths[idx],
                        response_length=response_lengths[idx],
                        pad_token_id=pad_token_id,
                    )
                padded_teacher_neg_ids.append(padded_ids)
                padded_teacher_neg_logprobs.append(padded_logprobs)
            optional_outputs["teacher_neg_ids"] = torch.cat(padded_teacher_neg_ids, dim=0)
            optional_outputs["teacher_neg_logprobs"] = torch.cat(padded_teacher_neg_logprobs, dim=0)

        if self.processor is None:
            position_ids = compute_position_id_with_mask(attention_mask)
            multi_modal_inputs_list = [{} for _ in inputs]
        else:
            multi_modal_inputs_list = []
            position_ids_list = []
            for idx, output in enumerate(inputs):
                sample_input_ids = input_ids[idx : idx + 1]
                sample_attention_mask = attention_mask[idx : idx + 1]
                multi_modal_inputs = self._compute_multi_modal_inputs(output, sample_input_ids)
                position_ids_list.append(
                    self._compute_position_ids(
                        sample_input_ids,
                        sample_attention_mask,
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    )
                )
                multi_modal_inputs_list.append(multi_modal_inputs)
            position_ids = torch.cat(position_ids_list, dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        teacher_stream_handles = []
        validation_reward_stream_handles = []
        for worker, worker_output in zip(self.agent_loop_workers, outputs, strict=True):
            stream_id = worker_output.meta_info.pop("teacher_logprob_stream_id", None)
            if stream_id is not None:
                teacher_stream_handles.append(
                    TeacherLogprobStreamHandle(worker=worker, stream_id=stream_id, length=len(worker_output))
                )
            stream_id = worker_output.meta_info.pop("validation_reward_stream_id", None)
            if stream_id is not None:
                validation_reward_stream_handles.append(
                    ValidationRewardStreamHandle(worker=worker, stream_id=stream_id, length=len(worker_output))
                )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        if teacher_stream_handles:
            output.meta_info["teacher_logprob_stream_handles"] = teacher_stream_handles
            output.non_tensor_batch["teacher_logprob_stream_index"] = np.arange(len(output), dtype=np.int64)
        if validation_reward_stream_handles:
            output.meta_info["validation_reward_stream_handles"] = validation_reward_stream_handles
            output.non_tensor_batch["validation_reward_stream_index"] = np.arange(len(output), dtype=np.int64)
        return output

    @auto_await
    async def resolve_teacher_logprob_streams(
        self,
        stream_handles: list[TeacherLogprobStreamHandle],
        order_indices: Optional[np.ndarray] = None,
    ) -> DataProto:
        if not stream_handles:
            raise ValueError("Teacher logprob stream handles must be non-empty.")

        outputs = await asyncio.gather(
            *[handle.worker.get_teacher_logprob_stream.remote(handle.stream_id) for handle in stream_handles]
        )
        teacher_output = DataProto.concat(outputs)
        expected_length = sum(handle.length for handle in stream_handles)
        if len(teacher_output) != expected_length:
            raise ValueError(
                f"Resolved teacher logprob stream length mismatch: {len(teacher_output)=}, {expected_length=}."
            )
        if order_indices is not None:
            order = torch.as_tensor(np.asarray(order_indices, dtype=np.int64), dtype=torch.long)
            teacher_output.reorder(order)
        return teacher_output

    @auto_await
    async def resolve_validation_reward_streams(
        self,
        stream_handles: list[ValidationRewardStreamHandle],
        order_indices: Optional[np.ndarray] = None,
    ) -> DataProto:
        if not stream_handles:
            raise ValueError("Validation reward stream handles must be non-empty.")

        outputs = await asyncio.gather(
            *[handle.worker.get_validation_reward_stream.remote(handle.stream_id) for handle in stream_handles]
        )
        reward_extra_keys = sorted({key for output in outputs for key in output.meta_info.get("reward_extra_keys", [])})
        for output in outputs:
            for key in reward_extra_keys:
                if key not in output.non_tensor_batch:
                    output.non_tensor_batch[key] = np.array([None] * len(output), dtype=object)
            output.meta_info["reward_extra_keys"] = reward_extra_keys
        reward_output = DataProto.concat(outputs)
        expected_length = sum(handle.length for handle in stream_handles)
        if len(reward_output) != expected_length:
            raise ValueError(
                f"Resolved validation reward stream length mismatch: {len(reward_output)=}, {expected_length=}."
            )
        if order_indices is not None:
            order = torch.as_tensor(np.asarray(order_indices, dtype=np.int64), dtype=torch.long)
            reward_output.reorder(order)
        return reward_output

    def _performance_metrics(self, metrics: list[list[dict[str, Any]]], output: DataProto) -> dict[str, float]:
        timing = {}
        flat_metrics = [metric for chunk in metrics for metric in chunk]
        if not flat_metrics:
            return timing

        t_generate_sequences = np.array([metric.get("generate_sequences", 0.0) for metric in flat_metrics])
        t_tool_calls = np.array([metric.get("tool_calls", 0.0) for metric in flat_metrics])
        t_compute_score = np.array([metric.get("compute_score", 0.0) for metric in flat_metrics])
        num_preempted = np.array([metric.get("num_preempted", -1) for metric in flat_metrics])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        metric_keys = sorted(
            {
                key
                for metric in flat_metrics
                for key, value in metric.items()
                if key != "num_preempted" and isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        for key in metric_keys:
            values = np.array([metric.get(key, 0.0) for metric in flat_metrics], dtype=np.float64)
            timing[f"agent_loop/{key}/min"] = values.min()
            timing[f"agent_loop/{key}/max"] = values.max()
            timing[f"agent_loop/{key}/mean"] = values.mean()
            timing[f"agent_loop/slowest/{key}"] = values[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

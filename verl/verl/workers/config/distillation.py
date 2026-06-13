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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = [
    "DistillationLossConfig",
    "DistillationEvidenceWeightingConfig",
    "DistillationTeacherEMAConfig",
    "DistillationTeacherCropAsyncPrepareConfig",
    "DistillationTeacherCropConfig",
    "DistillationTeacherModelConfig",
    "DistillationConfig",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationEvidenceWeightingConfig(BaseConfig):
    """Relative evidence weighting for OPD rollouts."""

    enabled: bool = False
    negative_mode: str = "downsample_teacher_crop"
    negative_image_key: str = "teacher_neg_images"
    negative_prompt_key: str = "teacher_neg_raw_prompt"
    downsample_factor: int = 2
    gamma: float = 1.0
    alpha: float = 0.5
    w_min: float = 0.0
    w_max: float = 2.0
    eps: float = 1e-6

    def __post_init__(self):
        if not self.enabled:
            return
        supported_modes = {"downsample_teacher_crop", "hard_negative_crop"}
        if self.negative_mode not in supported_modes:
            raise ValueError(
                "distillation.evidence_weighting.negative_mode must be one of "
                f"{sorted(supported_modes)}, got {self.negative_mode!r}."
            )
        if self.negative_mode == "hard_negative_crop" and not self.negative_image_key:
            raise ValueError(
                "distillation.evidence_weighting.negative_image_key must be set when "
                "negative_mode='hard_negative_crop'."
            )
        if self.negative_mode == "hard_negative_crop" and not self.negative_prompt_key:
            raise ValueError(
                "distillation.evidence_weighting.negative_prompt_key must be set when "
                "negative_mode='hard_negative_crop'."
            )
        if self.negative_mode == "downsample_teacher_crop" and self.downsample_factor <= 1:
            raise ValueError("distillation.evidence_weighting.downsample_factor must be greater than 1.")
        if self.gamma < 0:
            raise ValueError("distillation.evidence_weighting.gamma must be non-negative.")
        if self.alpha < 0:
            raise ValueError("distillation.evidence_weighting.alpha must be non-negative.")
        if self.w_min < 0:
            raise ValueError("distillation.evidence_weighting.w_min must be non-negative.")
        if self.w_max < self.w_min:
            raise ValueError("distillation.evidence_weighting.w_max must be >= w_min.")
        if self.eps <= 0:
            raise ValueError("distillation.evidence_weighting.eps must be positive.")


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationTeacherCropAsyncPrepareConfig(BaseConfig):
    """Configuration for overlapping teacher-crop preprocessing with student rollout."""

    enabled: bool = False
    concurrency: int = 8

    def __post_init__(self):
        if self.concurrency <= 0:
            raise ValueError("distillation.teacher_crop.async_prepare.concurrency must be positive.")


@dataclass
class DistillationTeacherCropConfig(BaseConfig):
    """Configuration for teacher-only crop supervision prompts."""

    enabled: bool = False
    prompt_key: str = "teacher_raw_prompt"
    image_key: str = "teacher_images"
    async_prepare: DistillationTeacherCropAsyncPrepareConfig = field(
        default_factory=DistillationTeacherCropAsyncPrepareConfig
    )

    def __post_init__(self):
        # Validate only the opt-in teacher-crop config surface.
        if not self.enabled:
            return
        if not self.prompt_key:
            raise ValueError("distillation.teacher_crop.prompt_key must be set when teacher crop is enabled.")
        if not self.image_key:
            raise ValueError("distillation.teacher_crop.image_key must be set when teacher crop is enabled.")


@dataclass
class DistillationTeacherEMAConfig(BaseConfig):
    """Configuration for EMA-updated distillation teacher weights."""

    enabled: bool = False
    decay: float = 0.9
    device: str = "cpu"
    dtype: str = "bf16"
    update_freq: int = 1
    sync_freq: int = 1
    warmup_steps: int = 0
    init_from_student: bool = True
    require_same_model: bool = True

    def __post_init__(self):
        device = str(self.device).lower()
        if not 0 <= self.decay < 1:
            raise ValueError("distillation.teacher_ema.decay must be in [0, 1).")
        if device not in {"cpu", "gpu", "cuda", "npu"}:
            raise ValueError("distillation.teacher_ema.device must be one of: cpu, gpu, cuda, npu.")
        if self.dtype not in {"bf16", "bfloat16", "fp32", "float32"}:
            raise ValueError("distillation.teacher_ema.dtype must be one of: bf16, bfloat16, fp32, float32.")
        if self.update_freq <= 0:
            raise ValueError("distillation.teacher_ema.update_freq must be positive.")
        if self.sync_freq <= 0:
            raise ValueError("distillation.teacher_ema.sync_freq must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("distillation.teacher_ema.warmup_steps must be non-negative.")


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    teacher_crop: DistillationTeacherCropConfig = field(default_factory=DistillationTeacherCropConfig)
    teacher_ema: DistillationTeacherEMAConfig = field(default_factory=DistillationTeacherEMAConfig)
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    evidence_weighting: DistillationEvidenceWeightingConfig = field(
        default_factory=DistillationEvidenceWeightingConfig
    )

    def __post_init__(self):
        teacher_crop_enabled = self.teacher_crop.get("enabled", False)
        if teacher_crop_enabled and not self.enabled:
            raise ValueError("distillation.teacher_crop.enabled=True requires distillation.enabled=True.")
        teacher_ema_enabled = self.teacher_ema.get("enabled", False)
        if teacher_ema_enabled and not self.enabled:
            raise ValueError("distillation.teacher_ema.enabled=True requires distillation.enabled=True.")
        evidence_weighting_enabled = self.evidence_weighting.get("enabled", False)
        if evidence_weighting_enabled and not self.enabled:
            raise ValueError("distillation.evidence_weighting.enabled=True requires distillation.enabled=True.")
        if evidence_weighting_enabled and not teacher_crop_enabled:
            raise ValueError(
                "distillation.evidence_weighting.enabled=True requires distillation.teacher_crop.enabled=True."
            )
        if teacher_ema_enabled and not self.teacher_ema.require_same_model:
            raise NotImplementedError("distillation.teacher_ema.require_same_model=False is not supported yet.")
        if not self.enabled:
            return

        self.teacher_models = self._resolve_teacher_models()
        if teacher_ema_enabled and len(self.teacher_models) != 1:
            raise ValueError("distillation.teacher_ema.enabled=True requires exactly one teacher model.")
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models

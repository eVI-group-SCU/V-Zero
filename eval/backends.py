from __future__ import annotations

import base64
import io
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .schema import EvalExample, EvalImage


ZWZ_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    " The assistant first thinks about the reasoning process in the mind and "
    "then provides the user with the answer. The answer are enclosed within "
    "<answer> </answer> tags, respectively, i.e., reasoning process here "
    "<answer> answer here </answer>"
)


@dataclass(slots=True)
class GeneratedBatch:
    responses: list[str]
    batch_elapsed_sec: float


@dataclass(slots=True)
class PreparedBatch:
    examples: list[EvalExample]
    payload: Any
    prepare_elapsed_sec: float = 0.0


class GenerationBackend:
    def prepare_batch(self, examples: list[EvalExample]) -> PreparedBatch:
        return PreparedBatch(examples=examples, payload=examples)

    def generate_prepared_batch(self, prepared: PreparedBatch) -> GeneratedBatch:
        return self.generate_batch(prepared.examples)

    def generate_batch(self, examples: list[EvalExample]) -> GeneratedBatch:
        raise NotImplementedError


def build_backend(args: Any) -> GenerationBackend:
    if args.engine == "vllm":
        return VLLMBackend(args)
    if args.engine == "vllm_server":
        return VLLMServerBackend(args)
    if args.engine == "transformers":
        return TransformersBackend(args)
    raise ValueError(f"Unsupported engine: {args.engine}")


def messages_for_example(
    example: EvalExample,
    system_prompt: str | None = None,
    *,
    include_image_payload: bool = False,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    for image in example.images:
        if include_image_payload:
            user_content.append(
                image_message_content(
                    image,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            )
        else:
            user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": example.question.replace("<image>", "")})
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append({"role": "user", "content": user_content})
    return messages


def pil_images_for_example(example: EvalExample) -> list[Any]:
    return [load_eval_image(image) for image in example.images]


def image_message_content(
    image: EvalImage,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "type": "image",
        "image": image_payload_for_message(image),
    }
    if min_pixels is not None:
        content["min_pixels"] = min_pixels
    if max_pixels is not None:
        content["max_pixels"] = max_pixels
    return content


def image_payload_for_message(image: EvalImage) -> Any:
    if image.kind == "path":
        return str(image.value)
    if image.kind == "pil":
        return image.value
    return load_eval_image(image)


def load_eval_image(image: EvalImage) -> Any:
    from PIL import Image

    if image.kind == "pil":
        return image.value.convert("RGB") if hasattr(image.value, "convert") else image.value
    if image.kind == "path":
        with Image.open(image.value) as handle:
            return handle.convert("RGB")
    if image.kind == "bytes":
        with Image.open(io.BytesIO(image.value)) as handle:
            return handle.convert("RGB")
    if image.kind == "base64":
        payload = str(image.value)
        if "," in payload and payload.lstrip().startswith("data:"):
            payload = payload.split(",", 1)[1]
        data = base64.b64decode(payload)
        with Image.open(io.BytesIO(data)) as handle:
            return handle.convert("RGB")
    raise ValueError(f"Unsupported image kind: {image.kind}")


def sampling_kwargs(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    repetition_penalty: float,
    seed: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": 0.0 if temperature <= 0 else temperature,
        "max_tokens": max_new_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
    }
    return kwargs


class TransformersBackend(GenerationBackend):
    def __init__(self, args: Any) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        model_kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, args.torch_dtype),
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "device_map": "auto",
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        processor_kwargs: dict[str, Any] = {}
        if args.max_pixels is not None:
            processor_kwargs["max_pixels"] = args.max_pixels
        self.processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            **processor_kwargs,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
        self.model.eval()
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.repetition_penalty = args.repetition_penalty
        self.seed = args.seed
        self.system_prompt = args.system_prompt.strip() or None
        self.enable_thinking = args.enable_thinking

    def generate_batch(self, examples: list[EvalExample]) -> GeneratedBatch:
        if len(examples) != 1:
            raise ValueError("Transformers backend currently supports one example per batch.")
        started = time.perf_counter()
        example = examples[0]
        prompt = self.processor.apply_chat_template(
            messages_for_example(example, self.system_prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.processor(
            text=[prompt],
            images=pil_images_for_example(example),
            padding=False,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "use_cache": True,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
            gen_kwargs["top_k"] = self.top_k
        self.torch.manual_seed(self.seed)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        trimmed = [output_ids[prompt_length:] for output_ids in generated_ids]
        responses = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return GeneratedBatch(
            responses=[responses[0].strip()],
            batch_elapsed_sec=time.perf_counter() - started,
        )


class VLLMBackend(GenerationBackend):
    def __init__(self, args: Any) -> None:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        try:
            from qwen_vl_utils import process_vision_info
        except ModuleNotFoundError:
            process_vision_info = None

        processor_kwargs: dict[str, Any] = {}
        if args.max_pixels is not None:
            processor_kwargs["max_pixels"] = args.max_pixels
        self.processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            **processor_kwargs,
        )
        llm_kwargs: dict[str, Any] = {
            "model": args.model,
            "dtype": args.vllm_dtype,
            "trust_remote_code": args.trust_remote_code,
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "limit_mm_per_prompt": {"image": args.vllm_limit_images_per_prompt},
            "disable_custom_all_reduce": args.vllm_disable_custom_all_reduce,
        }
        if args.vllm_data_parallel_size > 1:
            llm_kwargs["data_parallel_size"] = args.vllm_data_parallel_size
        if args.vllm_max_model_len is not None:
            llm_kwargs["max_model_len"] = args.vllm_max_model_len
        if args.vllm_max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
        if args.vllm_enforce_eager:
            llm_kwargs["enforce_eager"] = True
        self.llm = LLM(**llm_kwargs)
        self.sampling_params_cls = SamplingParams
        self.process_vision_info = process_vision_info
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.presence_penalty = args.presence_penalty
        self.repetition_penalty = args.repetition_penalty
        self.seed = args.seed
        self.min_pixels = args.min_pixels
        self.max_pixels = args.max_pixels
        self.system_prompt = args.system_prompt.strip() or None
        self.enable_thinking = args.enable_thinking

    def prepare_batch(self, examples: list[EvalExample]) -> PreparedBatch:
        started = time.perf_counter()
        requests: list[dict[str, Any]] = []
        for example in examples:
            if self.process_vision_info is None and example.images:
                raise RuntimeError(
                    "qwen_vl_utils is required for vLLM multimodal eval with "
                    "ZwZ/Qwen pixel settings. Install qwen-vl-utils in this environment."
                )
            messages = messages_for_example(
                example,
                self.system_prompt,
                include_image_payload=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            request: dict[str, Any] = {"prompt": prompt}
            if example.images:
                image_inputs, video_inputs = self.process_vision_info(messages)
                multi_modal_data: dict[str, Any] = {}
                if image_inputs is not None:
                    multi_modal_data["image"] = image_inputs
                if video_inputs is not None:
                    multi_modal_data["video"] = video_inputs
                if multi_modal_data:
                    request["multi_modal_data"] = multi_modal_data
            requests.append(request)
        return PreparedBatch(
            examples=examples,
            payload=requests,
            prepare_elapsed_sec=time.perf_counter() - started,
        )

    def generate_prepared_batch(self, prepared: PreparedBatch) -> GeneratedBatch:
        requests = prepared.payload
        params = self.sampling_params_cls(
            **sampling_kwargs(
                self.max_new_tokens,
                self.temperature,
                self.top_p,
                self.top_k,
                self.presence_penalty,
                self.repetition_penalty,
                self.seed,
            )
        )
        started = time.perf_counter()
        outputs = self.llm.generate(requests, sampling_params=params)
        elapsed = time.perf_counter() - started
        if len(outputs) != len(prepared.examples):
            raise RuntimeError("vLLM returned the wrong number of outputs.")
        responses: list[str] = []
        for output in outputs:
            completions = output.outputs
            if not completions:
                raise RuntimeError("vLLM returned no completion outputs.")
            responses.append(completions[0].text.strip())
        return GeneratedBatch(responses=responses, batch_elapsed_sec=elapsed)

    def generate_batch(self, examples: list[EvalExample]) -> GeneratedBatch:
        return self.generate_prepared_batch(self.prepare_batch(examples))


class VLLMServerBackend(GenerationBackend):
    def __init__(self, args: Any) -> None:
        self.base_url = chat_completions_url(args.vllm_server_base_url)
        self.openai_base_url = openai_base_url(args.vllm_server_base_url)
        self.model = args.served_model_name or args.model
        self.api_key = server_api_key(args.vllm_server_api_key_env)
        self.timeout = args.vllm_server_timeout
        self.max_retries = args.vllm_server_max_retries
        self.retry_sleep = args.vllm_server_retry_sleep
        self.request_concurrency = args.vllm_server_request_concurrency
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.presence_penalty = args.presence_penalty
        self.repetition_penalty = args.repetition_penalty
        self.seed = args.seed
        self.min_pixels = args.min_pixels
        self.max_pixels = args.max_pixels
        self.system_prompt = args.system_prompt.strip() or None
        self.enable_thinking = args.enable_thinking
        self._thread_local = threading.local()

    def prepare_batch(self, examples: list[EvalExample]) -> PreparedBatch:
        started = time.perf_counter()
        payload = [self.request_payload(example) for example in examples]
        return PreparedBatch(
            examples=examples,
            payload=payload,
            prepare_elapsed_sec=time.perf_counter() - started,
        )

    def request_payload(self, example: EvalExample) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages_for_example(
                example,
                self.system_prompt,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            ),
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0 if self.temperature <= 0 else self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return payload

    def generate_prepared_batch(self, prepared: PreparedBatch) -> GeneratedBatch:
        requests = prepared.payload
        started = time.perf_counter()
        max_workers = max(1, min(self.request_concurrency, len(requests)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses = list(executor.map(self.post_chat_completion, requests))
        return GeneratedBatch(
            responses=responses,
            batch_elapsed_sec=time.perf_counter() - started,
        )

    def generate_batch(self, examples: list[EvalExample]) -> GeneratedBatch:
        return self.generate_prepared_batch(self.prepare_batch(examples))

    def post_chat_completion(self, payload: dict[str, Any]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.create_chat_completion(payload)
                return extract_openai_chat_completion_text(response)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"vLLM server request failed: {last_error}") from last_error

    def openai_client(self) -> Any:
        client = getattr(self._thread_local, "openai_client", None)
        if client is None:
            client = create_openai_client(
                api_key=self.api_key or "EMPTY",
                base_url=self.openai_base_url,
                timeout=self.timeout,
            )
            self._thread_local.openai_client = client
        return client

    def create_chat_completion(self, payload: dict[str, Any]) -> Any:
        extra_body: dict[str, Any] = {}
        for key in ("top_k", "repetition_penalty", "chat_template_kwargs"):
            if key in payload:
                extra_body[key] = payload[key]
        kwargs: dict[str, Any] = {
            "model": payload["model"],
            "messages": payload["messages"],
            "max_tokens": payload["max_tokens"],
            "temperature": payload["temperature"],
            "top_p": payload["top_p"],
            "presence_penalty": payload["presence_penalty"],
            "seed": payload["seed"],
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        return self.openai_client().chat.completions.create(**kwargs)


def openai_messages_for_example(
    example: EvalExample,
    system_prompt: str | None,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    for image in example.images:
        user_content.append(
            openai_image_content(
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        )
    user_content.append({"type": "text", "text": example.question.replace("<image>", "")})
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def openai_image_content(
    image: EvalImage,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": image_data_url(image)},
    }
    if min_pixels is not None:
        content["min_pixels"] = min_pixels
    if max_pixels is not None:
        content["max_pixels"] = max_pixels
    return content


def image_data_url(image: EvalImage) -> str:
    if image.kind == "base64":
        payload = str(image.value)
        if payload.lstrip().startswith("data:image/"):
            return payload
        return f"data:image/jpeg;base64,{payload}"
    if image.kind == "bytes":
        return f"data:image/jpeg;base64,{base64.b64encode(image.value).decode('ascii')}"
    if image.kind == "path":
        path = image.value
        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    if image.kind == "pil":
        buffer = io.BytesIO()
        image.value.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    raise ValueError(f"Unsupported image kind: {image.kind}")


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def create_openai_client(*, api_key: str, base_url: str, timeout: float) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def server_api_key(env_name: str | None) -> str | None:
    if not env_name:
        return None
    import os

    return os.environ.get(env_name)


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from vLLM server: {body}") from exc


def extract_chat_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not choices:
        raise RuntimeError(f"vLLM server response has no choices: {response}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"vLLM server response has no message content: {response}")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(parts).strip()
    return str(content).strip()


def extract_openai_chat_completion_text(response: Any) -> str:
    if isinstance(response, dict):
        return extract_chat_completion_text(response)
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError(f"vLLM server response has no choices: {response}")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content")
    if content is None:
        raise RuntimeError(f"vLLM server response has no message content: {response}")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(parts).strip()
    return str(content).strip()

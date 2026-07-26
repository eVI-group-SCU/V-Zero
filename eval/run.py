from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    class _NullProgress:
        def update(self, _: int) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(*_: Any, **__: Any) -> _NullProgress:
        return _NullProgress()

from .backends import PreparedBatch, ZWZ_SYSTEM_PROMPT, build_backend
from .datasets import DATASET_ORDER, iter_dataset, normalize_dataset_name
from .judge import (
    add_judge_args,
    build_judge_config,
    judge_mcq_response,
    judge_open_response,
    should_judge_example,
    should_judge_mcq_parse_failure,
)
from .schema import EvalExample, EvalResult
from .scoring import expected_mcq_letter, score_response
from .summary import completed_indices, summarize_results, write_result


DEFAULT_DATALOADER_WORKERS = 8
DEFAULT_PREFETCH_BATCHES = 2


@dataclass(slots=True)
class ScheduledBatch:
    batch: list[tuple[int, EvalExample]]
    future: Future[PreparedBatch]
    submitted_at: float


@dataclass(slots=True)
class ServerSampleOutput:
    row_index: int
    example: EvalExample
    response: str
    generation_elapsed_sec: float
    sample_backend_elapsed_sec: float


def optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


def bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal eval benchmarks.")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("EVAL_DATASET", "all"),
        help=(
            "vstar, hrbench_4k, hrbench_8k, zoombench, mmstar, "
            "mme_realworld, mme_realworld_en, mme_realworld_cn, "
            "cvbench, colorbench, babyvision, pope, mmvp, "
            "default, core, or all."
        ),
    )
    parser.add_argument("--model", required=True, help="Model name, local checkpoint, or HF path.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("EVAL_DATA_ROOT", "data/eval")),
        help=(
            "Root containing benchmark directories (including POPE/ and MMVP/) "
            "or an HF cache root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("EVAL_OUTPUT_DIR", "results/eval")),
    )
    parser.add_argument("--output-name", default=None, help="Optional output file stem.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--engine",
        choices=("transformers", "vllm", "vllm_server"),
        default=os.environ.get("EVAL_ENGINE", "vllm"),
    )
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "256")))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("EVAL_MAX_NEW_TOKENS", "8192")),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("EVAL_TEMPERATURE", "0.7")),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.environ.get("EVAL_TOP_P", "0.8")),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("EVAL_TOP_K", "20")),
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=float(os.environ.get("EVAL_PRESENCE_PENALTY", "1.5")),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.environ.get("EVAL_REPETITION_PENALTY", "1.0")),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("EVAL_SEED", "42")),
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("EVAL_SYSTEM_PROMPT", ZWZ_SYSTEM_PROMPT),
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=bool_env("EVAL_ENABLE_THINKING", False),
        help="Enable Qwen hybrid-model thinking mode in chat templates. Default: disabled.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=int(os.environ.get("EVAL_MIN_PIXELS", "65536")),
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=int(os.environ.get("EVAL_MAX_PIXELS", "16777216")),
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--torch-dtype", default=os.environ.get("EVAL_TORCH_DTYPE", "bfloat16"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=bool_env("EVAL_LOCAL_FILES_ONLY", True),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=bool_env("EVAL_RESUME", False),
    )
    parser.add_argument(
        "--zoombench-view",
        choices=("full", "crop", "both"),
        default=os.environ.get("ZOOMBENCH_VIEW", "full"),
    )
    parser.add_argument(
        "--vllm-dtype",
        default=os.environ.get("EVAL_VLLM_DTYPE", "bfloat16"),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("EVAL_VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("EVAL_VLLM_TENSOR_PARALLEL_SIZE", "1")),
    )
    parser.add_argument(
        "--vllm-data-parallel-size",
        type=int,
        default=int(os.environ.get("EVAL_VLLM_DATA_PARALLEL_SIZE", "1")),
        help="Number of vLLM data-parallel replicas. Keep tensor parallel size at 1 unless one replica needs multiple GPUs.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=optional_int_env("EVAL_VLLM_MAX_MODEL_LEN"),
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=int(os.environ.get("EVAL_VLLM_MAX_NUM_SEQS", "192")),
    )
    parser.add_argument(
        "--vllm-limit-images-per-prompt",
        type=int,
        default=int(os.environ.get("EVAL_VLLM_LIMIT_IMAGES_PER_PROMPT", "1")),
    )
    parser.add_argument(
        "--vllm-disable-custom-all-reduce",
        action=argparse.BooleanOptionalAction,
        default=bool_env("EVAL_VLLM_DISABLE_CUSTOM_ALL_REDUCE", True),
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=bool_env("EVAL_VLLM_ENFORCE_EAGER", False),
        help="Disable vLLM cudagraph capture/replay for better TP stability.",
    )
    parser.add_argument(
        "--vllm-server-base-url",
        default=os.environ.get("EVAL_VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible vLLM server base URL for --engine vllm_server.",
    )
    parser.add_argument(
        "--served-model-name",
        default=os.environ.get("EVAL_SERVED_MODEL_NAME"),
        help="Model id served by the vLLM OpenAI server. Defaults to --model.",
    )
    parser.add_argument(
        "--vllm-server-api-key-env",
        default=os.environ.get("EVAL_VLLM_SERVER_API_KEY_ENV"),
        help="Optional environment variable that stores the vLLM server API key.",
    )
    parser.add_argument(
        "--vllm-server-timeout",
        type=float,
        default=float(os.environ.get("EVAL_VLLM_SERVER_TIMEOUT", "300")),
    )
    parser.add_argument(
        "--vllm-server-max-retries",
        type=int,
        default=int(os.environ.get("EVAL_VLLM_SERVER_MAX_RETRIES", "2")),
    )
    parser.add_argument(
        "--vllm-server-retry-sleep",
        type=float,
        default=float(os.environ.get("EVAL_VLLM_SERVER_RETRY_SLEEP", "2.0")),
    )
    parser.add_argument(
        "--vllm-server-request-concurrency",
        type=int,
        default=int(os.environ.get("EVAL_VLLM_SERVER_REQUEST_CONCURRENCY", "256")),
        help="Concurrent chat-completion requests sent per eval batch in vLLM server mode.",
    )
    parser.add_argument(
        "--vllm-server-global-concurrency",
        action=argparse.BooleanOptionalAction,
        default=bool_env("EVAL_VLLM_SERVER_GLOBAL_CONCURRENCY", True),
        help=(
            "Use a per-sample global scheduler for vLLM server mode so slow samples "
            "do not block an entire eval batch. Other engines keep the batch scheduler."
        ),
    )
    parser.add_argument(
        "--vllm-server-ready-timeout",
        type=float,
        default=float(os.environ.get("EVAL_VLLM_SERVER_READY_TIMEOUT", "1800")),
        help="Seconds to wait for the vLLM OpenAI server before evaluation starts.",
    )
    parser.add_argument(
        "--vllm-server-ready-interval",
        type=float,
        default=float(os.environ.get("EVAL_VLLM_SERVER_READY_INTERVAL", "10")),
        help="Seconds between vLLM server readiness checks.",
    )
    add_judge_args(parser)
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = normalize_dataset_name(args.dataset)
    output_stem = args.output_name or default_output_stem(args, dataset_name)
    output_path = args.output_dir / f"{output_stem}_results.jsonl"
    summary_path = args.output_dir / f"{output_stem}_summary.json"
    completed = completed_indices(output_path) if args.resume else set()
    judge_config = build_judge_config(args)

    backend = build_backend(args)
    started_at = time.perf_counter()
    progress = tqdm(desc=f"{args.model} {dataset_name} ({args.engine})")

    if use_vllm_server_global_scheduler(args):
        with output_path.open(output_file_mode(args.resume), encoding="utf-8") as output_handle:
            run_vllm_server_global_scheduler(
                args,
                backend,
                dataset_name,
                completed,
                output_handle,
                output_path,
                summary_path,
                started_at,
                progress,
                judge_config,
            )
    else:
        with output_path.open(output_file_mode(args.resume), encoding="utf-8") as output_handle:
            async for scheduled, prepared, prepare_error in iter_prepared_batches(
                args,
                backend,
                dataset_name,
                completed,
            ):
                if prepare_error is not None:
                    write_batch_failures(
                        args,
                        scheduled.batch,
                        output_handle,
                        str(prepare_error),
                        (time.perf_counter() - scheduled.submitted_at)
                        / max(len(scheduled.batch), 1),
                        started_at,
                    )
                else:
                    run_prepared_batch(
                        args,
                        backend,
                        scheduled.batch,
                        prepared,
                        output_handle,
                        started_at,
                        judge_config,
                    )
                write_summary(summary_path, output_path, started_at, args)
                progress.update(len(scheduled.batch))

    progress.close()
    write_summary(summary_path, output_path, started_at, args)
    print(json.dumps(summarize_results(output_path, time.perf_counter() - started_at), ensure_ascii=False, indent=2))


def use_vllm_server_global_scheduler(args: argparse.Namespace) -> bool:
    return args.engine == "vllm_server" and args.vllm_server_global_concurrency


def validate_args(args: argparse.Namespace) -> None:
    if args.data_root is None:
        raise ValueError("Set --data-root or EVAL_DATA_ROOT.")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1.")
    if args.engine == "transformers" and args.eval_batch_size != 1:
        raise ValueError("Transformers backend currently requires --eval-batch-size 1.")
    if args.vllm_tensor_parallel_size < 1:
        raise ValueError("--vllm-tensor-parallel-size must be at least 1.")
    if args.vllm_data_parallel_size < 1:
        raise ValueError("--vllm-data-parallel-size must be at least 1.")
    if args.vllm_server_request_concurrency < 1:
        raise ValueError("--vllm-server-request-concurrency must be at least 1.")
    if args.vllm_server_ready_timeout < 0:
        raise ValueError("--vllm-server-ready-timeout must be non-negative.")
    if args.vllm_server_ready_interval <= 0:
        raise ValueError("--vllm-server-ready-interval must be positive.")
    if args.min_pixels < 1:
        raise ValueError("--min-pixels must be positive.")
    if args.max_pixels < args.min_pixels:
        raise ValueError("--max-pixels must be greater than or equal to --min-pixels.")
    normalize_dataset_name(args.dataset)
    if args.engine == "vllm_server":
        wait_for_vllm_server_ready(args)


def wait_for_vllm_server_ready(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.vllm_server_ready_timeout
    models_url = vllm_server_models_url(args.vllm_server_base_url)
    expected_model = args.served_model_name
    while True:
        try:
            request = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            model_ids = {
                str(model.get("id"))
                for model in payload.get("data", [])
                if isinstance(model, dict) and model.get("id") is not None
            }
            if expected_model is None or expected_model in model_ids:
                print(f"vLLM server ready: {models_url}")
                return
            error = f"served model {expected_model!r} not in {sorted(model_ids)}"
        except (OSError, json.JSONDecodeError, urllib.error.URLError) as exc:
            error = str(exc)

        if time.monotonic() >= deadline:
            raise RuntimeError(f"vLLM server is not ready at {models_url}: {error}")
        print(f"api server not ready: {models_url} ({error})")
        time.sleep(args.vllm_server_ready_interval)


def vllm_server_models_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return f"{normalized}/models"


def run_vllm_server_global_scheduler(
    args: argparse.Namespace,
    backend: Any,
    dataset_name: str,
    completed: set[int],
    output_handle,
    output_path: Path,
    summary_path: Path,
    started_at: float,
    progress: Any,
    judge_config: Any,
) -> None:
    max_workers = args.vllm_server_request_concurrency
    max_pending = max(max_workers, args.eval_batch_size)
    written_since_summary = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: dict[Future[ServerSampleOutput], tuple[int, EvalExample]] = {}

        for row_index, example in iter_dataset(
            dataset_name,
            data_root=args.data_root,
            start_index=args.start_index,
            limit=args.limit,
            zoombench_view=args.zoombench_view,
        ):
            if row_index in completed:
                continue
            future = executor.submit(run_vllm_server_sample, backend, row_index, example)
            pending[future] = (row_index, example)
            if len(pending) >= max_pending:
                done, pending = drain_completed_vllm_server_futures(
                    pending,
                    args,
                    output_handle,
                    started_at,
                    judge_config,
                    progress,
                    return_when=FIRST_COMPLETED,
                )
                written_since_summary += done
                if written_since_summary >= args.eval_batch_size:
                    write_summary(summary_path, output_path, started_at, args)
                    written_since_summary = 0

        while pending:
            done, pending = drain_completed_vllm_server_futures(
                pending,
                args,
                output_handle,
                started_at,
                judge_config,
                progress,
                return_when=FIRST_COMPLETED,
            )
            written_since_summary += done
            if written_since_summary >= args.eval_batch_size:
                write_summary(summary_path, output_path, started_at, args)
                written_since_summary = 0

    if written_since_summary:
        write_summary(summary_path, output_path, started_at, args)


def drain_completed_vllm_server_futures(
    pending: dict[Future[ServerSampleOutput], tuple[int, EvalExample]],
    args: argparse.Namespace,
    output_handle,
    started_at: float,
    judge_config: Any,
    progress: Any,
    *,
    return_when: str,
) -> tuple[int, dict[Future[ServerSampleOutput], tuple[int, EvalExample]]]:
    done, remaining = wait(set(pending), return_when=return_when)
    remaining_map = {future: pending[future] for future in remaining}
    for future in done:
        row_index, example = pending[future]
        sample_backend_elapsed_sec = 0.0
        try:
            sample = future.result()
            sample_backend_elapsed_sec = sample.sample_backend_elapsed_sec
            result = build_success_result(
                args,
                sample.row_index,
                sample.example,
                sample.response,
                generation_elapsed_sec=sample.generation_elapsed_sec,
                batch_generation_elapsed_sec=sample.generation_elapsed_sec,
                sample_backend_elapsed_sec=sample.sample_backend_elapsed_sec,
                started_at=started_at,
                judge_config=judge_config,
            )
        except Exception as exc:
            result = failure_result(
                args,
                row_index,
                example,
                str(exc),
                sample_backend_elapsed_sec,
                started_at,
            )
        write_result(output_handle, result)
        progress.update(1)
    return len(done), remaining_map


def run_vllm_server_sample(
    backend: Any,
    row_index: int,
    example: EvalExample,
) -> ServerSampleOutput:
    started = time.perf_counter()
    payload = backend.request_payload(example)
    generation_started = time.perf_counter()
    response = backend.post_chat_completion(payload)
    generation_elapsed_sec = time.perf_counter() - generation_started
    return ServerSampleOutput(
        row_index=row_index,
        example=example,
        response=response,
        generation_elapsed_sec=generation_elapsed_sec,
        sample_backend_elapsed_sec=time.perf_counter() - started,
    )


def generation_batch_size(args: argparse.Namespace) -> int:
    if args.engine == "transformers":
        return 1
    return args.eval_batch_size


def output_file_mode(resume: bool) -> str:
    return "a" if resume else "w"


def dataloader_worker_count() -> int:
    return DEFAULT_DATALOADER_WORKERS


def prefetch_batch_count() -> int:
    return DEFAULT_PREFETCH_BATCHES


def iter_eval_batches(
    args: argparse.Namespace,
    dataset_name: str,
    completed: set[int],
) -> Iterator[list[tuple[int, EvalExample]]]:
    batch: list[tuple[int, EvalExample]] = []
    for row_index, example in iter_dataset(
        dataset_name,
        data_root=args.data_root,
        start_index=args.start_index,
        limit=args.limit,
        zoombench_view=args.zoombench_view,
    ):
        if row_index in completed:
            continue
        batch.append((row_index, example))
        if len(batch) >= generation_batch_size(args):
            yield batch
            batch = []
    if batch:
        yield batch


async def iter_prepared_batches(
    args: argparse.Namespace,
    backend: Any,
    dataset_name: str,
    completed: set[int],
) -> AsyncIterator[tuple[ScheduledBatch, PreparedBatch | None, Exception | None]]:
    loop = asyncio.get_running_loop()
    batches = iter(iter_eval_batches(args, dataset_name, completed))
    pending: deque[ScheduledBatch] = deque()
    exhausted = False

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        nonlocal exhausted
        if exhausted:
            return False
        try:
            batch = next(batches)
        except StopIteration:
            exhausted = True
            return False
        future = executor.submit(prepare_batch, backend, batch)
        pending.append(
            ScheduledBatch(
                batch=batch,
                future=future,
                submitted_at=time.perf_counter(),
            )
        )
        return True

    with ThreadPoolExecutor(max_workers=dataloader_worker_count()) as executor:
        for _ in range(prefetch_batch_count()):
            if not submit_next(executor):
                break

        while pending:
            scheduled = pending.popleft()
            try:
                prepared = await asyncio.wrap_future(scheduled.future, loop=loop)
                error = None
            except Exception as exc:
                prepared = None
                error = exc

            while len(pending) < prefetch_batch_count():
                if not submit_next(executor):
                    break

            yield scheduled, prepared, error


def prepare_batch(
    backend: Any,
    batch: list[tuple[int, EvalExample]],
) -> PreparedBatch:
    examples = [example for _, example in batch]
    if hasattr(backend, "prepare_batch"):
        return backend.prepare_batch(examples)
    return PreparedBatch(examples=examples, payload=examples)


def default_output_stem(args: argparse.Namespace, dataset_name: str) -> str:
    output_tag = re.sub(r"[^A-Za-z0-9]+", "_", args.output_dir.name).strip("_").lower()
    if not output_tag:
        output_tag = "eval"
    if dataset_name == "all":
        return f"{output_tag}_all_full"
    return f"{output_tag}_{dataset_name}_full"


def run_batch(
    args: argparse.Namespace,
    backend: Any,
    batch: list[tuple[int, EvalExample]],
    output_handle,
    started_at: float,
    judge_config: Any,
) -> None:
    try:
        prepared = prepare_batch(backend, batch)
    except Exception as exc:
        write_batch_failures(
            args,
            batch,
            output_handle,
            str(exc),
            0.0,
            started_at,
        )
        return
    run_prepared_batch(
        args,
        backend,
        batch,
        prepared,
        output_handle,
        started_at,
        judge_config,
    )


def run_prepared_batch(
    args: argparse.Namespace,
    backend: Any,
    batch: list[tuple[int, EvalExample]],
    prepared: PreparedBatch,
    output_handle,
    started_at: float,
    judge_config: Any,
) -> None:
    examples = prepared.examples
    row_indices = [row_index for row_index, _ in batch]
    batch_started_at = time.perf_counter()
    try:
        if hasattr(backend, "generate_prepared_batch"):
            generated = backend.generate_prepared_batch(prepared)
        else:
            generated = backend.generate_batch(examples)
    except Exception as exc:
        elapsed = (
            prepared.prepare_elapsed_sec + time.perf_counter() - batch_started_at
        ) / max(len(batch), 1)
        write_batch_failures(
            args,
            batch,
            output_handle,
            str(exc),
            elapsed,
            started_at,
        )
        return

    if len(generated.responses) != len(examples):
        elapsed = (
            prepared.prepare_elapsed_sec + time.perf_counter() - batch_started_at
        ) / max(len(batch), 1)
        message = "Generation backend returned the wrong number of responses."
        write_batch_failures(args, batch, output_handle, message, elapsed, started_at)
        return

    batch_generation_finished_at = time.perf_counter()
    per_sample_generation = generated.batch_elapsed_sec / max(len(examples), 1)
    per_sample_backend_elapsed = (
        prepared.prepare_elapsed_sec + batch_generation_finished_at - batch_started_at
    ) / max(len(examples), 1)
    for row_index, example, response in zip(row_indices, examples, generated.responses, strict=True):
        item_started_at = time.perf_counter()
        try:
            result = build_success_result(
                args,
                row_index,
                example,
                response,
                generation_elapsed_sec=per_sample_generation,
                batch_generation_elapsed_sec=generated.batch_elapsed_sec,
                sample_backend_elapsed_sec=per_sample_backend_elapsed,
                started_at=started_at,
                judge_config=judge_config,
            )
        except Exception as exc:
            result = failure_result(
                args,
                row_index,
                example,
                str(exc),
                per_sample_backend_elapsed + (time.perf_counter() - item_started_at),
                started_at,
            )
        write_result(output_handle, result)


def build_success_result(
    args: argparse.Namespace,
    row_index: int,
    example: EvalExample,
    response: str,
    *,
    generation_elapsed_sec: float,
    batch_generation_elapsed_sec: float,
    sample_backend_elapsed_sec: float,
    started_at: float,
    judge_config: Any,
) -> EvalResult:
    item_started_at = time.perf_counter()
    strict_prediction, strict_correct = score_response(example, response)
    prediction = strict_prediction
    correct = strict_correct
    judge_result = None
    judge_elapsed_sec = None
    judge_task = None
    judge_used_for_score = False
    if should_judge_example(judge_config, example):
        judge_started_at = time.perf_counter()
        judge_result = judge_open_response(judge_config, example, response)
        judge_elapsed_sec = time.perf_counter() - judge_started_at
        judge_task = judge_config.task
        if judge_result.error is None and judge_result.correct is not None:
            correct = judge_result.correct
            judge_used_for_score = True
    elif should_judge_mcq_parse_failure(judge_config, example, strict_prediction):
        judge_started_at = time.perf_counter()
        expected = expected_mcq_letter(example.answer, example.choices)
        judge_result = judge_mcq_response(judge_config, example, response, expected)
        judge_elapsed_sec = time.perf_counter() - judge_started_at
        judge_task = "mcq_correctness"
        if judge_result.error is None and judge_result.correct is not None:
            correct = judge_result.correct
            judge_used_for_score = True
    scoring_elapsed = time.perf_counter() - item_started_at
    return EvalResult(
        ok=True,
        row_index=row_index,
        id=example.id,
        dataset=example.dataset,
        split=example.split,
        model=args.model,
        engine=args.engine,
        answer=example.answer,
        prediction=prediction,
        correct=correct,
        accuracy=1.0 if correct else 0.0,
        strict_prediction=strict_prediction,
        strict_correct=strict_correct,
        response=response,
        category=example.category,
        question_type=example.question_type,
        generation_elapsed_sec=generation_elapsed_sec,
        batch_generation_elapsed_sec=batch_generation_elapsed_sec,
        judge_enabled=judge_result is not None,
        judge_provider=judge_config.provider if judge_result is not None else None,
        judge_task=judge_task,
        judge_model=judge_config.model if judge_result is not None else None,
        judge_prediction=judge_result.prediction if judge_result is not None else None,
        judge_correct=judge_result.correct if judge_result is not None else None,
        judge_used_for_score=judge_used_for_score,
        judge_response=judge_result.raw_response if judge_result is not None else None,
        judge_error=judge_result.error if judge_result is not None else None,
        judge_elapsed_sec=judge_elapsed_sec,
        sample_eval_elapsed_sec=sample_backend_elapsed_sec + scoring_elapsed,
        total_eval_elapsed_sec=time.perf_counter() - started_at,
        metadata=example.metadata,
    )


def write_batch_failures(
    args: argparse.Namespace,
    batch: list[tuple[int, EvalExample]],
    output_handle,
    error: str,
    elapsed: float,
    started_at: float,
) -> None:
    for row_index, example in batch:
        write_result(
            output_handle,
            failure_result(args, row_index, example, error, elapsed, started_at),
        )


def failure_result(
    args: argparse.Namespace,
    row_index: int,
    example: EvalExample,
    error: str,
    elapsed: float,
    started_at: float,
) -> EvalResult:
    return EvalResult(
        ok=False,
        row_index=row_index,
        id=example.id,
        dataset=example.dataset,
        split=example.split,
        model=args.model,
        engine=args.engine,
        category=example.category,
        question_type=example.question_type,
        sample_eval_elapsed_sec=elapsed,
        total_eval_elapsed_sec=time.perf_counter() - started_at,
        error=error,
        metadata=example.metadata,
    )


def write_summary(
    summary_path: Path,
    output_path: Path,
    started_at: float,
    args: argparse.Namespace,
) -> None:
    summary = summarize_results(output_path, time.perf_counter() - started_at)
    summary.update(
        {
            "model": args.model,
            "engine": args.engine,
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "output": str(output_path),
            "eval_batch_size": generation_batch_size(args),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "system_prompt": args.system_prompt,
            "enable_thinking": args.enable_thinking,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_data_parallel_size": args.vllm_data_parallel_size,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_max_num_seqs": args.vllm_max_num_seqs,
            "vllm_limit_images_per_prompt": args.vllm_limit_images_per_prompt,
            "vllm_disable_custom_all_reduce": args.vllm_disable_custom_all_reduce,
            "vllm_enforce_eager": args.vllm_enforce_eager,
            "vllm_server_base_url": args.vllm_server_base_url,
            "served_model_name": args.served_model_name or "",
            "vllm_server_request_concurrency": args.vllm_server_request_concurrency,
            "vllm_server_global_concurrency": args.vllm_server_global_concurrency,
            "judge": args.judge,
            "judge_task": args.judge_task,
            "judge_model": args.judge_model or "",
            "available_dataset_order": list(DATASET_ORDER),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Teacher-only, process-local backport of vLLM #51603 for the installed scheduler."""

import __future__

import ast
import inspect
import json
import os
import socket
import textwrap
import time
from pathlib import Path

PATCH_ID = "51603-encoder-order-with-resume-boundary-v1"
SCHEDULER_CLASS = "teacher_scheduler.TeacherScheduler"


def reordered_schedule(source):
    tree = ast.parse(textwrap.dedent(source))
    matches = []
    for owner in ast.walk(tree):
        for _, body in ast.iter_fields(owner):
            if not isinstance(body, list):
                continue
            for index, node in enumerate(body):
                if isinstance(node, ast.If) and ast.unparse(node.test) == "self.need_mamba_block_aligned_split":
                    matches.append((body, index, node))
    if len(matches) != 2:
        raise ValueError("Unrecognized scheduler: expected exactly two old alignment sites")
    for body, _, alignment in matches:
        index = body.index(alignment)
        preceding = body[index - 1]
        if not isinstance(preceding, ast.If):
            raise ValueError("Unrecognized scheduler alignment predecessor")
        condition = ast.unparse(preceding.test)
        if condition == "request.has_encoder_inputs":
            destination, at = body, index - 1
        elif condition == "load_kv_async":
            destination = preceding.orelse
            sites = [
                i
                for i, node in enumerate(destination)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "request.has_encoder_inputs"
            ]
            if len(sites) != 1:
                raise ValueError("Unrecognized waiting encoder scheduling branch")
            at = sites[0]
        else:
            raise ValueError("Already patched or unrecognized scheduler ordering")
        if "self._try_schedule_encoder_inputs(" not in ast.unparse(destination[at]):
            raise ValueError("Missing original encoder scheduling call")
        body.remove(alignment)
        destination.insert(at, alignment)
    return ast.unparse(ast.fix_missing_locations(tree))


def resume_boundary(original, scheduler, request, count, local=0, external=0):
    if external:
        raise ValueError("Teacher scheduler backport does not support external KV transfer")
    start = request.num_computed_tokens + local
    block = scheduler.cache_config.block_size
    last = request.num_tokens - request.num_tokens % block
    if scheduler.use_eagle:
        raise ValueError("Teacher backport is scoped to the non-speculative production teacher")
    # #51603 assumes an unaligned private state stops at the NEXT cache boundary.
    # The installed older splitter lacks that invariant; never cross it or bypass
    # a budget too small to reach it. All aligned-start behavior remains original.
    if start % block and start < last:
        distance = block - start % block
        return distance if count >= distance else 0
    return original(scheduler, request, count, local, external)


def report_zero_progress(scheduler, output):
    tokens = sum(output.num_scheduled_tokens.values())
    if tokens or not scheduler.has_requests():
        scheduler._vzero_zero_steps = 0
        return
    scheduler._vzero_zero_steps = getattr(scheduler, "_vzero_zero_steps", 0) + 1
    now = time.monotonic()
    if scheduler._vzero_zero_steps < 100 or now - getattr(scheduler, "_vzero_zero_logged", -60) < 60:
        return
    scheduler._vzero_zero_logged = now
    requests = []
    for request in scheduler.requests.values():
        requests.append(
            {
                "request_id": request.request_id,
                "num_computed_tokens": request.num_computed_tokens,
                "num_prompt_tokens": request.num_prompt_tokens,
                "num_tokens": request.num_tokens,
                "mm_positions": [
                    {"offset": mm.mm_position.offset, "length": mm.mm_position.length}
                    for mm in (request.mm_features or [])
                ],
            }
        )
        if len(requests) == 3:
            break
    cache = scheduler.encoder_cache_manager
    record = {
        "patch": PATCH_ID,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "consecutive_zero_steps": scheduler._vzero_zero_steps,
        "time_unix": time.time(),
        "pending_requests": len(scheduler.requests),
        "requests": requests,
        "max_num_encoder_input_tokens": scheduler.max_num_encoder_input_tokens,
        "encoder_cache": {
            key: getattr(cache, key, None) for key in ("cache_size", "num_free_slots", "num_freeable_slots")
        },
        "need_mamba_block_aligned_split": scheduler.need_mamba_block_aligned_split,
        "cache_config_block_size": scheduler.cache_config.block_size,
        "scheduler_block_size": getattr(scheduler, "block_size", None),
    }
    print("TEACHER_ZERO_PROGRESS " + json.dumps(record), flush=True)


def patched_class(base):
    original = base.schedule
    if original.__code__.co_freevars:
        raise ValueError("Unexpected scheduler closure")
    namespace = dict(original.__globals__)
    code = compile(
        reordered_schedule(inspect.getsource(original)),
        "<teacher-51603>",
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, namespace)
    original_split = base._mamba_block_aligned_split
    if "num_new_tokens // block_size * block_size" not in ast.unparse(
        ast.parse(textwrap.dedent(inspect.getsource(original_split)))
    ):
        raise ValueError("Unrecognized old Mamba splitter; review the backport before use")
    schedule = namespace["schedule"]

    class PatchedTeacherScheduler(base):
        def schedule(self, *args, **kwargs):
            output = schedule(self, *args, **kwargs)
            report_zero_progress(self, output)
            return output

        def _mamba_block_aligned_split(
            self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0
        ):
            return resume_boundary(
                original_split,
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )

    return PatchedTeacherScheduler


class TeacherScheduler:
    """vLLM scheduler_cls factory; preserve the resolved sync/async scheduler choice."""

    def __new__(cls, *args, **kwargs):
        config = kwargs.get("vllm_config", args[0] if args else None)
        if config is None or config.speculative_config is not None or config.kv_transfer_config is not None:
            raise ValueError("Teacher backport requires non-speculative, non-disaggregated inference")
        if config.scheduler_config.async_scheduling:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler as Base
        else:
            from vllm.v1.core.sched.scheduler import Scheduler as Base
        scheduler = patched_class(Base)(*args, **kwargs)
        if not scheduler.need_mamba_block_aligned_split or scheduler.use_eagle:
            raise ValueError("Teacher backport requires the audited Mamba-align configuration")
        record = {
            "patch": PATCH_ID,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "base_scheduler": Base.__name__,
            "cache_config_block_size": scheduler.cache_config.block_size,
            "scheduler_block_size": getattr(scheduler, "block_size", None),
            "effective_encoder_compute_budget": scheduler.max_num_encoder_input_tokens,
            "effective_encoder_cache_size": scheduler.encoder_cache_manager.cache_size,
            "need_mamba_block_aligned_split": scheduler.need_mamba_block_aligned_split,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "scope": "teacher_only",
            "arbitrary_sub_block_bypass": False,
        }
        path = (
            Path(os.environ["VZERO_FA3_TRAIN_OUTPUT_DIR"])
            / f"teacher-scheduler-{record['hostname']}-{record['pid']}.json"
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("TEACHER_SCHEDULER_INSTALLED " + json.dumps(record), flush=True)
        return scheduler

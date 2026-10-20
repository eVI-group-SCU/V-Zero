"""CPU scheduling model: installed ordering/splitter, observed image spans, no GPU imports."""
import importlib.util
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch as mock_patch

MODULE = Path(os.environ.get("TEACHER_SCHEDULER_TEST_MODULE",
                            Path(__file__).resolve().with_name("teacher_scheduler.py")))
spec = importlib.util.spec_from_file_location("teacher_scheduler_test", MODULE)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


class OldScheduler:
    def __init__(self, block, spans, tokens, cache, budget=8192):
        self.cache_config = SimpleNamespace(block_size=block)
        self.need_mamba_block_aligned_split = True
        self.use_eagle = False
        self.request = SimpleNamespace(num_computed_tokens=0, num_tokens=tokens, num_prompt_tokens=tokens,
                                       has_encoder_inputs=True)
        self.spans, self.cache, self.budget = spans, cache, budget
        self.resident = set()
        self.running = False

    def has_requests(self):
        return self.request.num_computed_tokens < self.request.num_tokens

    def _mamba_block_aligned_split(self, request, num_new_tokens, local=0, external=0):
        assert external == 0
        num_computed_tokens = request.num_computed_tokens + local + external
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            after = num_computed_tokens + num_new_tokens
            if after < last_cache_position:
                num_new_tokens = num_new_tokens // block_size * block_size
            elif num_computed_tokens < last_cache_position < after:
                num_new_tokens = last_cache_position - num_computed_tokens
        return num_new_tokens

    def _try_schedule_encoder_inputs(self, request, count):
        start = request.num_computed_tokens
        used = sum(self.spans[i][1] for i in self.resident)
        selected = set()
        for i, (offset, length) in enumerate(self.spans):
            if offset >= start + count:
                break
            if offset + length <= start or i in self.resident:
                continue
            if used + length > self.cache:
                count = max(0, offset - start)
                break
            used += length
            selected.add(i)
        return count, selected

    def schedule(self):
        request = self.request
        selected = set()
        num_new_tokens = min(self.budget, request.num_tokens - request.num_computed_tokens)
        if self.running:
            if request.has_encoder_inputs:
                num_new_tokens, selected = self._try_schedule_encoder_inputs(request, num_new_tokens)
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)
        else:
            for _ in range(1):
                load_kv_async = False
                if load_kv_async:
                    num_new_tokens = 0
                else:
                    if request.has_encoder_inputs:
                        num_new_tokens, selected = self._try_schedule_encoder_inputs(request, num_new_tokens)
                        if num_new_tokens == 0:
                            break
                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)
                    if num_new_tokens == 0:
                        break
        if num_new_tokens:
            request.num_computed_tokens += num_new_tokens
            self.resident.update(selected)
            self.resident = {i for i in self.resident
                             if sum(self.spans[i]) > request.num_computed_tokens}
            self.running = True
        return SimpleNamespace(num_scheduled_tokens={"request": num_new_tokens})


class TeacherSchedulerRegression(unittest.TestCase):
    def test_encoder_cap_progress_and_boundary_recovery_without_subblock_bypass(self):
        fixed = patch.patched_class(OldScheduler)
        # Upstream's two individually cacheable images: waiting then running path.
        args = (768, [(0, 500), (510, 500)], 1010, 600)
        self.assertEqual(OldScheduler(*args).schedule().num_scheduled_tokens["request"], 0)
        scheduler = fixed(*args)
        self.assertEqual([scheduler.schedule().num_scheduled_tokens["request"] for _ in range(3)], [510, 258, 242])

        # Actual pending spans. Frontend block=16 does NOT reproduce this hang;
        # effective engine block/cache must be read before claiming the root cause.
        for block, old_stalls in ((16, False), (1024, True)):
            args = (block, [(4, 15708), (15756, 3312)], 19311, 16384)
            for cls, stalls in ((OldScheduler, old_stalls), (fixed, False)):
                scheduler = cls(*args)
                chunks = []
                for _ in range(20):
                    if scheduler.request.num_computed_tokens == scheduler.request.num_tokens:
                        break
                    chunks.append(scheduler.schedule().num_scheduled_tokens["request"])
                self.assertEqual(scheduler.request.num_computed_tokens < scheduler.request.num_tokens, stalls)
                if stalls:
                    self.assertEqual(chunks[-3:], [0, 0, 0])
        # No global bypass for an exhausted token budget, and the original class
        # remains unchanged (student rollout never receives this scheduler_cls).
        scheduler = fixed(768, [], 3000, 600, budget=767)
        self.assertEqual(scheduler.schedule().num_scheduled_tokens["request"], 0)
        scheduler.request.num_computed_tokens = 510
        self.assertEqual(scheduler._mamba_block_aligned_split(scheduler.request, 257), 0)
        self.assertEqual(scheduler._mamba_block_aligned_split(scheduler.request, 1000), 258)
        self.assertEqual(OldScheduler(*args).schedule().num_scheduled_tokens["request"], 8192)
        with self.assertRaises(ValueError):
            patch.reordered_schedule("def schedule(self): return 0")
        scheduler.request.request_id = "pending"
        scheduler.request.mm_features = [SimpleNamespace(mm_position=SimpleNamespace(offset=4, length=15708))]
        scheduler.requests = {"pending": scheduler.request}
        scheduler.max_num_encoder_input_tokens = 16384
        scheduler.encoder_cache_manager = SimpleNamespace(cache_size=16384, num_free_slots=676, num_freeable_slots=0)
        output = SimpleNamespace(num_scheduled_tokens={})
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), mock_patch.object(patch.time, "monotonic", return_value=1000):
            for _ in range(200):
                patch.report_zero_progress(scheduler, output)
        self.assertEqual(len(stream.getvalue().splitlines()), 1)
        event = json.loads(stream.getvalue().split(" ", 1)[1])
        self.assertEqual(event["requests"][0]["mm_positions"], [{"offset": 4, "length": 15708}])
        self.assertEqual(event["encoder_cache"]["cache_size"], 16384)
        self.assertEqual(output.num_scheduled_tokens, {})
        patch.report_zero_progress(scheduler, SimpleNamespace(num_scheduled_tokens={"pending": 1}))
        self.assertEqual(scheduler._shx_zero_steps, 0)
        inference = {"dtype": "bfloat16", "tensor_model_parallel_size": 2, "max_model_len": 32768,
                     "max_num_batched_tokens": 8192, "max_num_seqs": 1024, "enable_chunked_prefill": True,
                     "enable_prefix_caching": True, "enforce_eager": True,
                     "engine_kwargs": {"vllm": {"max_logprobs": 0, "mm_processor_cache_gb": 0,
                                                "scheduler_cls": patch.SCHEDULER_CLASS,
                                                "disable_log_stats": False, "quantization": None}}}
        kwargs = installed_engine_kwargs(inference)
        self.assertEqual(kwargs["tensor_parallel_size"], 2)
        self.assertNotIn("tensor_model_parallel_size", kwargs)
        self.assertEqual(kwargs["max_logprobs"], 0)
        self.assertEqual(kwargs["mm_processor_cache_gb"], 0)
        self.assertIs(kwargs["disable_log_stats"], False)
        self.assertNotIn("quantization", kwargs)
        self.assertNotIn("scheduler_cls", kwargs)
        self.assertEqual(inference["engine_kwargs"]["vllm"]["scheduler_cls"], patch.SCHEDULER_CLASS)
        inference["engine_kwargs"]["vllm"]["tensor_parallel_size"] = 4
        self.assertEqual(installed_engine_kwargs(inference)["tensor_parallel_size"], 4)


def installed_engine_kwargs(inference):
    keys = ("dtype", "max_model_len", "max_num_batched_tokens", "max_num_seqs",
            "enable_chunked_prefill", "enable_prefix_caching", "enforce_eager")
    kwargs = {key: inference[key] for key in keys}
    kwargs["tensor_parallel_size"] = inference["tensor_model_parallel_size"]
    engine_kwargs = inference.get("engine_kwargs", {}).get("vllm", {}) or {}
    kwargs.update({key: value for key, value in engine_kwargs.items() if value is not None})
    kwargs.pop("scheduler_cls", None)
    return kwargs


def installed_pending_case(config_path, blocks):
    """CPU-only real processor budgets/cache/splitter; does not create an LLM or load weights."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    config = json.loads(Path(config_path).read_text())
    teacher = next(iter(config["distillation"]["teacher_models"].values()))
    inference = teacher["inference"]
    vconfig = EngineArgs(model=teacher["model_path"], **installed_engine_kwargs(inference)).create_engine_config()
    budget = MultiModalBudget(vconfig, MULTIMODAL_REGISTRY)
    results = {"encoder_compute_budget": budget.encoder_compute_budget,
               "encoder_cache_size": budget.encoder_cache_size,
               "mm_max_toks_per_item": budget.mm_max_toks_per_item,
               "block_scope": "supplied candidates, NOT an observed live EngineCore block size", "cases": []}
    fixed = patch.patched_class(Scheduler)  # Also compile/validate both REAL installed schedule branches.
    for block in blocks:
        for cls, label in ((Scheduler, "old"), (fixed, "backport")):
            scheduler = object.__new__(cls)
            scheduler.cache_config = SimpleNamespace(block_size=block)
            scheduler.scheduler_config = vconfig.scheduler_config
            scheduler.use_eagle = scheduler.is_encoder_decoder = False
            scheduler.ec_connector = None
            scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=budget.encoder_cache_size)
            request = SimpleNamespace(request_id="observed-pending-shape", num_computed_tokens=0,
                                      num_tokens=19311, num_prompt_tokens=19311, has_encoder_inputs=True,
                                      mm_features=[SimpleNamespace(identifier=str(i), mm_position=PlaceholderRange(offset=o, length=n))
                                                   for i, (o, n) in enumerate(((4, 15708), (15756, 3312)))])
            request.get_num_encoder_embeds = lambda i: request.mm_features[i].mm_position.get_num_embeds()
            chunks = []
            for _ in range(30):
                if request.num_computed_tokens == request.num_tokens:
                    break
                count = min(inference["max_num_batched_tokens"], request.num_tokens - request.num_computed_tokens)
                if label == "backport":
                    count = scheduler._mamba_block_aligned_split(request, count)
                selected, count, _, _ = scheduler._try_schedule_encoder_inputs(
                    request, request.num_computed_tokens, count, budget.encoder_compute_budget)
                if label == "old":
                    count = scheduler._mamba_block_aligned_split(request, count)
                chunks.append(count)
                if not count:
                    continue
                for index in selected:
                    scheduler.encoder_cache_manager.allocate(request, index)
                request.num_computed_tokens += count
                for index, feature in enumerate(request.mm_features):
                    if feature.mm_position.offset + feature.mm_position.length <= request.num_computed_tokens:
                        scheduler.encoder_cache_manager.free_encoder_input(request, index)
            results["cases"].append({"block_size": block, "scheduler": label, "chunks": chunks,
                                     "computed": request.num_computed_tokens,
                                     "complete": request.num_computed_tokens == request.num_tokens})
    print("TEACHER_INSTALLED_CPU_CASE " + json.dumps(results), flush=True)
    if not all(case["complete"] for case in results["cases"] if case["scheduler"] == "backport"):
        raise AssertionError("Backport did not finish the observed shape; do not deploy as a confirmed fix")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--installed-config":
        installed_pending_case(sys.argv[2], [int(value) for value in sys.argv[3:]] or [16, 1024])
    else:
        unittest.main()

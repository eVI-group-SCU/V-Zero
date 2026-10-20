"""Isolated SP1 Qwen3.5 actor overlay; production files and loss stay unchanged.

Ray worker_process_setup_hook: packed_boundary.setup_worker (or call it from an
existing setup hook). Both arms require SHX_PACKED_BOUNDARY_ARM=baseline|boundary
and SHX_PACKED_AUDIT_DIR. Keep this module importable in every actor worker.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import socket
from types import MethodType


API_VERSION = 1
TICKET_KEY = "_shx_packed_boundary_ticket"
_CALL = ContextVar("shx_packed_boundary_call", default=None)


class _Ticket:
    # FSDP recursively copies dict/list kwargs during mixed-precision casting.
    # An opaque object keeps audit state shared through FSDP and checkpointing;
    # the integer metadata tensors are already on the packed input's device.
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getitem__(self, key):
        return getattr(self, key)


def _options(arm=None, audit_dir=None):
    arm = arm or os.environ.get("SHX_PACKED_BOUNDARY_ARM")
    audit_dir = audit_dir or os.environ.get("SHX_PACKED_AUDIT_DIR")
    if arm not in ("baseline", "boundary") or not audit_dir:
        raise RuntimeError("Set SHX_PACKED_BOUNDARY_ARM=baseline|boundary and SHX_PACKED_AUDIT_DIR.")
    return arm, str(audit_dir)


def segment_ids(offsets, total):
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != total or total >= 2**31:
        raise ValueError("Offsets must span exactly one nonempty int32 packed row.")
    if any(type(x) is not int for x in offsets) or any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Empty, noninteger or nonmonotonic packed segments are unsupported.")
    return [i for i, (a, b) in enumerate(zip(offsets, offsets[1:])) for _ in range(b - a)]


def validate_layout(input_shape, position_shape, offsets, text_positions):
    if len(input_shape) != 2 or input_shape[0] != 1:
        raise ValueError("Expected input_ids [1, total_tokens]; no reshaping is performed.")
    if tuple(position_shape) != (4, 1, input_shape[1]):
        raise ValueError("Expected Qwen3.5 position_ids [4, 1, total_tokens]; no axis repair is performed.")
    ids = segment_ids(offsets, input_shape[1])
    expected = [j for a, b in zip(offsets, offsets[1:]) for j in range(b - a)]
    if text_positions != expected:
        raise ValueError("Text position resets do not match the nested input token boundaries.")
    return ids


def _check_kernel(fn, argument, module_prefix):
    if fn is None or argument not in inspect.signature(fn).parameters:
        raise RuntimeError(f"Required native kernel argument is unavailable: {argument}.")
    if not getattr(inspect.unwrap(fn), "__module__", "").startswith(module_prefix):
        raise RuntimeError(f"Unsupported kernel implementation for {argument}; no fallback is installed.")


class Audit:
    def __init__(self, arm, audit_dir, rank, *, in_backward):
        self.arm, self.rank = arm, rank
        self.in_backward = in_backward
        self.path = Path(audit_dir) / f"actor-{socket.gethostname()}-{os.getpid()}-rank{rank}.jsonl"
        self.emitted = set()
        self.counts = Counter()
        self.layers = ()
        self.batch = None
        self.next_ticket = 0

    def emit(self, event, *, once=None, **fields):
        key = event if once is None else once
        if key in self.emitted:
            return
        self.emitted.add(key)
        record = dict(api=API_VERSION, arm=self.arm, rank=self.rank, pid=os.getpid(), event=event, **fields)
        line = json.dumps(record, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print("SHX_PACKED_BOUNDARY " + line, flush=True)

    @contextmanager
    def training_batch(self):
        if self.batch is not None:
            raise RuntimeError("Nested training batches are outside this overlay.")
        self.batch = []
        try:
            yield
            if not self.batch or not self.layers:
                raise RuntimeError("No actor packed guard/model evidence in completed training batch.")
            for ticket in self.batch:
                expected = {(name, phase, kind) for name in self.layers
                            for phase in ("forward", "recompute") for kind in ("conv", "gdn")}
                missing = expected - ticket["seen"]
                if missing:
                    raise RuntimeError(f"Missing actual GDN/checkpoint kernel evidence: {sorted(missing)[:4]}")
            self.counts["backward_batches"] += 1
            self.emit("backward_batch_complete", microbatches=len(self.batch),
                      gdn_layers=len(self.layers), counts=dict(self.counts))
        except BaseException as exc:
            self.emit("audit_failure", error_type=type(exc).__name__, message=str(exc)[:500])
            raise
        finally:
            self.batch = None

    def prepare(self, model_inputs, offsets, torch):
        if TICKET_KEY in model_inputs or "seq_idx" in model_inputs or "cu_seq_lens_q" in model_inputs:
            raise RuntimeError("Unexpected existing boundary kwargs; refusing to overwrite them.")
        packed, positions = model_inputs["input_ids"], model_inputs["position_ids"]
        if model_inputs.get("attention_mask") is not None or model_inputs.get("use_cache", False):
            raise RuntimeError("This overlay requires uncached remove-padding actor inputs.")
        # Only metadata is observed. Never index differentiable activations in an observer.
        with torch.no_grad():
            if tuple(positions.shape) != (4, 1, packed.shape[-1]):
                raise ValueError("Expected position_ids [4, 1, total_tokens]; no axis repair is performed.")
            ids = validate_layout(tuple(packed.shape), tuple(positions.shape), offsets,
                                  positions.detach()[0, 0].cpu().tolist())
            cu = torch.tensor(offsets, dtype=torch.int32, device=packed.device)
            seq = torch.tensor([ids], dtype=torch.int32, device=packed.device)
        self.next_ticket += 1
        ticket = _Ticket(owner=self, id=self.next_ticket, cu=cu, seq=seq, offsets=offsets,
                         train=self.batch is not None, calls=Counter(), seen=set())
        if self.batch is not None:
            self.batch.append(ticket)
        self.counts["packed_guards"] += 1
        mode = "train" if ticket["train"] else "inference"
        self.emit("packed_guard", once=("guard", mode), mode=mode, ticket=ticket["id"],
                  input_shape=list(packed.shape), position_shape=list(positions.shape),
                  segments=len(offsets) - 1, offsets_head=offsets[:8], total_tokens=offsets[-1])
        return dict(model_inputs, **{TICKET_KEY: ticket})


def _wrap_kernel(original, state, layer_name, kind):
    native = "seq_idx" if kind == "conv" else "cu_seqlens"
    signature = inspect.signature(original)

    @wraps(original)
    def call(*args, **kwargs):
        current = _CALL.get()
        if current is None or current[0]["owner"] is not state or current[1] != layer_name:
            raise RuntimeError("GDN kernel ran without this actor's packed guard context.")
        ticket, _, phase = current
        actual = signature.bind_partial(*args, **kwargs).arguments.get(native)
        expected = ticket["seq" if kind == "conv" else "cu"] if state.arm == "boundary" else None
        if actual is not expected:
            raise RuntimeError(f"Actual {kind} {native} does not match the {state.arm} arm.")
        ticket["seen"].add((layer_name, phase, kind))
        mode = "train" if ticket["train"] else "inference"
        state.counts[f"{mode}_{phase}_{kind}"] += 1
        state.emit("kernel_args", once=("kernel", mode, phase, kind), mode=mode, phase=phase,
                   layer=layer_name, kernel=kind, native_arg=native, present=actual is not None,
                   shape=list(actual.shape) if actual is not None else None,
                   dtype=str(actual.dtype) if actual is not None else None,
                   offsets_head=ticket["offsets"][:8], ticket=ticket["id"],
                   callable_module=inspect.unwrap(original).__module__)
        return original(*args, **kwargs)

    return call


def _wrap_decoder(original, state, name, linear):
    signature = inspect.signature(original)

    @wraps(original.__func__)
    def forward(self, *args, **kwargs):
        ticket = kwargs.pop(TICKET_KEY, None)
        if ticket is None or ticket["owner"] is not state:
            raise RuntimeError("Actor decoder did not receive its prepared packed guard ticket.")
        if "seq_idx" in kwargs or "cu_seq_lens_q" in kwargs:
            raise RuntimeError("Unexpected native decoder boundaries; refusing double injection.")
        bound = signature.bind_partial(*args, **kwargs)
        if bound.arguments.get("past_key_values") is not None or kwargs.get("use_cache", False):
            raise RuntimeError("Cached decoding is outside this actor overlay.")
        if not linear:
            return original(*args, **kwargs)
        recompute = state.in_backward()
        if (recompute and not ticket["calls"][name]) or (not recompute and ticket["calls"][name]):
            raise RuntimeError("Unexpected decoder call order for this prepared checkpoint ticket.")
        ticket["calls"][name] += 1
        phase = "recompute" if recompute else "forward"
        if state.arm == "boundary":
            kwargs["seq_idx"] = ticket["seq"]
            kwargs["cu_seq_lens_q"] = ticket["cu"]
        # The ticket is a per-call kwarg captured by non-reentrant checkpointing.
        # This context is re-entered by the decoder itself during recomputation.
        token = _CALL.set((ticket, name, phase))
        try:
            return original(*args, **kwargs)
        finally:
            _CALL.reset(token)

    return forward


def _install_model(model, state, decoder_type):
    if getattr(model.config, "model_type", None) != "qwen3_5":
        raise RuntimeError("This overlay supports only the audited dense Qwen3.5 actor.")
    if getattr(model, "_shx_packed_boundary", None) is not None:
        raise RuntimeError("Model boundary overlay already installed.")
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, decoder_type)]
    linear_names = []
    if not layers:
        raise RuntimeError("No actual Qwen3.5 decoder layers found before FSDP wrapping.")
    for name, layer in layers:
        if not getattr(layer, "gradient_checkpointing", False):
            raise RuntimeError("Expected production gradient checkpointing to remain enabled.")
        checkpoint = getattr(layer, "_gradient_checkpointing_func", None)
        if getattr(checkpoint, "keywords", {}).get("use_reentrant") is not False:
            raise RuntimeError("Expected the production non-reentrant checkpoint function, unchanged.")
        linear = layer.layer_type == "linear_attention"
        if linear:
            attn = layer.linear_attn
            _check_kernel(attn.causal_conv1d_fn, "seq_idx", "causal_conv1d")
            _check_kernel(attn.chunk_gated_delta_rule, "cu_seqlens", "fla.ops.gated_delta_rule")
            attn.causal_conv1d_fn = _wrap_kernel(attn.causal_conv1d_fn, state, name, "conv")
            attn.chunk_gated_delta_rule = _wrap_kernel(attn.chunk_gated_delta_rule, state, name, "gdn")
            linear_names.append(name)
        elif layer.layer_type != "full_attention":
            raise RuntimeError("Unsupported Qwen3.5 decoder layer type.")
        layer.forward = MethodType(_wrap_decoder(layer.forward, state, name, linear), layer)
    if not linear_names:
        raise RuntimeError("No GDN layers found in this actor.")
    state.layers = tuple(linear_names)
    model._shx_packed_boundary = state
    text_config = getattr(model.config, "text_config", model.config)
    vision_config = getattr(model.config, "vision_config", None)
    state.emit("model_wrapped", decoder_layers=len(layers), gdn_layers=len(linear_names),
               checkpoint="unchanged_non_reentrant", model_type=model.config.model_type,
               attention_selector=getattr(model.config, "_attn_implementation", None),
               text_attention_selector=getattr(text_config, "_attn_implementation", None),
               vision_attention_selector=getattr(vision_config, "_attn_implementation", None))
    return model


def _patch_engine(engine, state, torch, metadata_get, decoder_type):
    if getattr(engine, "_shx_packed_boundary", None) is not None:
        if engine._shx_packed_boundary.arm != state.arm:
            raise RuntimeError("Cannot switch arms inside one actor process.")
        return engine._shx_packed_boundary
    if engine.engine_config.forward_only:
        raise RuntimeError("Do not install the actor overlay on a reference/teacher engine.")
    if engine.engine_config.strategy != "fsdp" or engine.use_ulysses_sp or engine.ulysses_sequence_parallel_size != 1:
        raise RuntimeError("Only the approved SP1 FSDP actor is supported.")
    if not engine.model_config.use_remove_padding or engine.model_config.use_fused_kernels:
        raise RuntimeError("Only remove-padding with the original unfused loss head is supported.")
    if not engine.model_config.enable_gradient_checkpointing:
        raise RuntimeError("Gradient checkpointing must remain enabled.")
    if getattr(engine, "_is_lora", False) or getattr(engine, "_qat_enabled", False):
        raise RuntimeError("LoRA/QAT are outside this isolated overlay.")
    if getattr(engine, "module", None) is not None:
        raise RuntimeError("Install on the actual engine before initialize()/FSDP wrapping.")
    original_build = engine._build_module
    original_prepare = engine.prepare_model_inputs
    original_batch = engine.forward_backward_batch

    @wraps(original_build.__func__)
    def build(self, *args, **kwargs):
        return _install_model(original_build(*args, **kwargs), state, decoder_type)

    @wraps(original_prepare.__func__)
    def prepare(self, micro_batch):
        if not metadata_get(micro_batch, "use_remove_padding", True) or metadata_get(micro_batch, "use_fused_kernels", False):
            raise RuntimeError("Microbatch metadata changed the approved packed/unfused path.")
        model_inputs, output_args = original_prepare(micro_batch)
        offsets = micro_batch["input_ids"].offsets().detach().cpu().tolist()
        return state.prepare(model_inputs, offsets, torch), output_args

    @wraps(original_batch.__func__)
    def batch(self, data, loss_function, forward_only=False):
        if forward_only:
            return original_batch(data, loss_function, forward_only=forward_only)
        with state.training_batch():
            return original_batch(data, loss_function, forward_only=forward_only)

    engine._build_module = MethodType(build, engine)
    engine.prepare_model_inputs = MethodType(prepare, engine)
    engine.forward_backward_batch = MethodType(batch, engine)
    engine._shx_packed_boundary = state
    state.emit("actor_engine_installed", engine_class=type(engine).__module__ + "." + type(engine).__qualname__,
               sp=1, fused_head=False, production_loss="unchanged")
    return state


def install_engine(engine, *, arm=None, audit_dir=None):
    """Direct entry for an isolated probe using the real, not-yet-built engine."""
    arm, audit_dir = _options(arm, audit_dir)
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    if not isinstance(engine, FSDPEngineWithLMHead):
        raise RuntimeError("Registry returned an unexpected actor engine class.")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if not hasattr(torch._C, "_current_graph_task_id"):
        raise RuntimeError("Runtime lacks the read-only autograd phase observer; cannot audit recomputation.")
    state = Audit(arm, audit_dir, rank, in_backward=lambda: torch._C._current_graph_task_id() >= 0)
    return _patch_engine(engine, state, torch, tu.get_non_tensor_data, Qwen3_5DecoderLayer)


def _patch_registry(registry, arm, audit_dir, installer):
    previous = getattr(registry, "_shx_packed_boundary_options", None)
    if previous is not None:
        if previous != (arm, audit_dir):
            raise RuntimeError("Worker setup invoked with conflicting boundary arm/audit directory.")
        return
    original = registry.new.__func__

    @classmethod
    @wraps(original)
    def new(cls, model_type, backend, *args, **kwargs):
        engine = original(cls, model_type, backend, *args, **kwargs)
        if model_type == "language_model" and not engine.engine_config.forward_only:
            if backend != "fsdp":
                raise RuntimeError("Requested actor backend is outside the SP1 FSDP overlay.")
            installer(engine, arm=arm, audit_dir=audit_dir)
        return engine

    registry.new = new
    registry._shx_packed_boundary_options = (arm, audit_dir)


def setup_worker():
    """Install once per Ray worker, before TrainingWorker constructs its engine."""
    arm, audit_dir = _options()
    from verl.workers.engine import EngineRegistry

    _patch_registry(EngineRegistry, arm, audit_dir, install_engine)
    print("SHX_PACKED_BOUNDARY " + json.dumps(dict(
        api=API_VERSION, event="worker_setup_installed", arm=arm, pid=os.getpid(),
        registry=EngineRegistry.__module__ + "." + EngineRegistry.__qualname__,
    ), sort_keys=True), flush=True)

"""Isolated, opt-in placement and evidence hooks. Never import on production entry."""
import json
import functools
import importlib
import importlib.util
import inspect
import math
import numbers
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import sys
import time


NNODES = 4
GPUS_PER_NODE = 8
POOL_SHAPE = (4,) * NNODES
ACTOR_WORLD_SIZE = sum(POOL_SHAPE)


def wandb_environment(env):
    # Read only literal W&B exports, never execute the training script or publish its key.
    source = Path(env["DLC_WANDB_ENV_FILE"])
    require(source.is_absolute() and source.is_file(), "W&B export source is missing")
    settings = {}
    allowed = {"WANDB_BASE_URL", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"}
    for line in source.read_text().splitlines():
        if not re.match(r"\s*export\s+WANDB_(?:BASE_URL|API_KEY|ENTITY|PROJECT)=", line):
            continue
        parts = shlex.split(line, comments=True)
        require(len(parts) == 2, "W&B exports must be literal assignments")
        key, _, value = parts[1].partition("=")
        require(key in allowed and value and "$" not in value and "`" not in value,
                "W&B exports must contain literal nonempty values")
        settings[key] = value
    require(all(settings.get(key) for key in ("WANDB_BASE_URL", "WANDB_API_KEY")),
            "The source must export W&B base URL and API key")
    from urllib.parse import urlsplit
    url = urlsplit(settings["WANDB_BASE_URL"])
    require(url.scheme == "https" and url.hostname and not url.username and not url.password,
            "Require an HTTPS W&B server URL without embedded credentials")
    settings.update(WANDB_MODE="online", WANDB_RESUME="never", WANDB_INIT_TIMEOUT="120",
                    WANDB_DIR=str(Path(env["DLC_TRAIN_OUTPUT_DIR"]) / "wandb"))
    return settings


def wandb_preflight(env):
    import requests
    settings = wandb_environment(env)
    for attempt in range(1, 6):
        try:
            response = requests.post(settings["WANDB_BASE_URL"].rstrip("/") + "/graphql",
                                     auth=("api", settings["WANDB_API_KEY"]),
                                     json={"query": "query { viewer { username } }"}, timeout=20)
            require(response.status_code not in {401, 403}, "W&B authentication rejected; credentials withheld")
            valid = response.ok and bool(response.json().get("data", {}).get("viewer", {}).get("username"))
            if valid:
                return {"authenticated": True, "base_url": settings["WANDB_BASE_URL"],
                        "mode": settings["WANDB_MODE"], "attempts": attempt}
            reason = 'HTTP_' + str(response.status_code)
        except (requests.RequestException, json.JSONDecodeError) as error:
            reason = type(error).__name__
        print(f"WANDB_PREFLIGHT_RETRY attempt={attempt}/5 reason={reason}", flush=True)
        if attempt < 5:
            time.sleep(5)
    raise RuntimeError("W&B connectivity preflight exhausted five attempts; credentials withheld")


def attach_teacher_with_timeout(original, trainer, batch, future, timing_raw, output, timeout=600):
    started = time.monotonic()
    record = {"step": trainer.global_steps, "stage": "teacher_wait", "timeout_seconds": timeout,
              "started_unix": time.time(), "status": "start"}

    def mark(status):
        record.update(status=status, elapsed_seconds=time.monotonic() - started, updated_unix=time.time())
        write_json(Path(output) / "teacher-wait.json", record)
        print("TEACHER_WAIT " + json.dumps(record), flush=True)

    class BoundedFuture:
        def result(self):
            return future.result(timeout=max(0, timeout - (time.monotonic() - started)))

    mark("start")
    try:
        result = original(trainer, batch, BoundedFuture() if future is not None else None, timing_raw)
    except TimeoutError as error:
        mark("timeout")
        raise TimeoutError(f"Teacher future exceeded {timeout}s at step {trainer.global_steps}; no retry/resume") from error
    except BaseException:
        mark("error")
        raise
    mark("complete")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_nodes(nodes):
    nodes = sorted((n for n in nodes if n.get("Alive")), key=lambda n: n["NodeManagerAddress"])
    require(len(nodes) == NNODES, "Exactly four live Ray nodes are required; no shared/existing cluster")
    require(len({n["NodeID"] for n in nodes}) == NNODES, "Duplicate Ray node IDs")
    require(len({n["NodeManagerAddress"] for n in nodes}) == NNODES, "Nodes must have distinct addresses")
    for node in nodes:
        resources = node["Resources"]
        require(resources.get("GPU") == GPUS_PER_NODE and resources.get("CPU") == 184,
                "Each Ray node must expose the approved 184 CPU / 8 GPU allocation")
        key = "node:" + node["NodeManagerAddress"]
        require(resources.get(key, 0) >= 0.008, "Missing Ray built-in node affinity resource")
        require(sum(other["Resources"].get(key, 0) > 0 for other in nodes) == 1,
                "Node affinity resource is not unique")
    return nodes


def pinned_bundles(pool, node):
    require(pool.name_prefix in {"global_pool", "teacher_pool"}, "Unexpected resource pool")
    require(tuple(pool._store) == POOL_SHAPE and pool.use_gpu and not pool.detached,
            "Only non-detached GPU pools [4,4,4,4] are supported")
    bundle = {"CPU": pool.max_colocate_count, "GPU": 1,
              "node:" + node["NodeManagerAddress"]: 0.001}
    if pool.accelerator_type is not None:
        bundle[pool.accelerator_type] = 0.0001
    return [bundle.copy() for _ in range(4)]


def verify_assignment(table, expected_id):
    assignments = table.get("bundles_to_node_id", {})
    require(table.get("state") == "CREATED", "Placement group was not created")
    require(len(assignments) == 4 and set(assignments.values()) == {expected_id},
            "Placement audit failed: four bundles must be on their explicit target node")
    return {str(index): node_id for index, node_id in assignments.items()}


def install_placement(output_dir):
    import ray
    from ray.util.placement_group import placement_group, placement_group_table
    from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager

    nodes = select_nodes(ray.nodes())
    audit = {"nodes": nodes, "pools": {}}
    original_create = ResourcePoolManager.create_resource_pool

    def get_groups(pool, strategy="STRICT_PACK", name=None, device_name="cuda"):
        if pool.pgs is not None:
            return pool.pgs
        require(strategy == "STRICT_PACK" and device_name == "cuda", "Unexpected placement contract")
        pgs = [placement_group(pinned_bundles(pool, node), strategy="STRICT_PACK",
                               name=f"{pool.name_prefix}-explicit-{index}")
               for index, node in enumerate(nodes)]
        ray.get([pg.ready() for pg in pgs], timeout=120)
        audit["pools"][pool.name_prefix] = [
            {"node_id": node["NodeID"], "bundles": verify_assignment(placement_group_table(pg), node["NodeID"])}
            for pg, node in zip(pgs, nodes)
        ]
        write_json(Path(output_dir) / "placement.json", audit)
        pool.pgs = pgs
        return pgs

    def create_pools(manager):
        require(manager.resource_pool_spec == {"global_pool": list(POOL_SHAPE), "teacher_pool": list(POOL_SHAPE)},
                "Unexpected pool topology; refusing to change global batch for divisibility")
        original_create(manager)
        # Reserve and audit both pools before any model initialization.
        for pool in manager.resource_pool_dict.values():
            pool.get_placement_groups()
        print("SMOKE_PLACEMENT_OK actor=[4,4,4,4] teacher=[4,4,4,4] actor_world_size=16", flush=True)

    RayResourcePool.get_placement_groups = get_groups
    ResourcePoolManager.create_resource_pool = create_pools


def checkpoint_space(path, checkpoint_bytes, minimum_free_gib):
    require(checkpoint_bytes > 0, "Missing model-only checkpoint size estimate")
    free = shutil.disk_usage(path).free
    # Retain the previous two-write safety margin on the shared filesystem.
    required = max(int(minimum_free_gib * 2**30), 2 * checkpoint_bytes + 8 * 2**30)
    result = {"free_bytes": free, "required_free_bytes": required,
              "estimated_checkpoint_bytes": checkpoint_bytes,
              "estimated_all_10_model_checkpoints_bytes": checkpoint_bytes * 10,
              "estimate_basis": "2x base safetensors bytes for FP32 FSDP model shards; excludes small metadata",
              "retention": "all_ten_baseline_checkpoints_no_deletion"}
    require(free >= required, f"Checkpoint disk gate: free={free} required={required}; no files deleted")
    return result


def checkpoint_files(path):
    path = Path(path)
    models = sorted(path.glob("actor/model_world_size_*_rank_*.pt"))
    expected = {f"model_world_size_{ACTOR_WORLD_SIZE}_rank_{rank}.pt" for rank in range(ACTOR_WORLD_SIZE)}
    require({p.name for p in models} == expected and all(p.stat().st_size > 0 for p in models),
            "Checkpoint must contain exactly sixteen nonempty model shards")
    forbidden = [p for p in path.rglob("*") if p.is_file() and
                 (p.name.startswith(("optim_", "extra_state_", "pytorch_model")) or p.suffix == ".safetensors")]
    require(not forbidden, f"Unexpected optimizer/extra/HF weights in model-only checkpoint: {forbidden}")
    return {"model_files": [str(p.relative_to(path)) for p in models],
            "model_bytes": sum(p.stat().st_size for p in models),
            "small_metadata_allowed": True, "optimizer_or_extra": False}


def callable_target(target):
    module_name, separator, attribute = target.partition(":")
    require(separator and module_name and attribute, "FA3 target must be verified module:callable")
    owner = importlib.import_module(module_name)
    parts = attribute.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    original = getattr(owner, parts[-1])
    require(callable(original), f"Not callable: {target}")
    return owner, parts[-1], original


def load_fa3_provider(controls):
    selector = controls["fa3_backend"]
    require(re.fullmatch(r"kernels-community/flash-attn3@[0-9a-f]{40}", selector), "Pinned FA3 provider required")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "Provider must load offline from the shared cache")
    from huggingface_hub import constants
    require(constants.HF_HUB_OFFLINE, "HF was imported before the offline environment was applied")
    from kernels import get_kernel
    repo, revision = selector.split("@")
    module = get_kernel(repo, revision=revision)
    source = Path(module.__file__).resolve()
    require(any(source.is_relative_to(Path(path).resolve()) for path in controls["caches"].values()),
            "FA3 provider did not load from the supplied shared cache")
    return {"selector": selector, "module": module.__name__, "source": str(source)}


def setup_boundary(controls):
    require(os.environ.get("SHX_PACKED_BOUNDARY_ARM") == controls["arm"]
            and os.environ.get("SHX_PACKED_AUDIT_DIR") == controls["audit_dir"], "Boundary child env mismatch")
    name = "shx_packed_boundary"
    require(name not in sys.modules, "Boundary module already loaded outside the isolated actor setup")
    spec = importlib.util.spec_from_file_location(name, controls["boundary_module"])
    require(spec is not None and spec.loader is not None, "Cannot load supplied boundary module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(module.API_VERSION == 1, "Unsupported boundary audit API")
    module.setup_worker()


def boundary_evidence(audit_dir, arm, rank, pid, hostname):
    path = Path(audit_dir) / f"actor-{hostname}-{pid}-rank{rank}.jsonl"
    require(path.is_file(), f"No actual boundary engine evidence: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(all(record["api"] == 1 and record["arm"] == arm and record["rank"] == rank
                and record["pid"] == pid for record in records), "Wrong boundary audit identity/API")
    events = {record["event"] for record in records}
    require("audit_failure" not in events, "Boundary audit reported a failure")
    require({"actor_engine_installed", "model_wrapped", "backward_batch_complete"} <= events,
            "Boundary hook installation alone is not successful backward evidence")
    guards = [record for record in records if record["event"] == "packed_guard" and record["mode"] == "train"]
    require(guards and all(record["position_shape"] == [4, 1, record["total_tokens"]] for record in guards),
            "Actual training packed-position guard did not pass")
    kernels = [record for record in records if record["event"] == "kernel_args" and record["mode"] == "train"]
    expected = {(phase, kind) for phase in ("forward", "recompute") for kind in ("conv", "gdn")}
    require({(record["phase"], record["kernel"]) for record in kernels} == expected,
            "Missing actual convolution/GDN forward or checkpoint-recompute calls")
    require(all(record["present"] == (arm == "boundary") for record in kernels), "Boundary injection arm mismatch")
    return {"audit_file": str(path), "packed_position_checks": len(guards),
            "boundary_installed": arm == "boundary", "boundary_calls": len(kernels) if arm == "boundary" else 0,
            "boundary_setup": True, "boundary_evidence_scope": "first_completed_backward_batch; module checks every batch"}


def verify_actor_evidence(output, arm, minimum_calls=1):
    records = []
    for rank in range(ACTOR_WORLD_SIZE):
        path = Path(output) / f"actor-runtime-rank-{rank}.json"
        require(path.is_file(), f"Missing actual actor runtime evidence for rank {rank}")
        record = json.loads(path.read_text())
        require(record["rank"] == rank and record["arm"] == arm, "Wrong rank/arm evidence")
        require(record["training_calls"] >= minimum_calls and record["fa3_forward"] > 0 and record["fa3_backward"] > 0,
                f"Rank {rank} lacks successful training FA3 forward/backward calls")
        require(record["packed_position_checks"] > 0, f"Rank {rank} never entered the input guard")
        require(record["boundary_setup"] and record["boundary_installed"] == (arm == "boundary"), "Boundary hook arm mismatch")
        require(arm == "baseline" or record["boundary_calls"] > 0,
                f"Rank {rank} installed the boundary hook but never called it")
        records.append(record)
    return records


def install_actor_hooks(output, controls):
    require(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "0" and os.environ.get("FLA_TILELANG") == "1",
            "Approved FLA flags must precede actor imports")
    provider = load_fa3_provider(controls)
    state = {"active": False, "training_calls": 0, "fa3_forward": 0, "fa3_backward": 0,
             "packed_position_checks": 0, "boundary_installed": False, "boundary_calls": 0,
             "arm": controls["arm"], "fa3_provider": provider, "fa3_targets": {}}

    def instrument(target, counter):
        owner, name, original = callable_target(target)
        try:
            source = inspect.getfile(original)
        except TypeError:
            source = getattr(inspect.getmodule(original), "__file__", None)
        state["fa3_targets"][counter] = {"target": target, "source": source}

        @functools.wraps(original)
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if state["active"]:
                state[counter] += 1
            return result

        setattr(owner, name, observed)

    instrument(controls["fa3_forward_target"], "fa3_forward")
    instrument(controls["fa3_backward_target"], "fa3_backward")
    setup_boundary(controls)
    import torch
    from verl.workers.engine_workers import TrainingWorker
    require(not getattr(TrainingWorker.train_mini_batch, "_v3_instrumented", False), "Actor hooks already installed")
    original_train = TrainingWorker.train_mini_batch

    @functools.wraps(original_train)
    def train_mini_batch(worker, *args, **kwargs):
        before = (state["fa3_forward"], state["fa3_backward"])
        state["active"] = True
        try:
            result = original_train(worker, *args, **kwargs)
        finally:
            state["active"] = False
        state["training_calls"] += 1
        state["rank"] = torch.distributed.get_rank()
        state["hostname"] = socket.gethostname()
        state.update(boundary_evidence(controls["audit_dir"], controls["arm"], state["rank"], os.getpid(), state["hostname"]))
        write_json(Path(output) / f"actor-runtime-rank-{state['rank']}.json", state)
        require(all(state[key] > value for key, value in zip(
            ("fa3_forward", "fa3_backward"), before)), "Actor update returned without actual FA3 forward/backward calls")
        if state["training_calls"] == 1:
            print("ACTOR_RUNTIME_VERIFIED " + json.dumps(state), flush=True)
        return result

    train_mini_batch._v3_instrumented = True
    TrainingWorker.train_mini_batch = train_mini_batch
    print(f"ACTOR_HOOKS_INSTALLED arm={controls['arm']} actual_call_evidence=pending", flush=True)


def task_runner_class(output_dir, prepared, baseline):
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import TaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.tracking import Tracking

    output = Path(output_dir)
    controls = prepared["experiment"]

    class ExperimentTaskRunner(TaskRunner):
        def add_actor_rollout_worker(self, config):
            from verl.workers import engine_workers
            original = engine_workers.ActorRolloutRefWorker

            class InstrumentedActor(original):
                def __init__(self, *args, **kwargs):
                    install_actor_hooks(output_dir, controls)
                    super().__init__(*args, **kwargs)

            # Install inside the actual worker constructor, not a conditionally inherited setup hook.
            engine_workers.ActorRolloutRefWorker = InstrumentedActor
            try:
                return super().add_actor_rollout_worker(config)
            finally:
                engine_workers.ActorRolloutRefWorker = original

        def run(self, config):
            os.environ.update(wandb_environment(os.environ))
            os.environ.pop("WANDB_RUN_ID", None)
            Path(os.environ["WANDB_DIR"]).mkdir(exist_ok=True)
            install_placement(output)
            counts = {"teacher": 0, "update": 0, "logged_steps": [], "checkpoints": []}
            original_init = RayPPOTrainer.init_workers
            original_teacher = RayPPOTrainer._attach_async_teacher_logprobs
            original_update = RayPPOTrainer._update_actor
            original_save = RayPPOTrainer._save_checkpoint
            original_log = Tracking.log
            original_tracking_init = Tracking.__init__
            (output / "updates").mkdir(exist_ok=False)

            def tracking_init(tracker, *args, **kwargs):
                original_tracking_init(tracker, *args, **kwargs)
                run = tracker.logger["wandb"].run
                require(run is not None and run.settings.mode == "online", "Expected a live online W&B run")
                write_json(output / "wandb-run.json", {"id": run.id, "name": run.name,
                           "project": run.project, "entity": run.entity, "url": run.url,
                           "mode": run.settings.mode, "dir": run.dir})

            def init_workers(trainer):
                require(trainer.total_training_steps == 100 and not trainer.use_critic, "Expected 100-step actor-only OPD")
                optim = trainer.config.actor_rollout_ref.actor.optim
                require(optim.total_training_steps == 100 and optim.lr == 1e-6 and optim.lr_scheduler_type == "constant",
                        "Unexpected optimizer horizon/LR; no schedule repair or silent changes")
                require(optim.lr_warmup_steps in {-1, 0} and optim.lr_warmup_steps_ratio == 0, "Warmup must remain zero")
                baseline_steps = baseline["trainer"]["total_training_steps"]
                if baseline_steps is None:
                    baseline_steps = len(trainer.train_dataloader) * baseline["trainer"]["total_epochs"]
                write_json(output / "schedule.json", {"trainer_stop_steps": 100, "optimizer_horizon": 100,
                    "baseline_horizon": baseline_steps, "lr": 1e-6, "scheduler": "constant", "warmup_steps": 0,
                    "note": "Batch32/mini16, n8; fresh four-node baseline, constant schedule, zero warmup"})
                write_json(output / "runtime_config.json", OmegaConf.to_container(trainer.config, resolve=True))
                return original_init(trainer)

            def teacher(trainer, batch, future, timing_raw):
                result = attach_teacher_with_timeout(original_teacher, trainer, batch, future, timing_raw, output)
                require("teacher_logprobs" in result.batch and "teacher_neg_logprobs" in result.batch,
                        "Both positive and negative teacher results are required")
                counts["teacher"] += 1
                write_json(output / "stages.json", counts)
                return result

            def update(trainer, batch):
                require(trainer.global_steps == counts["update"] + 1 <= 100, "Unexpected training step or budget overrun")
                require(len(batch) == 256, "Expected B32 x n8; never resize the global batch")
                result = original_update(trainer, batch)
                counts["update"] += 1
                verify_actor_evidence(output, controls["arm"], counts["update"])
                write_json(output / "stages.json", counts)
                print(f"ACTOR_UPDATE_VERIFIED step={trainer.global_steps} fa3_ranks={ACTOR_WORLD_SIZE}", flush=True)
                return result

            def save(trainer):
                step = trainer.global_steps
                require(step in range(10, 101, 10) and step not in counts["checkpoints"], "Unexpected/duplicate checkpoint step")
                actor = trainer.config.actor_rollout_ref.actor
                require(list(actor.checkpoint.save_contents) == ["model"] and not actor.checkpoint.get("async_save", False),
                        "Only synchronous model shards may be saved; no optimizer/extra")
                require(trainer.config.trainer.max_actor_ckpt_to_keep is None
                        and not trainer.config.trainer.remove_previous_ckpt_in_save, "Automatic deletion is forbidden")
                space = checkpoint_space(output, prepared["checkpoint_estimate_bytes"], controls["minimum_free_gib"])
                write_json(output / "checkpoint-space-latest.json", dict(space, step=step))
                result = original_save(trainer)
                evidence = checkpoint_files(Path(trainer.config.trainer.default_local_dir) / f"global_step_{step}")
                counts["checkpoints"].append(step)
                write_json(output / f"checkpoint-{step:03d}.json", dict(evidence, step=step, disk_before=space))
                write_json(output / "stages.json", counts)
                return result

            def log(tracker, data, step, *args, **kwargs):
                result = original_log(tracker, data, step, *args, **kwargs)
                if "training/global_step" in data:
                    require(step == len(counts["logged_steps"]) + 1 <= 100, "Unexpected logged training step")
                    counts["logged_steps"].append(step)
                    record = {"step": step, "arm": controls["arm"], "metrics": data,
                              "nonfinite_metric_keys": [key for key, value in data.items()
                                  if isinstance(value, numbers.Real) and not math.isfinite(value)]}
                    write_json(output / "updates" / f"step-{step:03d}.json", record)
                    write_json(output / "update_result.json", record)
                return result

            RayPPOTrainer.init_workers = init_workers
            RayPPOTrainer._attach_async_teacher_logprobs = teacher
            RayPPOTrainer._update_actor = update
            RayPPOTrainer._save_checkpoint = save
            Tracking.log = log
            Tracking.__init__ = tracking_init
            super().run(config)
            require(counts == {"teacher": 100, "update": 100, "logged_steps": list(range(1, 101)),
                               "checkpoints": list(range(10, 101, 10))}, "Incomplete training/checkpoint history")
            evidence = verify_actor_evidence(output, controls["arm"], 100)
            write_json(output / "training_complete.json", {"counts": counts, "arm": controls["arm"],
                       "actual_fa3_ranks": len(evidence), "precision_pass": False})
            print(f"TRAINING_COMPLETE steps=100 model_checkpoints=10 actual_fa3_ranks={ACTOR_WORLD_SIZE} precision_pass=false", flush=True)

    return ray.remote(num_cpus=1)(ExperimentTaskRunner)

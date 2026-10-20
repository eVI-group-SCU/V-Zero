#!/usr/bin/env python3
"""Prepare or run one 100-step FA3 comparison arm. Never submit a job."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pai_dlc_train_overlay import checkpoint_space, require, select_nodes, wandb_preflight, write_json
from teacher_scheduler import SCHEDULER_CLASS

PACKAGES = ("ray", "torch", "vllm", "transformers", "flash-attn", "fla-core", "flash-linear-attention", "verl")
RUNTIME_KEYS = ("CUDA_HOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "LIBRARY_PATH", "PATH", "FLA_DISABLE_BACKEND_DISPATCH", "FLA_TILELANG",
                "VERL_PRINT_STEP_PROGRESS", "VERL_TEACHER_LOGPROB_STREAMING_MODE", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "DLC_TRAIN_PYTHON",
                "DLC_PYTHON_BASE", "DLC_FA3_ENV_ROOT", "DLC_TRAIN_OUTPUT_DIR", "SHX_PACKED_BOUNDARY_ARM", "DLC_TRAIN_STEPS", "DLC_SAVE_FREQ",
                "DLC_CHECKPOINT_MIN_FREE_GIB", "REW_OPD_ENABLED", "DLC_FA3_BACKEND",
                "DLC_FA3_FORWARD_TARGET", "DLC_FA3_BACKWARD_TARGET", "DLC_BOUNDARY_MODULE",
                "SHX_PACKED_AUDIT_DIR", "HF_HOME", "HF_HUB_CACHE", "KERNELS_CACHE",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DLC_CUDA_COMPAT_DIR", "DLC_WANDB_ENV_FILE")
FROZEN = {"data.train_batch_size": 32, "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
          "actor_rollout_ref.rollout.n": 8}
APPROVED_FLA = {"FLA_DISABLE_BACKEND_DISPATCH": "0", "FLA_TILELANG": "1"}
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def approved_fla_flags(env):
    flags = {key: env.get(key) for key in APPROVED_FLA}
    require(flags == APPROVED_FLA, f"Approved isolated FLA flags 0/1 required before imports; observed {flags}")
    return flags


def merge_library_paths(*values):
    return os.pathsep.join(dict.fromkeys(part for value in values for part in value.split(os.pathsep) if part))


def child_runtime(env, prepared):
    approved_fla_flags(env)
    # Reject stale snapshots before they can override the approved process flags.
    approved_fla_flags(prepared["runtime_env"])
    require(experiment(env) == prepared["experiment"], "Run controls changed after prepare")
    require(experiment(prepared["runtime_env"]) == prepared["experiment"], "Stale prepared run controls")
    runtime = {**env, **prepared["runtime_env"], **OFFLINE,
               "SHX_PACKED_AUDIT_DIR": prepared["experiment"]["audit_dir"]}
    # Container driver libraries and a node's selected compat path must survive CPU preparation.
    runtime["LD_LIBRARY_PATH"] = merge_library_paths(env.get("LD_LIBRARY_PATH", ""),
                                                    prepared["runtime_env"].get("LD_LIBRARY_PATH", ""))
    if "LD_PRELOAD" in env:
        runtime["LD_PRELOAD"] = env["LD_PRELOAD"]
    return runtime


def ray_job_runtime_env(env, base):
    flags = approved_fla_flags(env)
    controls = experiment(env)
    runtime = copy.deepcopy(base)
    variables = runtime.setdefault("env_vars", {})
    require(all(key not in variables or variables[key] == value for key, value in flags.items()),
            "Ray runtime environment conflicts with the approved FLA flags")
    variables.update(flags)
    variables.update({key: env[key] for key in RUNTIME_KEYS if key in env})
    # Each Raylet starts with its own validated LD. A head-wide override can break a different driver node.
    variables.pop("LD_LIBRARY_PATH", None)
    variables.pop("LD_PRELOAD", None)
    variables.update(OFFLINE)
    variables["SHX_PACKED_AUDIT_DIR"] = controls["audit_dir"]
    return runtime


def required(env, key):
    value = env.get(key, "")
    require(bool(value), f"Export {key} explicitly; no production/legacy path fallback")
    return value


def training_python(env):
    path = Path(required(env, "DLC_TRAIN_PYTHON"))
    require(path.is_absolute() and path.parent.name == "bin", "DLC_TRAIN_PYTHON must be an absolute shared venv interpreter")
    return path


def experiment(env):
    require(required(env, "DLC_TRAIN_STEPS") == "100", "Only 100 training steps are authorized")
    require(required(env, "DLC_SAVE_FREQ") == "10", "Save every ten steps; preserve all ten checkpoints")
    require(required(env, "REW_OPD_ENABLED").lower() == "true", "Positive AND negative evidence must stay enabled")
    arm = required(env, "SHX_PACKED_BOUNDARY_ARM")
    require(arm in {"baseline", "boundary"}, "Unknown boundary arm")
    minimum = float(required(env, "DLC_CHECKPOINT_MIN_FREE_GIB"))
    require(48 <= minimum < float("inf"), "Checkpoint free-space floor must be at least 48 GiB")
    backend = required(env, "DLC_FA3_BACKEND")
    require(re.fullmatch(r"kernels-community/flash-attn3@[0-9a-f]{40}", backend),
            "Require the verified flash-attn3 provider with its immutable 40-hex revision")
    require(all(env.get(key, value) == value for key, value in OFFLINE.items()), "DLC must use offline shared caches")
    caches = {key: required(env, key) for key in ("HF_HOME", "KERNELS_CACHE")}
    if env.get("HF_HUB_CACHE"):
        caches["HF_HUB_CACHE"] = env["HF_HUB_CACHE"]
    require(all(Path(value).is_absolute() for value in caches.values()), "Use absolute shared cache paths")
    audit = str(Path(required(env, "DLC_TRAIN_OUTPUT_DIR")) / "actor-boundary")
    require(env.get("SHX_PACKED_AUDIT_DIR", audit) == audit, "Boundary audit must stay in this arm's output")
    compat = env.get("DLC_CUDA_COMPAT_DIR", "")
    require(not compat or Path(compat).is_absolute(), "DLC_CUDA_COMPAT_DIR must be an explicit absolute shared path")
    return {"arm": arm, "steps": 100, "save_freq": 10, "minimum_free_gib": minimum,
            "python": str(training_python(env)), "python_base": required(env, "DLC_PYTHON_BASE"),
            "env_root": required(env, "DLC_FA3_ENV_ROOT"),
            "fa3_backend": backend, "fa3_forward_target": required(env, "DLC_FA3_FORWARD_TARGET"),
            "fa3_backward_target": required(env, "DLC_FA3_BACKWARD_TARGET"),
            "boundary_module": required(env, "DLC_BOUNDARY_MODULE"), "audit_dir": audit, "caches": caches,
            "cuda_compat_dir": compat}


def cuda_probe(result_path):
    """Fresh-process, eight-device gate only; never run in the Ray driver process."""
    import ctypes
    result = {"passed": False, "hostname": socket.gethostname(), "pid": os.getpid(),
              "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
              "ld_preload": os.environ.get("LD_PRELOAD", "")}
    kernel = Path("/proc/driver/nvidia/version")
    match = re.search(r"Kernel Module\s+(\d+)\.", kernel.read_text() if kernel.is_file() else "")
    result["kernel_driver_branch"] = int(match.group(1)) if match else None
    try:
        import torch
        result.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
        require(torch.cuda.is_available(), "Torch CUDA unavailable; refusing CPU/NCCL fallback")
        result["device_count"] = torch.cuda.device_count()
        require(result["device_count"] == 8, "Exactly eight visible GPUs required before Ray starts")
        for device in range(8):
            value = torch.ones(1, device=f"cuda:{device}")
            require((value + 1).item() == 2.0, f"CUDA tensor check failed on device {device}")
            del value
        result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    try:
        version = ctypes.c_int()
        code = ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version))
        result.update(cuda_driver_api=version.value, driver_query_result=code)
    except Exception as error:
        result["driver_query_error"] = str(error)
    maps = Path("/proc/self/maps")
    result["libcuda_paths"] = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                                      if "libcuda.so" in line}) if maps.is_file() else []
    write_json(result_path, result)
    print("CUDA_NODE_PROBE " + json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def select_cuda_runtime(env, execution, rank):
    """Try inherited/container libraries first, then only the explicit supported fallback."""
    def probe(runtime, name):
        path = Path(execution) / f"cuda-{name}-{rank}-{uuid.uuid4().hex[:8]}.json"
        completed = subprocess.run([str(training_python(runtime)), "-I", "-B", str(Path(__file__).resolve()),
                                    "_cuda_probe", str(path)], env=runtime, timeout=120)
        require(path.is_file(), f"CUDA {name} probe exited without evidence (code {completed.returncode})")
        evidence = json.loads(path.read_text())
        require(completed.returncode == (0 if evidence.get("passed") else 1), "CUDA probe status/evidence mismatch")
        return evidence

    selected = dict(env)
    native = probe(selected, "native")
    evidence = {"selection": "inherited", "native": native}
    if not native["passed"]:
        directory = Path(required(env, "DLC_CUDA_COMPAT_DIR"))
        require(directory.is_absolute() and directory.resolve().is_relative_to(Path(required(env, "DLC_SHARED_MOUNT")).resolve()),
                "CUDA compat fallback must be an explicit shared directory")
        require(native.get("torch_cuda") == "13.0" and native.get("kernel_driver_branch") in {535, 570},
                "Do not apply CUDA 13.0 compat to newer/unknown kernel drivers or another CUDA runtime")
        for name in ("libcuda.so.1", "libnvidia-ptxjitcompiler.so.1", "libnvidia-nvvm.so.4"):
            library = directory / name
            require(library.is_file() and library.resolve().is_relative_to(directory.resolve()), "Incomplete/escaping CUDA compat library")
        require((directory / "libcuda.so.1").resolve().name.startswith("libcuda.so.580."), "Require the supplied CUDA 13.0 / 580 compat family")
        selected["LD_LIBRARY_PATH"] = merge_library_paths(str(directory), env.get("LD_LIBRARY_PATH", ""))
        fallback = probe(selected, "compat")
        require(fallback["passed"] and any(Path(path).is_relative_to(directory.resolve()) for path in fallback["libcuda_paths"]),
                "Explicit CUDA compat fallback did not pass using its own libcuda; no Ray/model startup")
        evidence.update(selection="compat", fallback=fallback)
    evidence["selected_ld_library_path"] = selected.get("LD_LIBRARY_PATH", "")
    write_json(Path(execution) / f"cuda-selected-{rank}.json", evidence)
    print(f"CUDA_NODE_READY rank={rank} selection={evidence['selection']} devices=8", flush=True)
    return selected


def get(config, path):
    for part in path.split("."):
        config = config[part]
    return config


def put(config, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        config = config.setdefault(part, {})
    config[parts[-1]] = value


def changes(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = {}
        for key in before.keys() | after.keys():
            result.update(changes(before.get(key), after.get(key), prefix + ("." if prefix else "") + key))
        return result
    return {} if before == after else {prefix: {"before": before, "after": after}}


def validate_training(config):
    for key, value in FROZEN.items():
        require(get(config, key) == value, f"Frozen global configuration changed: {key}")
    actor = config["actor_rollout_ref"]["actor"]
    require(actor["strategy"] == "fsdp", "The audited boundary module requires the production FSDP actor")
    require(actor["fsdp_config"]["ulysses_sequence_parallel_size"] == 1,
            "This authorized topology requires SP=1, DP=8; do not silently change SP")
    require(actor["fsdp_config"]["fsdp_size"] in {-1, 8}, "FSDP sharding must span all eight actor ranks")
    require(actor["ppo_mini_batch_size"] * 8 % 8 == 0 and 32 * 8 % 8 == 0, "Invalid DP divisibility")
    distillation = config["distillation"]
    require(distillation["enabled"] and not distillation["teacher_ema"]["enabled"], "Unexpected OPD mode")
    require(len(distillation["teacher_models"]) == 1, "Only the production single teacher is supported")
    require(not distillation["distillation_loss"]["use_task_rewards"], "Expected pure OPD objective")
    require(distillation["evidence_weighting"]["enabled"] is True, "Relative evidence must be enabled in both arms")
    require(distillation["teacher_crop"]["enabled"] is True, "Positive teacher crop must be enabled")
    require(distillation["evidence_weighting"]["negative_mode"] == "hard_negative_crop", "Negative crop must be preserved")
    optim = actor["optim"]
    require(optim["lr"] == 1e-6 and optim["lr_scheduler_type"] == "constant", "Require constant 1e-6 actor LR")
    require(optim.get("lr_warmup_steps", -1) in {-1, 0} and optim.get("lr_warmup_steps_ratio", 0) == 0,
            "Require zero warmup; do not silently change schedule")
    inference_configs = [config["actor_rollout_ref"]["rollout"],
                         next(iter(distillation["teacher_models"].values()))["inference"]]
    for inference in inference_configs:
        width = (inference["tensor_model_parallel_size"] * inference.get("data_parallel_size", 1)
                 * inference.get("pipeline_model_parallel_size", 1))
        require(width > 0 and 4 % width == 0, "Inference TP/DP/PP replica must divide each four-GPU partition")


def training_config(baseline, run_dir, run_tag, controls):
    validate_training(baseline)
    config = copy.deepcopy(baseline)
    overrides = {
        "trainer.nnodes": 2, "trainer.n_gpus_per_node": 4,
        "distillation.nnodes": 2, "distillation.n_gpus_per_node": 4,
        "actor_rollout_ref.rollout.n_gpus_per_node": 4,
        "trainer.total_training_steps": 100, "trainer.save_freq": 10, "trainer.test_freq": -1,
        "trainer.val_before_train": False, "trainer.resume_mode": "disable", "trainer.logger": ["console", "wandb"],
        "trainer.max_actor_ckpt_to_keep": None, "trainer.max_critic_ckpt_to_keep": None,
        "trainer.remove_previous_ckpt_in_save": False, "trainer.default_hdfs_dir": None,
        "trainer.experiment_name": run_tag, "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.rollout_data_dir": str(run_dir / "rollouts"),
        "trainer.validation_data_dir": str(run_dir / "validation"),
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model"],
        "actor_rollout_ref.actor.checkpoint.async_save": False,
        "actor_rollout_ref.model.override_config.attn_implementation": controls["fa3_backend"],
    }
    for key, value in overrides.items():
        put(config, key, value)
    for key, teacher in config["distillation"]["teacher_models"].items():
        path = f"distillation.teacher_models.{key}.inference.engine_kwargs.vllm.scheduler_cls"
        existing = teacher["inference"].get("engine_kwargs", {}).get("vllm", {}).get("scheduler_cls")
        require(existing is None, "Do not replace an existing teacher scheduler override")
        overrides[path] = SCHEDULER_CLASS
        put(config, path, SCHEDULER_CLASS)
    # Resolve-derived tracing metadata follows the isolated run name, not production outputs.
    trace = config["actor_rollout_ref"]["rollout"].get("trace", {})
    if "experiment_name" in trace:
        key = "actor_rollout_ref.rollout.trace.experiment_name"
        overrides[key] = run_tag
        put(config, key, run_tag)
    diff = changes(baseline, config)
    require(set(diff) <= set(overrides), "Unexpected training configuration mutation")
    validate_training(config)
    return config, diff


def locations(env):
    project = Path(required(env, "PROJECT_ROOT"))
    output = Path(required(env, "DLC_TRAIN_OUTPUT_DIR"))
    mount = Path(required(env, "DLC_SHARED_MOUNT"))
    tag = required(env, "RUN_TAG")
    require(re.fullmatch(r"shx-[a-z0-9-]+", tag), "RUN_TAG must start shx- and use lower-case letters/digits/hyphens")
    require(all(p.is_absolute() for p in (project, output, mount)), "Use absolute shared paths")
    require(project.resolve() == project and (project / "verl").is_dir(), "PROJECT_ROOT must be canonical production package root")
    require(output.name == tag and output.is_relative_to(project / "results"), "Output must be a fresh results/RUN_TAG directory")
    require(project.is_relative_to(mount) and Path(__file__).resolve().is_relative_to(mount),
            "Project and isolated harness must both be on DLC_SHARED_MOUNT")
    return project, output, mount, tag


def environment_inventory(project, env):
    python = training_python(env)
    prefix = python.parent.parent.resolve()
    require(prefix != (project / ".venv").resolve(), "Production .venv is frozen; use the shared experimental venv")
    root = Path(required(env, "DLC_FA3_ENV_ROOT"))
    require(root.is_absolute() and root.resolve() == prefix.parent and prefix.name == ".venv",
            "DLC_FA3_ENV_ROOT must identify the private project containing this .venv")
    require(root.resolve().is_relative_to(project.parent)
            and root.resolve().is_relative_to(Path(required(env, "DLC_SHARED_MOUNT"))),
            "Experimental project must stay in the personal project parent and shared mount")
    require(Path(sys.prefix).resolve() == prefix, "Wrong interpreter: use DLC_TRAIN_PYTHON")
    require(Path(sys.base_prefix).resolve() == Path(required(env, "DLC_PYTHON_BASE")).resolve(), "Unexpected Python base")
    require(Path(sys.base_prefix).resolve().is_relative_to(Path(required(env, "DLC_SHARED_MOUNT"))), "Python base is not shared")
    require(sys.prefix != sys.base_prefix, "Not running inside the shared venv")
    for value in experiment(env)["caches"].values():
        path = Path(value)
        require(path.is_dir() and path.resolve().is_relative_to(Path(required(env, "DLC_SHARED_MOUNT"))),
                "FA3 cache must already exist on the shared mount; no download fallback")
    return {"python": sys.version, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES}}


def source_inventory(project, env):
    paths = [project / "scripts/run_vzero_qwen35_vl_fsdp.sh",
             project / "verl/trainer/main_ppo.py", project / "verl/trainer/ppo/ray_trainer.py",
             project / "verl/single_controller/ray/base.py", Path(__file__).resolve(),
             Path(__file__).resolve().with_name("pai_dlc_train_overlay.py"),
             Path(__file__).resolve().with_name("teacher_scheduler.py"),
             project / "verl/workers/engine_workers.py", project / "verl/workers/engine/fsdp/transformer_impl.py",
             project / "verl/experimental/agent_loop/agent_loop.py",
             project / "verl/experimental/reward_loop/reward_manager/naive.py"]
    boundary = Path(required(env, "DLC_BOUNDARY_MODULE"))
    require(boundary.is_absolute() and boundary.resolve().is_relative_to(Path(required(env, "DLC_SHARED_MOUNT"))),
            "Boundary module must be on the shared mount")
    paths.append(boundary)
    paths += sorted((project / "verl/trainer/config").rglob("*.yaml"))
    result = {}
    for path in paths:
        require(path.stat().st_size < 2_000_000, "Source fingerprint unexpectedly large")
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def image_paths(value):
    if isinstance(value, list):
        require(bool(value), "Image list cannot be empty")
        return [path for item in value for path in image_paths(item)]
    if isinstance(value, dict):
        for key in ("image", "image_path", "path", "url", "image_url"):
            if isinstance(value.get(key), str):
                return image_paths(value[key])
        raise ValueError("Expected explicit local image paths, not embedded/unknown payloads")
    require(isinstance(value, str) and Path(value).is_absolute(), "Images must have absolute local paths")
    return [value]


def training_columns(config, schema_names):
    data = config["data"]
    distillation = config["distillation"]
    columns = [data["prompt_key"], data["image_key"], distillation["teacher_crop"]["prompt_key"],
               distillation["teacher_crop"]["image_key"], distillation["evidence_weighting"]["negative_image_key"]]
    # Pure OPD currently still invokes the configured reward manager during rollout.
    manager = config["reward"]["reward_manager"]
    legacy_manager = config.get("reward_model", {}).get("reward_manager")
    if (legacy_manager or manager["name"]) == "naive" and manager["source"] == "register":
        columns += ["data_source", "reward_model"]
    missing = sorted(set(columns) - set(schema_names))
    require(not missing, f"Training schema missing required fields: {missing}; never fabricate ground_truth")
    return columns


def inspect_inputs(config, mount):
    import pyarrow.parquet as pq
    data = config["data"]
    require(len(data["train_files"]) == len(data["val_files"]) == 1, "Expected one train and one validation parquet")
    keys = [data["image_key"], config["distillation"]["teacher_crop"]["image_key"],
            config["distillation"]["evidence_weighting"]["negative_image_key"]]
    train = Path(data["train_files"][0])
    val = Path(data["val_files"][0])
    for path in (train, val):
        require(path.is_file() and path.resolve().is_relative_to(mount), "Dataset is not on the shared mount")
    columns = training_columns(config, pq.read_schema(train).names)
    table = pq.read_table(train, columns=columns)
    require(table.num_rows >= 32 and pq.read_metadata(val).num_rows > 0, "Empty/too-small dataset")
    rows = table.to_pylist()
    null_ground_truth_count = 0
    for row in rows:
        for key in (data["prompt_key"], config["distillation"]["teacher_crop"]["prompt_key"]):
            require(isinstance(row[key], list) and row[key], f"Missing message list: {key}")
        if "reward_model" in columns:
            require(isinstance(row["data_source"], str) and row["data_source"], "Empty reward data_source")
            require(isinstance(row["reward_model"], dict) and "ground_truth" in row["reward_model"],
                    "Default reward manager requires the ground_truth field; preserve genuine null labels")
            null_ground_truth_count += row["reward_model"]["ground_truth"] is None
    paths = sorted({path for row in rows for key in keys for path in image_paths(row[key])})
    missing = [path for path in paths if not Path(path).is_file() or not Path(path).resolve().is_relative_to(mount)]
    require(not missing, f"Training image preflight failed: {len(missing)} missing/unshared out of {len(paths)}; no fallback")
    models = [config["actor_rollout_ref"]["model"]["path"],
              next(iter(config["distillation"]["teacher_models"].values()))["model_path"]]
    shards = {}
    for model in models:
        root = Path(model)
        require(root.is_dir() and root.resolve().is_relative_to(mount), "Model is not on the shared mount")
        model_config = json.loads((root / "config.json").read_text())
        require(model_config.get("architectures") == ["Qwen3_5ForConditionalGeneration"], "Unexpected base model architecture")
        index = json.loads((root / "model.safetensors.index.json").read_text())
        names = sorted(set(index["weight_map"].values()))
        require(names and all((root / name).is_file() and (root / name).stat().st_size > 0 for name in names),
                "Missing/empty model shard; no download or alternate model fallback")
        shards[model] = {name: (root / name).stat().st_size for name in names}
    return {"train_rows": table.num_rows, "null_ground_truth_count": null_ground_truth_count,
            "unique_images": len(paths), "image_paths": paths,
            "parquets": {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in (train, val)},
            "model_shards": shards}


def capture(args):
    approved_fla_flags(os.environ)
    controls = experiment(os.environ)
    require(args[:3] == ["-u", "-m", "verl.trainer.main_ppo"], "Unexpected production wrapper entry")
    write_json(os.environ["DLC_CAPTURE_FILE"], {"argv": args[3:],
               "runtime_env": {**{key: os.environ[key] for key in RUNTIME_KEYS if key in os.environ},
                               **OFFLINE, "SHX_PACKED_AUDIT_DIR": controls["audit_dir"]}})


def prepare(env):
    approved_fla_flags(env)
    controls = experiment(env)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    project, output, mount, tag = locations(env)
    for key in ("STUDENT_MODEL", "TEACHER_MODEL", "TRAIN_FILE", "VAL_FILE"):
        required(env, key)
    require(env.get("DLC_INPUTS_VERIFIED") == "1", "Parent must finish data/schema/path audit before prepare")
    inventory = environment_inventory(project, env)
    output.mkdir(parents=True, exist_ok=False)
    child_env = dict(env, **OFFLINE, VERL_REPO_ROOT=str(project), DLC_CAPTURE_FILE=str(output / "wrapper.json"),
                     DLC_CAPTURE_PYTHON=sys.executable, DLC_CAPTURE_ENTRY=str(Path(__file__).resolve()))
    command = ('python3() { "$DLC_CAPTURE_PYTHON" -I -B "$DLC_CAPTURE_ENTRY" _capture "$@"; }; '
               'export -f python3; exec bash "$1"')
    subprocess.run(["bash", "-c", command, "capture", str(project / "scripts/run_vzero_qwen35_vl_fsdp.sh")],
                   env=child_env, check=True, timeout=60)
    captured = json.loads((output / "wrapper.json").read_text())
    with initialize_config_dir(version_base=None, config_dir=str(project / "verl/trainer/config")):
        baseline = OmegaConf.to_container(compose(config_name="ppo_trainer", overrides=captured["argv"]), resolve=True)
    config, diff = training_config(baseline, output, tag, controls)
    write_json(output / "baseline_config.json", baseline)
    write_json(output / "resolved_config.json", config)
    write_json(output / "config_diff.json", diff)
    inputs = inspect_inputs(config, mount)
    index = json.loads((Path(config["actor_rollout_ref"]["model"]["path"]) / "model.safetensors.index.json").read_text())
    model_bytes = index["metadata"]["total_size"]
    require(isinstance(model_bytes, int) and model_bytes > 0, "Model index must provide total_size for the disk estimate")
    space = checkpoint_space(output, model_bytes * 2, controls["minimum_free_gib"])
    write_json(output / "checkpoint_space.json", space)
    write_json(output / "prepared.json", {"run_tag": tag, "environment": inventory,
               "sources": source_inventory(project, env), "inputs": inputs, "runtime_env": captured["runtime_env"],
               "experiment": controls, "checkpoint_estimate_bytes": model_bytes * 2,
               "acceptance": "requires_actual_fa3_forward_backward_all_actor_ranks",
               "optimizer_horizon": 100})
    print(f"PREPARED arm={controls['arm']} steps=100 save=10/model-only B=32 M=16 n=8 actor=[4,4] teacher=[4,4] train_rows={inputs['train_rows']}", flush=True)
    print(f"INPUT_METADATA null_ground_truth_count={inputs['null_ground_truth_count']} rows_filtered=0", flush=True)
    print("Optimizer horizon=100; constant LR=1e-6, zero warmup; all ten model-only checkpoints retained.", flush=True)


def role(env):
    require(env.get("DLC_TRAIN_AUTHORIZED") == "1", "Parent must explicitly authorize this DLC execution")
    require(env.get("WORLD_SIZE") == "2" and env.get("NPROC_PER_NODE") == "8", "Expected DLC two-node/eight-GPU role environment")
    require(env.get("RANK") in {"0", "1"}, "DLC RANK must be node rank 0 or 1")
    required(env, "MASTER_ADDR")
    require(1 <= int(required(env, "MASTER_PORT")) <= 65535, "Invalid DLC MASTER_PORT")
    return int(env["RANK"])


def wait_for(predicate, execution, deadline):
    while True:
        require(not list(execution.glob("failed-*.json")), "Peer failed; no retry")
        if predicate():
            return
        require(time.monotonic() < deadline, "DLC training deadline exceeded; no retry")
        time.sleep(1)


def ray_command(env, rank, ip, project, tag):
    command = [str(training_python(env)), "-B", "-m", "ray.scripts.scripts", "start",
               f"--node-ip-address={ip}", "--num-cpus=184", "--num-gpus=8", "--disable-usage-stats"]
    if rank == 0:
        # Ray appends a session directory and socket name under the Unix path limit.
        scratch_tag = hashlib.sha256(tag.encode()).hexdigest()[:12]
        command += ["--head", f"--port={env['MASTER_PORT']}", "--include-dashboard=false",
                    f"--temp-dir={Path(tempfile.gettempdir()) / ('vzero-ray-' + scratch_tag)}"]
    else:
        command += [f"--address={env['MASTER_ADDR']}:{env['MASTER_PORT']}"]
    return command


def node(env):
    approved_fla_flags(env)
    rank = role(env)
    project, output, mount, tag = locations(env)
    execution = output / "execution"
    deadline = time.monotonic() + 12 * 3600 - 60
    nonce = required(env, "DLC_NODE_NONCE")
    try:
        with (output / f"node-claim-{rank}.json").open("x") as handle:
            json.dump({"nonce": nonce, "rank": rank}, handle)
        if rank == 0:
            execution.mkdir(exist_ok=False)
        else:
            wait_for(execution.is_dir, output, time.monotonic() + 300)
        require(os.path.ismount(mount), "DLC_SHARED_MOUNT is not an actual container mount")
        prepared = json.loads((output / "prepared.json").read_text())
        runtime = child_runtime(env, prepared)
        require(prepared["run_tag"] == tag, "Run identity mismatch")
        require(environment_inventory(project, env) == prepared["environment"], "Shared Python/package versions changed")
        require(source_inventory(project, env) == prepared["sources"], "Production or harness source changed after prepare")
        config = json.loads((output / "resolved_config.json").read_text())
        baseline = json.loads((output / "baseline_config.json").read_text())
        require(config == training_config(baseline, output, tag, prepared["experiment"])[0], "Unreviewed resolved configuration mutation")
        if rank == 0:
            write_json(execution / "wandb-preflight-0.json", wandb_preflight(runtime))
        require(inspect_inputs(config, mount) == prepared["inputs"], "Inputs changed after preparation")
        require(prepared["environment"]["packages"]["ray"] == "2.55.1", "Overlay audited only for Ray 2.55.1")
        runtime = select_cuda_runtime(runtime, execution, rank)
        write_json(execution / f"ready-{rank}.json", {"rank": rank, "hostname": socket.gethostname(),
                   "input_checks": "passed", "fla_env": approved_fla_flags(runtime), "experiment": experiment(runtime)})
        wait_for(lambda: all((execution / f"ready-{r}.json").exists() for r in (0, 1)), execution, min(deadline, time.monotonic() + 300))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((env["MASTER_ADDR"], int(env["MASTER_PORT"])))
            ip = route.getsockname()[0]
        require(not ip.startswith("127."), "Ray requires a routable node address")
        runtime["PYTHONPATH"] = os.pathsep.join([str(project), str(Path(__file__).resolve().parent)])
        runtime["PYTHONDONTWRITEBYTECODE"] = "1"
        if rank == 1:
            wait_for(lambda: (execution / "head-started.json").exists(), execution, min(deadline, time.monotonic() + 300))
        subprocess.run(ray_command(env, rank, ip, project, tag), env=runtime, check=True, timeout=min(180, max(1, deadline - time.monotonic())))
        if rank == 0:
            write_json(execution / "head-started.json", {"node_ip": ip})
            result = subprocess.run([str(training_python(env)), "-B", str(Path(__file__).resolve()), "driver"],
                                    env=runtime, timeout=max(1, deadline - time.monotonic()))
            require(result.returncode == 0, f"Training driver failed with exit code {result.returncode}; no retry")
            require((output / "training_complete.json").is_file(), "Missing 100-step training and checkpoint evidence")
            write_json(execution / "complete.json", {"operational": True, "steps": 100, "precision_pass": False})
        else:
            wait_for(lambda: (execution / "complete.json").exists(), execution, deadline)
        write_json(execution / f"node-result-{rank}.json", {"exit_code": 0, "nonce": nonce})
    except BaseException as error:
        failure_dir = execution if execution.is_dir() else output
        write_json(failure_dir / f"failed-{rank}-{nonce}.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise


def driver(env):
    approved_fla_flags(env)
    role(env)
    project, output, _, tag = locations(env)
    prepared = json.loads((output / "prepared.json").read_text())
    child_runtime(env, prepared)

    import ray
    from omegaconf import OmegaConf
    from pai_dlc_train_overlay import task_runner_class
    from verl.trainer.main_ppo import auto_set_device, migrate_legacy_reward_impl, run_ppo
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    runtime_env = ray_job_runtime_env(env, get_ppo_ray_runtime_env())
    runtime_env.setdefault("env_vars", {})["PYTHONPATH"] = os.pathsep.join([str(project), str(Path(__file__).resolve().parent)])
    write_json(output / "ray_runtime_env.json", {"fla_env": approved_fla_flags(runtime_env["env_vars"]),
               "experiment": experiment(runtime_env["env_vars"]), "offline": OFFLINE,
               "ld_policy": "inherit_each_node_validated_raylet_environment"})
    ray.init(address=f"{env['MASTER_ADDR']}:{env['MASTER_PORT']}", namespace=tag, runtime_env=runtime_env)
    try:
        wait_for(lambda: len([n for n in ray.nodes() if n.get("Alive")]) >= 2, output / "execution", time.monotonic() + 180)
        select_nodes(ray.nodes())
        config = OmegaConf.create(json.loads((output / "resolved_config.json").read_text()))
        auto_set_device(config)
        config = migrate_legacy_reward_impl(config)
        baseline = json.loads((output / "baseline_config.json").read_text())
        run_ppo(config, task_runner_class=task_runner_class(str(output), prepared, baseline))
    finally:
        ray.shutdown()


TERMINFO_CANDIDATES = ("/usr/share/terminfo", "/etc/terminfo", "/lib/terminfo")


def tmux_argv(env, *arguments):
    terminal = env.get("TERM") or "xterm"
    if terminal == "dumb":
        terminal = "xterm"
    directories = [Path(path) for path in TERMINFO_CANDIDATES if Path(path).is_dir()]
    entries = (terminal[0], format(ord(terminal[0]), "x"))
    require(any((directory / entry / terminal).is_file() for directory in directories for entry in entries),
            f"No system terminfo entry for {terminal}; no installation or environment fallback")
    # Scope the database override to tmux; preserve the shared Python's library path.
    return ["env", f"TERM={terminal}", "TERMINFO_DIRS=" + os.pathsep.join(map(str, directories)),
            "tmux", *arguments]


def probe_tmux(env):
    import pty
    project, _, _, tag = locations(env)
    environment_inventory(project, env)
    library_path = env.get("LD_LIBRARY_PATH", "")
    require(str(Path(sys.base_prefix) / "lib") in library_path.split(os.pathsep),
            "CPU tmux probe must use the same shared-base LD_LIBRARY_PATH")
    nonce = uuid.uuid4().hex
    socket_name = f"{tag}-cpu-{nonce[:8]}"
    child = (
        "import hashlib,os,sys,time;"
        "assert os.environ.get('TMUX','').split(',')[0].endswith('/'+sys.argv[1]);"
        "assert hashlib.sha256(os.environ.get('LD_LIBRARY_PATH','').encode()).hexdigest()==sys.argv[2];"
        "assert os.environ.get('TERMINFO_DIRS');"
        "print('TMUX_CPU_STDOUT_'+sys.argv[3],flush=True);"
        "print('TMUX_CPU_STDERR_'+sys.argv[3],file=sys.stderr,flush=True);"
        "time.sleep(0.3)"
    )
    command = shlex.join([str(training_python(env)), "-I", "-B", "-c", child, socket_name,
                          hashlib.sha256(library_path.encode()).hexdigest(), nonce])
    observed = bytearray()

    def direct_output(fd):
        data = os.read(fd, 4096)
        observed.extend(data)
        return data

    # A fresh private server prevents a previously healthy DSW tmux from masking this failure.
    argv = ["env", "-u", "TMUX", "-u", "TMUX_PANE",
            *tmux_argv(env, "-L", socket_name, "new-session", "-s", socket_name, command)]
    status = pty.spawn(argv, master_read=direct_output)
    passed = status == 0 and all(("TMUX_CPU_" + stream + "_" + nonce).encode() in observed
                                  for stream in ("STDOUT", "STDERR"))
    evidence = Path(__file__).resolve().parent / "cpu_probes" / f"{tag}-{nonce}.json"
    evidence.parent.mkdir(exist_ok=True)
    write_json(evidence, {"passed": passed, "private_socket": socket_name, "pty_wait_status": status,
                         "same_shared_ld_library_path": True, "stdout_and_stderr_seen": passed,
                         "terminfo_dirs": next(value.partition("=")[2] for value in argv if value.startswith("TERMINFO_DIRS=")), "training_started": False})
    require(passed, f"CPU PTY/tmux probe failed; evidence: {evidence}")
    print(f"TMUX_CPU_PROBE_PASSED private_socket={socket_name} evidence={evidence}", flush=True)


def launch(env):
    approved_fla_flags(env)
    experiment(env)
    # A PTY keeps the task in its own tmux and streams terminal output directly.
    import pty
    rank = role(env)
    project, output, _, tag = locations(env)
    session = f"{tag}-node{rank}"
    nonce = uuid.uuid4().hex
    command = shlex.join([str(training_python(env)), "-I", "-B", str(Path(__file__).resolve()), "node"])
    private_tmux = shlex.join(tmux_argv(env, "-L", tag))
    command = (f'{private_tmux} set-option -t {shlex.quote(session)} remain-on-exit on; '
               f'DLC_NODE_NONCE={nonce} {command}; {private_tmux} detach-client -s {shlex.quote(session)}')
    pty.spawn(tmux_argv(env, "-L", tag, "new-session", "-s", session, command))
    result = output / "execution" / f"node-result-{rank}.json"
    require(result.is_file() and json.loads(result.read_text()) == {"exit_code": 0, "nonce": nonce},
            "DLC node failed or result is stale; inspect direct terminal output and failure JSON")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_cuda_probe":
        raise SystemExit(cuda_probe(sys.argv[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "_capture":
        capture(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "node", "driver", "probe-tmux"))
    args = parser.parse_args()
    globals()[args.mode.replace("-", "_")](os.environ)


if __name__ == "__main__":
    main()

"""CPU-only topology and run-control regressions; never start Ray or import torch."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pai_dlc_train as launcher
import pai_dlc_train_overlay as overlay


class FourNodeLauncherTest(unittest.TestCase):
    def test_config(self):
        config = {
            "trainer": {},
            "data": {"train_batch_size": 32},
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp", "ppo_mini_batch_size": 16, "checkpoint": {},
                          "fsdp_config": {"ulysses_sequence_parallel_size": 1, "fsdp_size": -1},
                          "optim": {"lr": 1e-6, "lr_scheduler_type": "constant"}},
                "rollout": {"n": 8, "tensor_model_parallel_size": 2},
                "model": {"path": "models/Qwen3.5-9B", "use_remove_padding": True,
                          "override_config": {}}},
            "distillation": {
                "enabled": True, "teacher_ema": {"enabled": False},
                "teacher_models": {"teacher": {"model_path": "models/Qwen3.5-27B",
                                               "inference": {"tensor_model_parallel_size": 2,
                                                             "engine_kwargs": {"vllm": {}}}}},
                "distillation_loss": {"use_task_rewards": False},
                "evidence_weighting": {"enabled": True, "negative_mode": "hard_negative_crop"},
                "teacher_crop": {"enabled": True}}}
        original = copy.deepcopy(config)
        resolved, _ = launcher.training_config(config, Path("unused"), "shx-cpu-test",
                                               {"fa3_backend": "kernels-community/flash-attn3@" + "a" * 40})
        self.assertEqual(config, original)
        for key in ("trainer.nnodes", "distillation.nnodes"):
            self.assertEqual(launcher.get(resolved, key), 4)
        for key in ("trainer.n_gpus_per_node", "distillation.n_gpus_per_node"):
            self.assertEqual(launcher.get(resolved, key), 4)
        self.assertEqual(launcher.get(resolved, "trainer.total_training_steps"), 100)
        self.assertEqual(launcher.get(resolved, "trainer.save_freq"), 10)
        self.assertEqual(launcher.get(resolved, "actor_rollout_ref.actor.checkpoint.save_contents"), ["model"])
        self.assertEqual(launcher.get(resolved, "distillation.teacher_models.teacher.inference.engine_kwargs.vllm.scheduler_cls"),
                         launcher.SCHEDULER_CLASS)
        for key, value in (("data.train_batch_size", 128), ("actor_rollout_ref.actor.ppo_mini_batch_size", 64),
                           ("actor_rollout_ref.actor.optim.lr", 2e-6), ("actor_rollout_ref.actor.fsdp_config.fsdp_size", 8)):
            bad = copy.deepcopy(resolved)
            launcher.put(bad, key, value)
            with self.assertRaises(ValueError, msg=key):
                launcher.validate_training(bad)

    def test_role_and_timeout(self):
        env = {"DLC_TRAIN_AUTHORIZED": "1", "WORLD_SIZE": "4", "NPROC_PER_NODE": "8",
               "MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "6379"}
        for rank in range(4):
            self.assertEqual(launcher.role(dict(env, RANK=str(rank))), rank)
        for changes in ({"RANK": "4"}, {"RANK": "0", "WORLD_SIZE": "2"},
                        {"RANK": "0", "NPROC_PER_NODE": "4"}):
            with self.assertRaises(ValueError):
                launcher.role(dict(env, **changes))
        self.assertEqual(launcher.max_runtime_seconds({}), 129600)
        self.assertEqual(launcher.max_runtime_seconds({"DLC_MAX_RUNTIME_SECONDS": "7200"}), 7200)
        self.assertIn("DLC_MAX_RUNTIME_SECONDS", launcher.RUNTIME_KEYS)
        for invalid in ("0", "60", "-1", "nan", "129600.0", ""):
            with self.assertRaises(ValueError):
                launcher.max_runtime_seconds({"DLC_MAX_RUNTIME_SECONDS": invalid})

    def test_resource_pools(self):
        nodes = [{"Alive": True, "NodeID": str(i), "NodeManagerAddress": f"10.0.0.{i}",
                  "Resources": {"GPU": 8, "CPU": 184, f"node:10.0.0.{i}": 1}} for i in range(4)]
        self.assertEqual(overlay.select_nodes(nodes), nodes)
        for bad in (nodes[:2], nodes + [nodes[0]]):
            with self.assertRaises(ValueError):
                overlay.select_nodes(bad)
        for resource, value in (("GPU", 4), ("CPU", 183)):
            bad = copy.deepcopy(nodes)
            bad[3]["Resources"][resource] = value
            with self.assertRaises(ValueError):
                overlay.select_nodes(bad)
        for name in ("global_pool", "teacher_pool"):
            pool = SimpleNamespace(name_prefix=name, _store=[4] * 4, use_gpu=True,
                                   detached=False, max_colocate_count=1, accelerator_type=None)
            self.assertEqual(sum(len(overlay.pinned_bundles(pool, node)) for node in nodes), 16)
            pool._store = [4, 4]
            with self.assertRaises(ValueError):
                overlay.pinned_bundles(pool, nodes[0])

    def test_sixteen_checkpoint_shards_and_actor_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "actor").mkdir()
            for rank in range(16):
                (root / "actor" / f"model_world_size_16_rank_{rank}.pt").write_bytes(b"cpu fixture")
                overlay.write_json(root / f"actor-runtime-rank-{rank}.json", {
                    "rank": rank, "arm": "baseline", "training_calls": 100, "fa3_forward": 100,
                    "fa3_backward": 100, "packed_position_checks": 1, "boundary_setup": True,
                    "boundary_installed": False, "boundary_calls": 0})
            self.assertEqual(len(overlay.checkpoint_files(root)["model_files"]), 16)
            self.assertEqual(len(overlay.verify_actor_evidence(root, "baseline", 100)), 16)
            (root / "actor" / "model_world_size_16_rank_15.pt").unlink()
            with self.assertRaises(ValueError):
                overlay.checkpoint_files(root)
            (root / "actor-runtime-rank-15.json").unlink()
            with self.assertRaises(ValueError):
                overlay.verify_actor_evidence(root, "baseline", 100)

    def test_all_node_branches_wait_for_four_ready_and_workers_wait_for_head(self):
        for rank in range(4):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                output = Path(temporary)
                execution = output / "execution"
                if rank:
                    execution.mkdir()
                controls = {"max_runtime_seconds": 129600}
                prepared = {"run_tag": "shx-cpu-test", "environment": {"packages": {"ray": "2.55.1"}},
                            "sources": {}, "inputs": {}, "experiment": controls}
                for name, value in (("prepared", prepared), ("resolved_config", {}), ("baseline_config", {}),
                                    ("training_complete", {})):
                    overlay.write_json(output / (name + ".json"), value)
                env = {"DLC_TRAIN_AUTHORIZED": "1", "WORLD_SIZE": "4", "NPROC_PER_NODE": "8",
                       "RANK": str(rank), "MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "6379",
                       "DLC_NODE_NONCE": "cpu", "DLC_TRAIN_PYTHON": str(output / "venv/bin/python"),
                       "FLA_DISABLE_BACKEND_DISPATCH": "0", "FLA_TILELANG": "1"}
                state = {"ready": False, "head": False, "commands": []}

                def wait(predicate, directory, deadline):
                    if not state["ready"] and execution.is_dir() and (execution / f"ready-{rank}.json").exists():
                        for other in range(4):
                            if other != rank:
                                self.assertFalse(predicate(), f"rank {rank} ignored missing peer {other}")
                                overlay.write_json(execution / f"ready-{other}.json", {})
                        state["ready"] = True
                    elif state["ready"] and rank and not state["head"]:
                        self.assertFalse(predicate())
                        overlay.write_json(execution / "head-started.json", {})
                        state["head"] = True
                    elif state["head"]:
                        self.assertEqual(deadline, 1000 + 129600 - 60)
                        overlay.write_json(execution / "complete.json", {})
                    self.assertTrue(predicate())

                def run(command, **kwargs):
                    self.assertTrue(state["ready"])
                    if rank:
                        self.assertTrue(state["head"])
                    state["commands"].append(command)
                    return SimpleNamespace(returncode=0)

                for name, value in {"locations": (output, output, output, "shx-cpu-test"),
                                    "child_runtime": env, "environment_inventory": prepared["environment"],
                                    "source_inventory": {}, "training_config": ({}, {}), "wandb_preflight": {},
                                    "inspect_inputs": {}, "select_cuda_runtime": env, "experiment": controls}.items():
                    stack.enter_context(patch.object(launcher, name, return_value=value))
                stack.enter_context(patch.object(launcher.os.path, "ismount", return_value=True))
                stack.enter_context(patch.object(launcher.time, "monotonic", return_value=1000))
                stack.enter_context(patch.object(launcher, "wait_for", side_effect=wait))
                stack.enter_context(patch.object(launcher.subprocess, "run", side_effect=run))
                socket_mock = stack.enter_context(patch.object(launcher.socket, "socket"))
                socket_mock.return_value.__enter__.return_value.getsockname.return_value = ("10.0.0.2", 1)
                launcher.node(env)
                self.assertEqual(len(state["commands"]), 2 if rank == 0 else 1)
                self.assertEqual(json.loads((execution / f"node-result-{rank}.json").read_text())["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()

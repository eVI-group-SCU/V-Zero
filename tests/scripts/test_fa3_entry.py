"""Exercise shell dispatch without creating Ray processes or loading GPU libraries."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/run_vzero_qwen35_vl_fsdp.sh"


class EntryTest(unittest.TestCase):
    def test_profiles_prepare_and_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / "bin/python"
            python.parent.mkdir()
            python.write_text(
                "#!/bin/sh\nexec /usr/bin/env python3 -c "
                '\'import os,sys,json; print(json.dumps({"argv":sys.argv[1:],"env":dict(os.environ)}))\' "$@"\n'
            )
            python.chmod(0o755)
            (root / "out").mkdir()
            (root / "out/prepared.json").write_text("{}")
            env = dict(
                os.environ,
                RUN_TAG="vzero-test",
                VZERO_FA3_SHARED_MOUNT=tmp,
                VZERO_FA3_ENV_ROOT=tmp,
                VZERO_FA3_PYTHON_BASE=tmp,
                VZERO_FA3_WANDB_ENV_FILE=str(root / "private.env"),
                VZERO_FA3_TRAIN_PYTHON=str(python),
                VZERO_FA3_TRAIN_OUTPUT_DIR=str(root / "out"),
                MODEL_ROOT="models",
                TRAIN_FILE="train.parquet",
                VAL_FILE="val.parquet",
                RANK="0",
                MASTER_ADDR="127.0.0.1",
            )
            for key in ["ATTN_IMPLEMENTATION", "STUDENT_MODEL", "TEACHER_MODEL", "VZERO_CONFIG_CAPTURE"]:
                env.pop(key, None)
            for profile, nodes, arm, model in [
                ("four_b", "2", "baseline", "4B"),
                ("four_b_boundary", "2", "boundary", "4B"),
                ("nine_b", "4", "baseline", "9B"),
            ]:
                for mode in ("prepare", "launch"):
                    run = subprocess.run(
                        ["bash", str(ENTRY), "--" + mode],
                        env=dict(env, FA3_PROFILE=profile, VZERO_TRAIN_BACKEND="fa3"),
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    result = json.loads(run.stdout)
                    actual = result["env"]
                    self.assertEqual(result["argv"], ["-I", "-B", str(ROOT / "scripts/fa3/launch.py"), mode])
                    self.assertEqual(actual["NNODES"], nodes)
                    self.assertEqual(actual["VZERO_PACKED_BOUNDARY_ARM"], arm)
                    self.assertEqual(actual["STUDENT_MODEL"], "models/Qwen3.5-" + model)
                    self.assertEqual(actual["FLA_DISABLE_BACKEND_DISPATCH"], "0")
                    self.assertEqual(actual["FLA_TILELANG"], "1")
                    self.assertEqual(actual["VZERO_FA3_TRAIN_STEPS"], "100")
                    # Check the real runtime parser with the shell-exported controls.
                    # A fresh interpreter is needed because topology is fixed at import.
                    subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            "import sys,json; sys.path.insert(0,sys.argv[1]); "
                            "import launch; env=json.load(sys.stdin); "
                            "launch.approved_fla_flags(env); launch.experiment(env)",
                            str(ROOT / "scripts/fa3"),
                        ],
                        input=json.dumps(actual),
                        env=actual,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    if mode == "launch":
                        self.assertEqual(actual["WORLD_SIZE"], nodes)
            custom = subprocess.run(
                ["bash", str(ENTRY), "--prepare"],
                env=dict(
                    env,
                    FA3_PROFILE="four_b",
                    VZERO_TRAIN_BACKEND="fa3",
                    NNODES="4",
                    CPUS_PER_NODE="64",
                    TOTAL_TRAINING_STEPS="60",
                    SAVE_FREQ="15",
                ),
                text=True,
                capture_output=True,
                check=True,
            )
            custom_env = json.loads(custom.stdout)["env"]
            self.assertEqual(custom_env["VZERO_FA3_TRAIN_STEPS"], "60")
            self.assertEqual(custom_env["VZERO_FA3_SAVE_FREQ"], "15")
            self.assertEqual(custom_env["VZERO_FA3_NNODES"], "4")
            self.assertEqual(custom_env["VZERO_FA3_CPUS_PER_NODE"], "64")
            bad = subprocess.run(
                ["bash", str(ENTRY), "--prepare"],
                env=dict(env, ATTN_IMPLEMENTATION="flash_attention_2"),
                capture_output=True,
            )
            self.assertNotEqual(bad.returncode, 0)

    def test_harness_capture_does_not_reenter_dispatch(self):
        env = dict(
            os.environ,
            VZERO_CONFIG_CAPTURE="1",
            VZERO_TRAIN_BACKEND="fa3",
            STUDENT_MODEL="student",
            TEACHER_MODEL="teacher",
            TRAIN_FILE="train",
            VAL_FILE="val",
            ATTN_IMPLEMENTATION="kernels-community/flash-attn3@" + "a" * 40,
        )
        command = 'python3() { printf "%s\\n" "$@"; }; export -f python3; bash "$1"'
        run = subprocess.run(
            ["bash", "-c", command, "test", str(ENTRY)], env=env, text=True, capture_output=True, check=True
        )
        self.assertTrue(run.stdout.startswith("-u\n-m\nverl.trainer.main_ppo\n"))
        self.assertIn(
            "+actor_rollout_ref.model.override_config.attn_implementation=" + env["ATTN_IMPLEMENTATION"], run.stdout
        )


if __name__ == "__main__":
    unittest.main()

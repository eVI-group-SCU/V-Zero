"""Run VA unit tests on CPU without importing Ray worker initializers.

Requires torch, Pillow, numpy, pytest, tensordict, transformers, omegaconf,
hydra-core and codetiming. The tested loss/config modules are loaded unchanged;
only the package initializers that register distributed workers are bypassed.
Linux training environments can instead use `python -m pytest tests/distillation`.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name, relative in [("verl", "verl"), ("verl.utils.profiler", "verl/utils/profiler")]:
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / relative)]
    sys.modules[name] = package

from verl.utils.profiler.config import ProfilerConfig  # noqa: E402

sys.modules["verl.utils.profiler"].ProfilerConfig = ProfilerConfig

import pytest  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", str(ROOT / "tests/distillation"), *sys.argv[1:]]))

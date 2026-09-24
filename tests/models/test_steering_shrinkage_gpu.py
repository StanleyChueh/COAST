"""Manual GPU tests: the Phase 2A ``shrinkage`` ablation on the real pi0.5 LIBERO checkpoint.

Stack as in serve_policy.py: NoiseControlledPolicyWrapper(SteeredPolicyWrapper(Policy)) with the
repository-faithful Phase 1B NPZ. One fixed observation and one fixed explicit noise key:

  - shrinkage at beta = 0 reproduces the unsteered actions exactly
  - shrinkage at beta = 0.1 is a pure contraction at layer 5 (cosine -1, norm ratio = bf16(0.9))
  - prints max |action difference| for baseline / COAST (global) / shrinkage (descriptive)

Skipped by default; run explicitly with::

    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/models/test_steering_shrinkage_gpu.py -m manual -v -s
"""

import math
import pathlib

import numpy as np
from openpi_client.steering import STEERING_DIAGNOSTICS_KEY
import pytest

from openpi.policies import policy_config as _policy_config
from openpi.serving.noise_control import NoiseControlledPolicyWrapper
from openpi.serving.noise_control import noise_shape_from_policy
from openpi.serving.steering import SteeredPolicyWrapper
from openpi.training import config as _config

_CHECKPOINT_DIR = "checkpoints/openpi-libero-2000"
_NPZ = "conceptors/phase1b_repo_task02.npz"
_TASK = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
_KEY = {"master_seed": 100, "task_id": 2, "init_state": 15, "rollout_step": 0}
_BF16_POINT_NINE = 0.8984375

pytestmark = pytest.mark.manual


def _steer(strategy: str, beta: float) -> dict:
    return {"task": _TASK, "layer": 5, "alpha": 0.5, "beta": beta, "strategy": strategy}


@pytest.fixture(scope="module")
def stack():
    missing = [p for p in (_CHECKPOINT_DIR, _NPZ) if not pathlib.Path(p).exists()]
    if missing:
        pytest.skip(f"missing local assets: {missing}")
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), _CHECKPOINT_DIR)
    horizon, dim = noise_shape_from_policy(policy)
    steered = SteeredPolicyWrapper(
        policy,
        conceptor_npz_path=_NPZ,
        device=str(policy._pytorch_device),  # noqa: SLF001
        record_diagnostics=True,
    )
    return NoiseControlledPolicyWrapper(steered, action_horizon=horizon, action_dim=dim)


def _infer(stack, steer: dict | None) -> dict:
    rng = np.random.default_rng(0)
    obs = {
        "observation/image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": "turn on the stove and put the moka pot on it",
        "__noise_control__": dict(_KEY),
    }
    if steer is not None:
        obs["__steering__"] = steer
    return stack.infer(obs)


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def test_shrinkage_beta_zero_equals_baseline(stack):
    base = _infer(stack, None)["actions"]
    shrink0 = _infer(stack, _steer("shrinkage", 0.0))["actions"]
    assert np.array_equal(base, shrink0)


def test_shrinkage_is_pure_contraction_on_real_model(stack):
    records = _infer(stack, _steer("shrinkage", 0.1))[STEERING_DIAGNOSTICS_KEY]
    print("\nshrinkage norm ratio per step:", [round(r["mean_norm_ratio"], 6) for r in records])
    print("shrinkage relative delta per step:", [round(r["mean_relative_delta"], 6) for r in records])
    print("shrinkage cosine per step:", [round(r["mean_cosine_delta_hidden"], 6) for r in records])
    assert [r["denoising_step"] for r in records] == list(range(10))
    # h' = bf16(0.8984375 * h) is rounded again to bf16 (relative error <= 2**-9 per element), so the
    # per-token ratio scatters around 0.8984375 and the delta is not exactly antiparallel to h.
    for r in records:
        assert r["layer"] == 5
        assert all(math.isfinite(v) for v in r.values() if isinstance(v, float))
        assert r["mean_cosine_delta_hidden"] == pytest.approx(-1.0, abs=1e-3)
        assert r["mean_norm_ratio"] == pytest.approx(_BF16_POINT_NINE, abs=1e-3)


def test_action_differences_baseline_coast_shrinkage(stack):
    """Descriptive: how far COAST and shrinkage move the actions, and how far apart they are."""
    base = _infer(stack, None)["actions"]
    coast_result = _infer(stack, _steer("global", 0.1))
    shrink_result = _infer(stack, _steer("shrinkage", 0.1))
    coast, shrink = coast_result["actions"], shrink_result["actions"]
    d_cb, d_sb, d_cs = _max_abs(coast, base), _max_abs(shrink, base), _max_abs(coast, shrink)
    print(f"\nmax|COAST-base|={d_cb:.4f}  max|shrink-base|={d_sb:.4f}  max|COAST-shrink|={d_cs:.4f}")
    coast_ratio = [round(r["mean_norm_ratio"], 5) for r in coast_result[STEERING_DIAGNOSTICS_KEY]]
    print("COAST norm ratio per step:", coast_ratio)
    assert d_cb > 0
    assert d_sb > 0
    assert d_cs > 0

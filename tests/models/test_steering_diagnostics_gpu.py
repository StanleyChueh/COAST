"""Manual GPU tests: steering intervention diagnostics on the real pi0.5 LIBERO checkpoint.

Builds the serve_policy.py stack NoiseControlledPolicyWrapper(SteeredPolicyWrapper(Policy))
with the repository-faithful Phase 1B NPZ and checks, on one fixed observation and one
fixed explicit noise key, through the real ``PI0Pytorch.sample_actions_with_steering``:

  - one record per denoising step (0..9), labeled layer 5, 10 action tokens, hidden dim 1024
  - all values finite; beta = 0.1 gives a nonzero delta, beta = 0 gives exactly zero
  - actions are bit-identical with diagnostics on and off (same noise)
  - an unsteered request on a diagnostics server carries no diagnostics

Skipped by default; run explicitly with::

    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/models/test_steering_diagnostics_gpu.py -m manual -v -s
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
_MAGNITUDE_FIELDS = (
    "mean_delta_norm",
    "max_delta_norm",
    "mean_relative_delta",
    "max_relative_delta",
    "mean_cosine_delta_hidden",
)

pytestmark = pytest.mark.manual


def _steer(beta: float) -> dict:
    return {"task": _TASK, "layer": 5, "alpha": 0.5, "beta": beta, "strategy": "global"}


@pytest.fixture(scope="module")
def policy():
    missing = [p for p in (_CHECKPOINT_DIR, _NPZ) if not pathlib.Path(p).exists()]
    if missing:
        pytest.skip(f"missing local assets: {missing}")
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), _CHECKPOINT_DIR)
    assert policy._is_pytorch_model  # noqa: SLF001
    return policy


def _stack(policy, *, record: bool):
    horizon, dim = noise_shape_from_policy(policy)
    steered = SteeredPolicyWrapper(
        policy,
        conceptor_npz_path=_NPZ,
        device=str(policy._pytorch_device),  # noqa: SLF001
        record_diagnostics=record,
    )
    return NoiseControlledPolicyWrapper(steered, action_horizon=horizon, action_dim=dim)


def _obs(steer: dict | None) -> dict:
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
    return obs


def test_real_sampler_records_every_denoising_step(policy):
    result = _stack(policy, record=True).infer(_obs(_steer(0.1)))
    records = result[STEERING_DIAGNOSTICS_KEY]
    print("\nper-step mean_delta_norm:", [round(r["mean_delta_norm"], 4) for r in records])
    print("per-step mean_relative_delta:", [round(r["mean_relative_delta"], 5) for r in records])
    print("per-step mean_cosine_delta_hidden:", [round(r["mean_cosine_delta_hidden"], 5) for r in records])
    assert [r["denoising_step"] for r in records] == list(range(10))
    assert {r["layer"] for r in records} == {5}
    assert {r["token_count"] for r in records} == {10}
    assert {r["hidden_dim"] for r in records} == {1024}
    for record in records:
        assert all(math.isfinite(v) for v in record.values() if isinstance(v, float))
        assert record["mean_delta_norm"] > 0
        assert record["max_delta_norm"] >= record["mean_delta_norm"]


def test_real_sampler_beta_zero_is_exactly_zero(policy):
    records = _stack(policy, record=True).infer(_obs(_steer(0.0)))[STEERING_DIAGNOSTICS_KEY]
    assert len(records) == 10
    for record in records:
        assert all(record[field] == 0.0 for field in _MAGNITUDE_FIELDS)


def test_diagnostics_do_not_change_actions(policy):
    """Same obs, same explicit noise: bit-identical actions with diagnostics on and off."""
    off = np.asarray(_stack(policy, record=False).infer(_obs(_steer(0.1)))["actions"])
    on = np.asarray(_stack(policy, record=True).infer(_obs(_steer(0.1)))["actions"])
    print(f"\nmax |on - off| = {float(np.max(np.abs(on.astype(np.float64) - off))):.3e}")
    assert np.array_equal(off, on)


def test_unsteered_request_has_no_diagnostics(policy):
    result = _stack(policy, record=True).infer(_obs(None))
    assert STEERING_DIAGNOSTICS_KEY not in result

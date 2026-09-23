"""Manual GPU tests: controlled (paired) flow noise on the real pi0.5 LIBERO checkpoint.

Builds the serve_policy.py stack NoiseControlledPolicyWrapper(SteeredPolicyWrapper(Policy))
for both Phase 1B conceptor NPZs and checks, on one fixed observation:

  A. same explicit noise + same observation -> identical baseline actions
  B. the explicit noise reaches sample_actions_with_steering unchanged (on device)
  C. beta = 0 steering with the same noise == unsteered actions
  D. a different noise key changes the actions

Tolerance: exact equality (np.array_equal / torch.equal) for A, B and C. Skipped by
default; run explicitly with::

    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/models/test_noise_control_gpu.py -m manual -v -s
"""

# ruff: noqa: N802
import pathlib

import numpy as np
import pytest
import torch

from openpi.policies import policy_config as _policy_config
from openpi.serving.noise_control import NoiseControlledPolicyWrapper
from openpi.serving.noise_control import derive_noise
from openpi.serving.noise_control import noise_shape_from_policy
from openpi.serving.steering import SteeredPolicyWrapper
from openpi.training import config as _config

_CHECKPOINT_DIR = "checkpoints/openpi-libero-2000"
_NPZS = {"repo_V0": "conceptors/phase1b_repo_task02.npz", "paper_V3": "conceptors/phase1b_paper_task02.npz"}
_TASK = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
_KEY = {"master_seed": 100, "task_id": 2, "init_state": 15, "rollout_step": 0}

pytestmark = pytest.mark.manual


def _steer(beta: float) -> dict:
    return {"task": _TASK, "layer": 5, "alpha": 0.5, "beta": beta, "strategy": "global"}


@pytest.fixture(scope="module")
def policy():
    missing = [p for p in (_CHECKPOINT_DIR, *_NPZS.values()) if not pathlib.Path(p).exists()]
    if missing:
        pytest.skip(f"missing local assets: {missing}")
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), _CHECKPOINT_DIR)
    assert policy._is_pytorch_model  # noqa: SLF001
    return policy


@pytest.fixture(scope="module")
def stacks(policy):
    horizon, dim = noise_shape_from_policy(policy)
    assert (horizon, dim) == (10, 32)
    device = str(policy._pytorch_device)  # noqa: SLF001
    return {
        name: NoiseControlledPolicyWrapper(
            SteeredPolicyWrapper(policy, conceptor_npz_path=path, device=device), action_horizon=horizon, action_dim=dim
        )
        for name, path in _NPZS.items()
    }


def _obs() -> dict:
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": "turn on the stove and put the moka pot on it",
    }


def _actions(stack, key: dict, steer: dict | None = None) -> np.ndarray:
    obs = _obs()
    obs["__noise_control__"] = dict(key)
    if steer is not None:
        obs["__steering__"] = steer
    return np.asarray(stack.infer(obs)["actions"])


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def test_A_baseline_reproducible_with_same_noise(stacks, policy):
    stack = stacks["repo_V0"]
    first = _actions(stack, _KEY)
    torch.manual_seed(987)  # perturb the global RNG; explicit noise must make it irrelevant
    torch.cuda.manual_seed_all(987)
    repeats = [_actions(stack, _KEY) for _ in range(3)]
    direct = np.asarray(policy.infer(_obs(), noise=derive_noise(dict(_KEY), 10, 32))["actions"])
    diffs = [_max_abs(first, r) for r in [*repeats, direct]]
    print(f"\n[A] actions shape={first.shape}; max|diff| repeat x3 + direct Policy.infer = {diffs}")
    for other in [*repeats, direct]:
        assert np.array_equal(first, other)


def test_B_explicit_noise_reaches_steering_sampler(stacks, policy, monkeypatch):
    model = policy._model  # noqa: SLF001
    received = []
    original = model.sample_actions_with_steering

    def spy(*args, **kwargs):
        received.append(kwargs.get("noise"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "sample_actions_with_steering", spy)
    _actions(stacks["paper_V3"], _KEY, _steer(0.1))
    (noise,) = received
    expected = torch.from_numpy(derive_noise(dict(_KEY), 10, 32))[None, ...]
    print(f"\n[B] sampler noise: shape={tuple(noise.shape)} dtype={noise.dtype} device={noise.device}")
    assert noise.shape == (1, 10, 32)
    assert noise.dtype == torch.float32
    assert noise.device.type == "cuda"
    assert torch.equal(noise.cpu(), expected)


@pytest.mark.parametrize("name", list(_NPZS))
def test_C_beta_zero_steering_matches_unsteered(stacks, name):
    stack = stacks[name]
    baseline = _actions(stack, _KEY)
    beta0 = _actions(stack, _KEY, _steer(0.0))
    print(f"\n[C] {name}: max|steered(beta=0) - baseline| = {_max_abs(baseline, beta0)}")
    assert np.array_equal(baseline, beta0)


def test_C_sampler_code_paths_match_without_hooks(policy):
    """sample_actions vs sample_actions_with_steering(steering_hooks=None) on identical noise."""
    import jax

    from openpi.models import model as _model

    inputs = policy._input_transform(jax.tree.map(lambda x: x, _obs()))  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    noise = torch.from_numpy(derive_noise(dict(_KEY), 10, 32))[None, ...].to(device)
    a = policy._model.sample_actions(device, _model.Observation.from_dict(inputs), noise=noise)  # noqa: SLF001
    b, _ = policy._model.sample_actions_with_steering(  # noqa: SLF001
        device, _model.Observation.from_dict(inputs), noise=noise, steering_hooks=None
    )
    print(f"\n[C] raw sampler max|sample_actions - sample_actions_with_steering| = {(a - b).abs().max().item()}")
    assert torch.equal(a, b)


def test_D_different_noise_key_changes_actions(stacks):
    stack = stacks["repo_V0"]
    base = _actions(stack, _KEY)
    diffs = {
        field: _max_abs(base, _actions(stack, {**_KEY, field: _KEY[field] + delta}))
        for field, delta in (("master_seed", 100), ("rollout_step", 5))
    }
    print(f"\n[D] max|diff| when changing the noise key: {diffs}")
    for diff in diffs.values():
        assert diff > 1e-3


def test_steering_beta_nonzero_changes_actions_with_same_noise(stacks):
    """Sanity: with paired noise, beta = 0.1 steering is not a no-op, and V0 and V3 differ."""
    baseline = _actions(stacks["repo_V0"], _KEY)
    repo = _actions(stacks["repo_V0"], _KEY, _steer(0.1))
    paper = _actions(stacks["paper_V3"], _KEY, _steer(0.1))
    diffs = {
        "repo_vs_base": _max_abs(repo, baseline),
        "paper_vs_base": _max_abs(paper, baseline),
        "repo_vs_paper": _max_abs(repo, paper),
    }
    print(f"\n[sanity] beta=0.1 max|diff| on same noise: {diffs}")
    assert all(d > 0 for d in diffs.values())

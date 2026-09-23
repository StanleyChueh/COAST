"""Unit tests for src/openpi/serving/noise_control.py (controlled / paired flow noise).

Pure-Python / torch-on-CPU — no GPU or checkpoint required. The integration
tests drive a real ``Policy`` (with a recording input transform) wrapped as in
``scripts/serve_policy.py`` — NoiseControlledPolicyWrapper(SteeredPolicyWrapper(Policy))
— around a fake PyTorch model whose samplers record the ``noise`` they receive.

Covers:
- payload validation and the deterministic noise derivation / fingerprint
- the same explicit noise tensor reaches ``sample_actions`` and
  ``sample_actions_with_steering`` (the paired-noise guarantee)
- a different noise key changes the noise that reaches the sampler
- noise control off (no key, or no wrapper) leaves the previous call path unchanged
- neither ``__steering__`` nor ``__noise_control__`` reaches the input transforms

Numerical checks on the real pi0.5 checkpoint live in
tests/models/test_noise_control_gpu.py (manual).
"""

from __future__ import annotations

import pathlib
import types

import numpy as np
from openpi_client.noise_control import NOISE_CONTROL_ECHO_KEY
from openpi_client.noise_control import NOISE_CONTROL_KEY
from openpi_client.noise_control import NOISE_KEY_FIELDS
from openpi_client.noise_control import build_noise_control_payload
import pytest
import torch

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.serving import noise_control
from openpi.serving.noise_control import NoiseControlledPolicyWrapper
from openpi.serving.noise_control import derive_noise
from openpi.serving.noise_control import noise_fingerprint
from openpi.serving.noise_control import noise_shape_from_policy
from openpi.serving.noise_control import validate_noise_control_payload
from openpi.serving.steering import SteeredPolicyWrapper

_HORIZON, _DIM = 10, 32
_KEY = {"master_seed": 100, "task_id": 2, "init_state": 15, "rollout_step": 0}
_STEER = {"task": "taskA", "layer": 5, "alpha": 0.5, "beta": 0.1, "strategy": "global"}

# ═══════════════════════════════════════════════════════════════════════════════
# Protocol, validation, derivation
# ═══════════════════════════════════════════════════════════════════════════════


def test_wire_constants():
    assert NOISE_CONTROL_KEY == "__noise_control__"
    assert NOISE_CONTROL_ECHO_KEY == "noise_control"
    assert NOISE_KEY_FIELDS == ("master_seed", "task_id", "init_state", "rollout_step")


def test_build_payload_matches_schema():
    payload = build_noise_control_payload(master_seed=100, task_id=2, init_state=15, rollout_step=0)
    assert payload == _KEY
    validate_noise_control_payload(payload)


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        {k: v for k, v in _KEY.items() if k != "rollout_step"},
        {**_KEY, "extra": 1},
        {**_KEY, "master_seed": True},
        {**_KEY, "task_id": 2.0},
        {**_KEY, "init_state": "15"},
        {**_KEY, "rollout_step": -5},
    ],
)
def test_invalid_payload_rejected(bad):
    with pytest.raises(ValueError, match=NOISE_CONTROL_KEY):
        validate_noise_control_payload(bad)


def test_numpy_integer_payload_accepted():
    validate_noise_control_payload({k: np.int64(v) for k, v in _KEY.items()})


def test_derive_noise_is_deterministic_float32_with_requested_shape():
    a = derive_noise(dict(_KEY), _HORIZON, _DIM)
    b = derive_noise(dict(_KEY), _HORIZON, _DIM)
    assert a.shape == (_HORIZON, _DIM)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)
    assert noise_fingerprint(a) == noise_fingerprint(b)


def test_derive_noise_golden_fingerprint():
    """Pins the schedule: SeedSequence([100, 2, 15, 0]) -> PCG64 -> standard_normal float32 (10, 32).

    If this fails, the noise stream changed (e.g. a NumPy upgrade) and
    controlled-noise runs are no longer comparable with earlier ones.
    """
    noise = derive_noise(dict(_KEY), _HORIZON, _DIM)
    assert noise_fingerprint(noise) == "87ae314ae03d4e549d3bfa442c28cb94f594bff03d5c196422ae8b165ab113a0"


@pytest.mark.parametrize("field", NOISE_KEY_FIELDS)
def test_each_key_field_changes_noise(field):
    base = derive_noise(dict(_KEY), _HORIZON, _DIM)
    other = derive_noise({**_KEY, field: _KEY[field] + 1}, _HORIZON, _DIM)
    assert not np.array_equal(base, other)
    assert noise_fingerprint(base) != noise_fingerprint(other)


def test_derive_noise_is_standard_normal():
    samples = np.concatenate(
        [derive_noise({**_KEY, "rollout_step": 5 * i}, _HORIZON, _DIM).ravel() for i in range(200)]
    )
    assert abs(samples.mean()) < 0.02
    assert abs(samples.std() - 1.0) < 0.02


def test_fingerprint_is_over_float32_bytes():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert noise_fingerprint(x) == noise_fingerprint(x.astype(np.float64))
    assert noise_fingerprint(x) != noise_fingerprint(x + 1e-6)


def test_noise_shape_from_pytorch_style_policy():
    policy = types.SimpleNamespace(
        _model=types.SimpleNamespace(config=types.SimpleNamespace(action_horizon=10, action_dim=32))
    )
    assert noise_shape_from_policy(policy) == (10, 32)


def test_noise_shape_from_jax_style_policy():
    policy = types.SimpleNamespace(_model=types.SimpleNamespace(action_horizon=50, action_dim=32))
    assert noise_shape_from_policy(policy) == (50, 32)


def test_noise_shape_from_policy_without_model_raises():
    with pytest.raises(ValueError, match="action_horizon"):
        noise_shape_from_policy(types.SimpleNamespace())


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: real Policy + fake PyTorch pi0.5-like model
# ═══════════════════════════════════════════════════════════════════════════════


class _RecordingModel(torch.nn.Module):
    """Stands in for PI0Pytorch: records the ``noise`` argument each sampler receives."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(action_horizon=_HORIZON, action_dim=_DIM)
        self.sample_actions_noise: list = []
        self.steering_noise: list = []
        self.steering_hooks: list = []

    def _actions(self, noise, bsize):
        if noise is None:  # emulate PI0Pytorch.sample_noise
            noise = torch.normal(0.0, 1.0, size=(bsize, _HORIZON, _DIM), dtype=torch.float32)
        return noise * 2.0

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        self.sample_actions_noise.append(noise)
        return self._actions(noise, observation.state.shape[0])

    def sample_actions_with_steering(self, device, observation, *, noise=None, num_steps=10, steering_hooks=None):
        self.steering_noise.append(noise)
        self.steering_hooks.append(steering_hooks)
        return self._actions(noise, observation.state.shape[0]), {}


class _RecordingInputTransform:
    """First input transform: records the keys it sees, emits a minimal valid model input."""

    def __init__(self):
        self.seen_keys: list[set] = []

    def __call__(self, data: dict) -> dict:
        self.seen_keys.append(set(data))
        return {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {"base_0_rgb": np.zeros((4, 4, 3), dtype=np.float32)},
            "image_mask": {"base_0_rgb": np.ones((), dtype=bool)},
        }


@pytest.fixture
def npz_path(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "mini.npz"
    np.savez(path, **{"taskA__L5__0.5__C_contrastive": np.eye(8, dtype=np.float32) * 0.5})
    return path


@pytest.fixture
def stack(npz_path):
    """NoiseControlledPolicyWrapper(SteeredPolicyWrapper(Policy)), as built by serve_policy.py."""
    model = _RecordingModel()
    recorder = _RecordingInputTransform()
    policy = _policy.Policy(
        model,
        model_type=_model.ModelType.PI05,
        transforms=[recorder],
        is_pytorch=True,
        pytorch_device="cpu",
    )
    steered = SteeredPolicyWrapper(policy, conceptor_npz_path=npz_path, device="cpu")
    horizon, dim = noise_shape_from_policy(policy)
    wrapped = NoiseControlledPolicyWrapper(steered, action_horizon=horizon, action_dim=dim)
    return types.SimpleNamespace(model=model, recorder=recorder, policy=policy, steered=steered, wrapped=wrapped)


def _obs(**extra) -> dict:
    return {"observation/state": np.arange(8, dtype=np.float32), "prompt": "turn on the stove", **extra}


def test_paired_noise_reaches_both_samplers_identically(stack):
    """(B) The identical explicit noise tensor reaches sample_actions and sample_actions_with_steering."""
    expected = torch.from_numpy(derive_noise(dict(_KEY), _HORIZON, _DIM))[None, ...]

    base = stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: dict(_KEY)}))
    steered = stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: dict(_KEY), "__steering__": dict(_STEER)}))

    (base_noise,) = stack.model.sample_actions_noise
    (steer_noise,) = stack.model.steering_noise
    for received in (base_noise, steer_noise):
        assert isinstance(received, torch.Tensor)
        assert received.shape == (1, _HORIZON, _DIM)
        assert received.dtype == torch.float32
        assert torch.equal(received, expected)
    assert torch.equal(base_noise, steer_noise)
    # The steering hook was actually attached on the steered path.
    ((layer, _hook),) = stack.model.steering_hooks[0]
    assert layer == 5
    # Fake model is actions = 2 * noise, so outputs are identical too.
    np.testing.assert_array_equal(base["actions"], steered["actions"])
    # The echo fingerprints the exact array that was passed down.
    expected_sha = noise_fingerprint(expected[0].numpy())
    assert base[NOISE_CONTROL_ECHO_KEY]["sha256"] == expected_sha
    assert steered[NOISE_CONTROL_ECHO_KEY]["sha256"] == expected_sha
    assert base[NOISE_CONTROL_ECHO_KEY] == {"sha256": expected_sha, "shape": [_HORIZON, _DIM], **_KEY}


def test_same_key_is_reproducible_through_stack(stack):
    """(A, plumbing) Repeating a key repeats the noise — independent of the global torch RNG state."""
    first = stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: dict(_KEY)}))
    torch.manual_seed(12345)  # would change sample_noise() draws, must not matter here
    second = stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: dict(_KEY)}))
    np.testing.assert_array_equal(first["actions"], second["actions"])
    assert torch.equal(*stack.model.sample_actions_noise)


def test_different_master_seed_changes_sampler_noise(stack):
    """(D, plumbing) A different master seed yields different noise at the sampler."""
    a = stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: dict(_KEY)}))
    b = stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: {**_KEY, "master_seed": 200}}))
    assert not torch.equal(*stack.model.sample_actions_noise)
    assert not np.array_equal(a["actions"], b["actions"])
    assert a[NOISE_CONTROL_ECHO_KEY]["sha256"] != b[NOISE_CONTROL_ECHO_KEY]["sha256"]


@pytest.mark.parametrize("steer", [False, True])
def test_magic_keys_do_not_reach_input_transforms(stack, steer):
    """(F) Neither __noise_control__ nor __steering__ is visible to the policy's input transforms."""
    obs = _obs(**{NOISE_CONTROL_KEY: dict(_KEY)})
    if steer:
        obs["__steering__"] = dict(_STEER)
    stack.wrapped.infer(obs)
    (seen,) = stack.recorder.seen_keys
    assert seen == {"observation/state", "prompt"}
    # The caller's obs dict is not mutated.
    assert NOISE_CONTROL_KEY in obs
    assert ("__steering__" in obs) == steer


@pytest.mark.parametrize("steer", [False, True])
def test_no_noise_key_preserves_previous_behavior(stack, steer):
    """(E) Without the key, the wrapper passes through and the sampler draws its own noise (noise=None)."""
    obs = _obs(**({"__steering__": dict(_STEER)} if steer else {}))
    result = stack.wrapped.infer(obs)
    received = stack.model.steering_noise if steer else stack.model.sample_actions_noise
    assert received == [None]
    assert NOISE_CONTROL_ECHO_KEY not in result
    assert stack.recorder.seen_keys == [{"observation/state", "prompt"}]


@pytest.mark.parametrize("steer", [False, True])
def test_without_wrapper_policy_api_unchanged(stack, steer):
    """(E) Without the wrapper (server run without --noise_control) the steering stack is as before."""
    obs = _obs(**({"__steering__": dict(_STEER)} if steer else {}))
    result = stack.steered.infer(obs)
    received = stack.model.steering_noise if steer else stack.model.sample_actions_noise
    assert received == [None]
    assert set(result) == {"state", "actions", "policy_timing"}


def test_policy_infer_with_steering_explicit_noise_matches_infer(stack):
    """Policy.infer and Policy.infer_with_steering convert the same numpy noise to the same tensor."""
    noise = derive_noise(dict(_KEY), _HORIZON, _DIM)
    stack.policy.infer(_obs(), noise=noise)
    stack.policy.infer_with_steering(_obs(), steering_hooks=None, noise=noise)
    assert torch.equal(stack.model.sample_actions_noise[0], stack.model.steering_noise[0])


def test_steered_wrapper_forwards_noise_only_when_given():
    """SteeredPolicyWrapper must not pass noise= to inner policies when none is given (old stub API)."""

    class _OldApiPolicy:
        metadata: dict = {}  # noqa: RUF012

        def infer(self, obs):  # no noise kwarg
            return {"actions": np.zeros(1)}

    class _Wrapper(SteeredPolicyWrapper):
        def __init__(self, policy):  # skip NPZ loading
            self._policy = policy

    _Wrapper(_OldApiPolicy()).infer({"observation/state": np.zeros(8)})


def test_wrapper_rejects_batched_observations(stack):
    obs = {"observation/state": np.zeros((2, 8), dtype=np.float32), NOISE_CONTROL_KEY: dict(_KEY)}
    with pytest.raises(ValueError, match="unbatched"):
        stack.wrapped.infer(obs)


def test_wrapper_validates_payload(stack):
    with pytest.raises(ValueError, match="rollout_step"):
        stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: {**_KEY, "rollout_step": -1}}))
    assert stack.model.sample_actions_noise == []


def test_wrapper_metadata(stack):
    meta = stack.wrapped.metadata
    assert meta["noise_control_enabled"] is True
    assert meta["noise_control_shape"] == [_HORIZON, _DIM]
    assert meta["noise_control_scheme"] == noise_control.NOISE_SCHEME
    assert meta["steering_enabled"] is True  # underlying metadata preserved


def test_fast_steering_rejects_explicit_noise(npz_path):
    class _FastStub:
        _model_type = _model.ModelType.PI0_FAST
        _sample_kwargs: dict = {}  # noqa: RUF012
        metadata: dict = {}  # noqa: RUF012

    wrapper = SteeredPolicyWrapper.__new__(SteeredPolicyWrapper)
    wrapper._policy = _FastStub()  # noqa: SLF001
    wrapper._available_tasks = {"taskA"}  # noqa: SLF001
    wrapper._is_fast = True  # noqa: SLF001
    with pytest.raises(NotImplementedError, match="pi0-fast"):
        wrapper.infer({"__steering__": dict(_STEER)}, noise=np.zeros((_HORIZON, _DIM), dtype=np.float32))

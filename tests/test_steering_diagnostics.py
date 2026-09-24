"""Unit tests for the research-only steering intervention diagnostics.

Pure-Python / torch-on-CPU — no GPU or checkpoint required. Covers
``openpi.serving.steering.intervention_summary``, the ``record_diagnostics``
mode of ``ConceptorSteeringHook`` / ``LinearSteeringHook``, and
``SteeredPolicyWrapper(record_diagnostics=True)`` driving a real ``Policy``
around a fake model whose ``sample_actions_with_steering`` mirrors the
PI0Pytorch sampler loop (register hooks on expert layers, call
``set_denoise_step(t)`` before each of ``num_steps`` expert passes).

Requirements (Phase 1E):
1. steering disabled -> no diagnostics
2. steering enabled -> diagnostics produced (one record per hook application)
3. beta = 0 -> intervention magnitude ~0 (exactly 0 here)
4. all diagnostic values finite
5. no hidden states stored (records are scalars only)
6. steering behavior unchanged by diagnostics

The real-sampler / real-checkpoint counterpart is
tests/models/test_steering_diagnostics_gpu.py (manual).
"""

# ruff: noqa: N806
from __future__ import annotations

import json
import math
import pathlib
import types

import numpy as np
from openpi_client.noise_control import NOISE_CONTROL_KEY
from openpi_client.steering import STEERING_DIAGNOSTICS_KEY
from openpi_client.steering import STEERING_KEY
import pytest
import torch

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.serving.noise_control import NoiseControlledPolicyWrapper
from openpi.serving.steering import ConceptorSteeringHook
from openpi.serving.steering import LinearSteeringHook
from openpi.serving.steering import SteeredPolicyWrapper
from openpi.serving.steering import compute_random_conceptor
from openpi.serving.steering import intervention_summary

_D = 8  # hidden dim of the fake expert
_HORIZON, _ACTION_DIM = 10, 32
_NUM_STEPS = 10
_LAYER = 2
_TASK = "taskA"
_STEER = {"task": _TASK, "layer": _LAYER, "alpha": 0.5, "beta": 0.1, "strategy": "global"}
_KEY = {"master_seed": 100, "task_id": 2, "init_state": 15, "rollout_step": 0}

_SUMMARY_FIELDS = {
    "token_count",
    "hidden_dim",
    "mean_delta_norm",
    "max_delta_norm",
    "mean_relative_delta",
    "max_relative_delta",
    "mean_hidden_norm",
    "mean_cosine_delta_hidden",
    "mean_norm_ratio",
}
_CONCEPTOR_RECORD_FIELDS = {"layer", "denoising_step", "beta"} | _SUMMARY_FIELDS
_MAGNITUDE_FIELDS = (
    "mean_delta_norm",
    "max_delta_norm",
    "mean_relative_delta",
    "max_relative_delta",
    "mean_cosine_delta_hidden",
)


def _conceptor() -> np.ndarray:
    return compute_random_conceptor(d=_D, alpha=1.0, seed=3)


# ═══════════════════════════════════════════════════════════════════════════════
# intervention_summary
# ═══════════════════════════════════════════════════════════════════════════════


def test_summary_matches_independent_numpy_computation():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((2, 5, _D)).astype(np.float32)
    h_steered = h + rng.standard_normal((2, 5, _D)).astype(np.float32) * 0.1

    summary = intervention_summary(torch.from_numpy(h), torch.from_numpy(h_steered))

    delta = np.linalg.norm((h_steered - h).reshape(-1, _D), axis=-1)
    h_norm = np.linalg.norm(h.reshape(-1, _D), axis=-1)
    assert summary["token_count"] == 10
    assert summary["hidden_dim"] == _D
    assert summary["mean_delta_norm"] == pytest.approx(delta.mean(), rel=1e-6)
    assert summary["max_delta_norm"] == pytest.approx(delta.max(), rel=1e-6)
    assert summary["mean_relative_delta"] == pytest.approx((delta / h_norm).mean(), rel=1e-6)
    assert summary["max_relative_delta"] == pytest.approx((delta / h_norm).max(), rel=1e-6)
    assert summary["mean_hidden_norm"] == pytest.approx(h_norm.mean(), rel=1e-6)
    ratio = np.linalg.norm(h_steered.reshape(-1, _D), axis=-1) / h_norm
    assert summary["mean_norm_ratio"] == pytest.approx(ratio.mean(), rel=1e-6)
    d = (h_steered - h).reshape(-1, _D)
    cosine = (d * h.reshape(-1, _D)).sum(-1) / (delta * h_norm)
    assert summary["mean_cosine_delta_hidden"] == pytest.approx(cosine.mean(), rel=1e-5, abs=1e-6)


def test_summary_cosine_identifies_pure_shrinkage():
    """h' = 0.9 h is a pure shrink: cosine -1 and relative delta 0.1 for every token."""
    h = torch.randn(1, 10, _D)
    summary = intervention_summary(h, 0.9 * h)
    assert summary["mean_cosine_delta_hidden"] == pytest.approx(-1.0, abs=1e-6)
    assert summary["mean_relative_delta"] == pytest.approx(0.1, rel=1e-5)
    assert summary["mean_norm_ratio"] == pytest.approx(0.9, rel=1e-5)


def test_summary_of_bf16_tensors_measures_realized_delta():
    """For bf16 activations the delta is taken between the bf16 tensors the model actually sees."""
    h = torch.randn(1, 10, _D).to(torch.bfloat16)
    h_steered = torch.matmul(h, torch.from_numpy(_conceptor()).to(torch.bfloat16).T)
    summary = intervention_summary(h, h_steered)
    expected = torch.linalg.vector_norm(h_steered.float() - h.float(), dim=-1)
    assert summary["mean_delta_norm"] == pytest.approx(expected.mean().item(), rel=1e-6)


def test_summary_zero_hidden_state_is_finite():
    """(4) A zero token (||h|| = 0) must not produce NaN/inf in the relative delta."""
    h = torch.zeros(1, 3, _D)
    summary = intervention_summary(h, h.clone())
    assert all(math.isfinite(summary[k]) for k in _MAGNITUDE_FIELDS)
    assert summary["mean_relative_delta"] == 0.0
    assert summary["mean_norm_ratio"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Hooks
# ═══════════════════════════════════════════════════════════════════════════════


def _hooks(*, record: bool, beta: float = 0.1):
    """One hook per kind: global conceptor, per-step conceptor, linear."""
    C = _conceptor()
    return {
        "global": ConceptorSteeringHook(C, beta=beta, device="cpu", layer=_LAYER, record_diagnostics=record),
        "per_step": ConceptorSteeringHook(
            beta=beta,
            device="cpu",
            matrices_per_step=[C * (0.5 + 0.05 * t) for t in range(_NUM_STEPS)],
            layer=_LAYER,
            record_diagnostics=record,
        ),
        "linear": LinearSteeringHook(
            np.eye(_D, dtype=np.float32)[0], alpha=0.5, device="cpu", layer=_LAYER, record_diagnostics=record
        ),
    }


def _run_denoising_loop(hook, h_per_step):
    outputs = []
    for t, h in enumerate(h_per_step):
        hook.set_denoise_step(t)
        outputs.append(hook(None, None, h))
    return outputs


@pytest.mark.parametrize("kind", ["global", "per_step", "linear"])
def test_hook_records_nothing_by_default(kind):
    """(1) Diagnostics are opt-in: a default hook produces no records."""
    hook = _hooks(record=False)[kind]
    _run_denoising_loop(hook, [torch.randn(1, _HORIZON, _D) for _ in range(_NUM_STEPS)])
    assert hook.diagnostics == []


@pytest.mark.parametrize("kind", ["global", "per_step", "linear"])
def test_hook_records_one_entry_per_application(kind):
    """(2) One record per call, labeled with the layer and the sampler's denoising step."""
    hook = _hooks(record=True)[kind]
    _run_denoising_loop(hook, [torch.randn(1, _HORIZON, _D) for _ in range(_NUM_STEPS)])
    assert [r["denoising_step"] for r in hook.diagnostics] == list(range(_NUM_STEPS))
    assert all(r["layer"] == _LAYER for r in hook.diagnostics)
    assert all(r["token_count"] == _HORIZON and r["hidden_dim"] == _D for r in hook.diagnostics)
    assert all(r["mean_delta_norm"] > 0 for r in hook.diagnostics)


def test_hook_record_schema():
    hook = _hooks(record=True)["global"]
    hook(None, None, torch.randn(1, _HORIZON, _D))
    (record,) = hook.diagnostics
    assert set(record) == _CONCEPTOR_RECORD_FIELDS
    assert record["beta"] == 0.1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_beta_zero_intervention_is_zero(dtype):
    """(3) beta = 0 gives M = I exactly, so every magnitude is exactly zero."""
    hook = ConceptorSteeringHook(_conceptor(), beta=0.0, device="cpu", layer=_LAYER, record_diagnostics=True)
    _run_denoising_loop(hook, [torch.randn(1, _HORIZON, _D).to(dtype) for _ in range(_NUM_STEPS)])
    assert len(hook.diagnostics) == _NUM_STEPS
    for record in hook.diagnostics:
        for field in _MAGNITUDE_FIELDS:
            assert record[field] == 0.0
        assert record["mean_hidden_norm"] > 0


@pytest.mark.parametrize("kind", ["global", "per_step", "linear"])
def test_all_diagnostic_values_finite(kind):
    """(4) Every numeric value is finite, including for bf16 inputs and a zero token."""
    hook = _hooks(record=True)[kind]
    h_per_step = [torch.randn(1, _HORIZON, _D).to(torch.bfloat16) for _ in range(_NUM_STEPS)]
    h_per_step[3][0, 0] = 0.0
    _run_denoising_loop(hook, h_per_step)
    for record in hook.diagnostics:
        for value in record.values():
            if isinstance(value, float):
                assert math.isfinite(value)


@pytest.mark.parametrize("kind", ["global", "per_step", "linear"])
def test_no_hidden_states_stored(kind):
    """(5) Records hold JSON scalars only, and the hook keeps no activation tensors.

    The only tensors on the hook are the ones it had before any call (M / M_t / v).
    """
    hook = _hooks(record=True)[kind]
    tensor_attrs_before = {k for k, v in vars(hook).items() if isinstance(v, torch.Tensor)}
    _run_denoising_loop(hook, [torch.randn(1, _HORIZON, _D) for _ in range(_NUM_STEPS)])

    assert {k for k, v in vars(hook).items() if isinstance(v, torch.Tensor)} == tensor_attrs_before
    for record in hook.diagnostics:
        assert all(isinstance(v, int | float | str) or v is None for v in record.values())
        json.dumps(record)
    assert all(isinstance(n, float) for n in hook.intervention_norms)


@pytest.mark.parametrize("kind", ["global", "per_step", "linear"])
@pytest.mark.parametrize("tuple_output", [False, True])
def test_hook_output_identical_with_and_without_diagnostics(kind, tuple_output):
    """(6) Recording does not change the steered activations or the legacy norm log."""
    off, on = _hooks(record=False)[kind], _hooks(record=True)[kind]
    h_per_step = [torch.randn(1, _HORIZON, _D) for _ in range(_NUM_STEPS)]
    wrap = (lambda h: (h, "cache")) if tuple_output else (lambda h: h)
    out_off = _run_denoising_loop(off, [wrap(h) for h in h_per_step])
    out_on = _run_denoising_loop(on, [wrap(h) for h in h_per_step])
    for a, b in zip(out_off, out_on, strict=True):
        if tuple_output:
            assert a[1:] == b[1:] == ("cache",)
            assert torch.equal(a[0], b[0])
        else:
            assert torch.equal(a, b)
    assert off.intervention_norms == on.intervention_norms


def test_reset_logs_clears_diagnostics():
    hook = _hooks(record=True)["global"]
    hook(None, None, torch.randn(1, _HORIZON, _D))
    hook.reset_logs()
    assert hook.diagnostics == []
    assert hook.intervention_norms == []


# ═══════════════════════════════════════════════════════════════════════════════
# SteeredPolicyWrapper + real Policy + sampler-shaped fake model
# ═══════════════════════════════════════════════════════════════════════════════


class _ExpertLoopModel(torch.nn.Module):
    """Stands in for PI0Pytorch with the same hook contract as sample_actions_with_steering.

    Registers each (layer_idx, hook) on ``self.layers[layer_idx]``, calls
    ``set_denoise_step(t)`` before each of ``num_steps`` expert passes over
    ``action_horizon`` tokens, and removes the hooks afterwards.
    """

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(action_horizon=_HORIZON, action_dim=_ACTION_DIM)
        # Identical weights for every instance, without touching the global torch RNG.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            self.layers = torch.nn.ModuleList(torch.nn.Linear(_D, _D) for _ in range(4))
            self.out = torch.nn.Linear(_D, _ACTION_DIM)

    def _denoise(self, x_t, num_steps):
        dt = -1.0 / num_steps
        for t in range(num_steps):
            yield t
            h = x_t[..., :_D]
            for layer in self.layers:
                h = layer(h)
            x_t = x_t + dt * self.out(h)
        self.result = x_t

    def _noise(self, noise, bsize):
        if noise is None:
            return torch.normal(0.0, 1.0, size=(bsize, _HORIZON, _ACTION_DIM))
        return noise

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=_NUM_STEPS):
        for _ in self._denoise(self._noise(noise, observation.state.shape[0]), num_steps):
            pass
        return self.result

    @torch.no_grad()
    def sample_actions_with_steering(
        self, device, observation, *, noise=None, num_steps=_NUM_STEPS, steering_hooks=None
    ):
        handles = [self.layers[i].register_forward_hook(fn) for i, fn in (steering_hooks or [])]
        try:
            for t in self._denoise(self._noise(noise, observation.state.shape[0]), num_steps):
                for _, fn in steering_hooks or []:
                    fn.set_denoise_step(t)
        finally:
            for handle in handles:
                handle.remove()
        return self.result, {}


class _InputTransform:
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
    np.savez(
        path,
        **{
            f"{_TASK}__L{_LAYER}__0.5__C_contrastive": _conceptor(),
            f"{_TASK}__L{_LAYER}__linear_direction": np.eye(_D, dtype=np.float32)[0],
        },
    )
    return path


def _stack(npz_path, *, record: bool, noise_control: bool = False):
    transform = _InputTransform()
    policy = _policy.Policy(
        _ExpertLoopModel(),
        model_type=_model.ModelType.PI05,
        transforms=[transform],
        is_pytorch=True,
        pytorch_device="cpu",
    )
    wrapped = SteeredPolicyWrapper(policy, conceptor_npz_path=npz_path, device="cpu", record_diagnostics=record)
    if noise_control:
        wrapped = NoiseControlledPolicyWrapper(wrapped, action_horizon=_HORIZON, action_dim=_ACTION_DIM)
    return types.SimpleNamespace(wrapped=wrapped, transform=transform)


def _obs(**extra) -> dict:
    return {"observation/state": np.arange(8, dtype=np.float32), "prompt": "turn on the stove", **extra}


def test_wrapper_without_flag_produces_no_diagnostics(npz_path):
    """(1) Default server: steered responses carry no diagnostics key; metadata says disabled."""
    stack = _stack(npz_path, record=False)
    result = stack.wrapped.infer(_obs(**{STEERING_KEY: dict(_STEER)}))
    assert STEERING_DIAGNOSTICS_KEY not in result
    assert stack.wrapped.metadata["steering_diagnostics_enabled"] is False


def test_unsteered_request_produces_no_diagnostics(npz_path):
    """(1) Diagnostics server, no __steering__ key: nothing is attached."""
    stack = _stack(npz_path, record=True, noise_control=True)
    result = stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: dict(_KEY)}))
    assert STEERING_DIAGNOSTICS_KEY not in result


def test_steered_request_produces_one_record_per_denoising_step(npz_path):
    """(2) Through Policy.infer_with_steering: 10 records, steps 0..9, the payload's layer, horizon tokens."""
    stack = _stack(npz_path, record=True, noise_control=True)
    result = stack.wrapped.infer(_obs(**{STEERING_KEY: dict(_STEER), NOISE_CONTROL_KEY: dict(_KEY)}))
    records = result[STEERING_DIAGNOSTICS_KEY]
    assert [r["denoising_step"] for r in records] == list(range(_NUM_STEPS))
    assert {r["layer"] for r in records} == {_LAYER}
    assert {r["token_count"] for r in records} == {_HORIZON}
    assert {r["beta"] for r in records} == {_STEER["beta"]}
    assert all(set(r) == _CONCEPTOR_RECORD_FIELDS for r in records)
    assert all(r["mean_delta_norm"] > 0 for r in records)
    assert stack.wrapped.metadata["steering_diagnostics_enabled"] is True


def test_records_are_per_call_on_cached_hook(npz_path):
    """The cached hook is reset per call, so each response has only its own 10 records."""
    stack = _stack(npz_path, record=True)
    first = stack.wrapped.infer(_obs(**{STEERING_KEY: dict(_STEER)}))[STEERING_DIAGNOSTICS_KEY]
    second = stack.wrapped.infer(_obs(**{STEERING_KEY: dict(_STEER)}))[STEERING_DIAGNOSTICS_KEY]
    assert len(first) == len(second) == _NUM_STEPS


def test_linear_strategy_records_alpha_and_layer(npz_path):
    stack = _stack(npz_path, record=True)
    payload = {**_STEER, "strategy": "linear"}
    records = stack.wrapped.infer(_obs(**{STEERING_KEY: payload}))[STEERING_DIAGNOSTICS_KEY]
    assert len(records) == _NUM_STEPS
    assert {r["layer"] for r in records} == {_LAYER}
    assert {r["alpha"] for r in records} == {_STEER["alpha"]}


def test_beta_zero_through_stack_is_zero_and_matches_unsteered(npz_path):
    """(3) beta = 0 through the full stack: zero magnitudes and actions equal to the unsteered call."""
    stack = _stack(npz_path, record=True, noise_control=True)
    base = stack.wrapped.infer(_obs(**{NOISE_CONTROL_KEY: dict(_KEY)}))
    steered = stack.wrapped.infer(_obs(**{STEERING_KEY: {**_STEER, "beta": 0.0}, NOISE_CONTROL_KEY: dict(_KEY)}))
    for record in steered[STEERING_DIAGNOSTICS_KEY]:
        assert all(record[field] == 0.0 for field in _MAGNITUDE_FIELDS)
    np.testing.assert_array_equal(base["actions"], steered["actions"])


def test_diagnostics_do_not_change_actions(npz_path):
    """(6) Same noise, same obs: identical actions with the diagnostics flag on and off."""
    obs = _obs(**{STEERING_KEY: dict(_STEER), NOISE_CONTROL_KEY: dict(_KEY)})
    off = _stack(npz_path, record=False, noise_control=True).wrapped.infer(dict(obs))
    on = _stack(npz_path, record=True, noise_control=True).wrapped.infer(dict(obs))
    np.testing.assert_array_equal(off["actions"], on["actions"])
    assert set(on) - set(off) == {STEERING_DIAGNOSTICS_KEY}


def test_diagnostics_are_json_scalars_and_keys_do_not_reach_transforms(npz_path):
    """(5) The response payload is scalars only; magic keys never reach input transforms."""
    stack = _stack(npz_path, record=True, noise_control=True)
    result = stack.wrapped.infer(_obs(**{STEERING_KEY: dict(_STEER), NOISE_CONTROL_KEY: dict(_KEY)}))
    json.dumps(result[STEERING_DIAGNOSTICS_KEY])
    assert stack.transform.seen_keys == [{"observation/state", "prompt"}]


def test_pi0_fast_rejects_diagnostics(npz_path):
    policy = types.SimpleNamespace(_model_type=_model.ModelType.PI0_FAST, metadata={})
    with pytest.raises(ValueError, match="pi0-fast"):
        SteeredPolicyWrapper(policy, conceptor_npz_path=npz_path, device="cpu", record_diagnostics=True)

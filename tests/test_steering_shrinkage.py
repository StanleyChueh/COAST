"""Unit tests for the Phase 2A ``shrinkage`` research ablation.

``shrinkage`` is ``global`` steering with the conceptor replaced by zero:
M = (1-β)I + β·0 = (1-β)I, so h' = (1-β)h. Pure-Python / torch-on-CPU.

Requirements (Phase 2A):
1. original COAST (``global``) behavior is unchanged
2. shrinkage builds exactly M = 0.9 I at β = 0.1
3. β = 0 gives the identity for every conceptor-matrix strategy
4. shrinkage never reads a conceptor array from the NPZ
5. existing steering tests pass (tests/test_steering.py, tests/client/test_steering.py)

The real-checkpoint counterpart is tests/models/test_steering_shrinkage_gpu.py (manual).
"""

# ruff: noqa: N802, N806, RUF002
from __future__ import annotations

import logging
import pathlib

import numpy as np
from openpi_client.steering import ALLOWED_STRATEGIES
from openpi_client.steering import STEERING_DIAGNOSTICS_KEY
from openpi_client.steering import STEERING_KEY
import pytest
import torch

from openpi.serving import steering
from openpi.serving.steering import ConceptorSteeringHook
from openpi.serving.steering import ShrinkageSteeringHook
from openpi.serving.steering import SteeredPolicyWrapper
from openpi.serving.steering import compute_random_conceptor

_D = 8
_TASK = "taskA"
_LAYER = 5


@pytest.fixture
def npz_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Tiny NPZ with every per-strategy key for (taskA, L5)."""
    C = compute_random_conceptor(d=_D, alpha=1.0, seed=3)
    arrays = {
        f"{_TASK}__L{_LAYER}__0.5__C_contrastive": C,
        f"{_TASK}__L{_LAYER}__0.5__C_success": C * 0.8,
        f"{_TASK}__L{_LAYER}__linear_direction": np.eye(_D, dtype=np.float32)[0],
    }
    for t in range(10):
        arrays[f"{_TASK}__L{_LAYER}__per_step_{t}__C_contrastive"] = C * (0.5 + 0.05 * t)
    path = tmp_path / "mini.npz"
    np.savez(path, **arrays)
    return path


class _StubPolicy:
    """Records the hooks the wrapper passes to infer_with_steering."""

    def __init__(self):
        self.hooks = None
        self.metadata = {}

    def infer(self, obs):
        return {"actions": np.zeros((1, 4))}

    def infer_with_steering(self, obs, *, steering_hooks):
        self.hooks = steering_hooks
        h = torch.randn(1, 10, _D)
        for _, hook in steering_hooks:
            for t in range(10):
                hook.set_denoise_step(t)
                hook(None, None, h)
        return {"actions": np.ones((1, 4))}, {}


class _CountingNpz:
    """Proxy for the NpzFile that counts array reads (``npz[key]``); member listing stays free."""

    def __init__(self, npz):
        self._npz = npz
        self.reads: list[str] = []

    @property
    def files(self):
        return self._npz.files

    def __contains__(self, key):
        return key in self._npz

    def __getitem__(self, key):
        self.reads.append(key)
        return self._npz[key]


def _payload(strategy: str, beta: float = 0.1, alpha: float = 0.5) -> dict:
    return {"task": _TASK, "layer": _LAYER, "alpha": alpha, "beta": beta, "strategy": strategy}


def _hook_for(npz_path, payload, *, record: bool = False):
    policy = _StubPolicy()
    wrapper = SteeredPolicyWrapper(policy, conceptor_npz_path=npz_path, device="cpu", record_diagnostics=record)
    result = wrapper.infer({STEERING_KEY: payload})
    ((layer, hook),) = policy.hooks
    return wrapper, layer, hook, result


def test_shrinkage_is_an_allowed_wire_strategy():
    assert "shrinkage" in ALLOWED_STRATEGIES
    assert steering.ALLOWED_STRATEGIES is ALLOWED_STRATEGIES


# ── 1. COAST unchanged ────────────────────────────────────────────────────────


def test_global_still_builds_conceptor_hook_with_original_matrix(npz_path):
    """(1) ``global`` still yields ConceptorSteeringHook with M = (1-β)I + βC from the NPZ key."""
    _, layer, hook, _ = _hook_for(npz_path, _payload("global"))
    assert type(hook) is ConceptorSteeringHook
    assert layer == _LAYER
    C = np.load(npz_path)[f"{_TASK}__L{_LAYER}__0.5__C_contrastive"]
    expected = 0.9 * torch.eye(_D) + 0.1 * torch.from_numpy(C)
    assert torch.equal(hook.M, expected)


def test_global_output_is_h_times_M_transpose(npz_path):
    """(1) The COAST forward is still h @ M.T."""
    _, _, hook, _ = _hook_for(npz_path, _payload("global"))
    h = torch.randn(1, 10, _D)
    assert torch.equal(hook(None, None, h), torch.matmul(h, hook.M.T))


def test_shrinkage_and_global_are_distinct_cached_hooks(npz_path):
    policy = _StubPolicy()
    wrapper = SteeredPolicyWrapper(policy, conceptor_npz_path=npz_path, device="cpu")
    wrapper.infer({STEERING_KEY: _payload("global")})
    coast = policy.hooks[0][1]
    wrapper.infer({STEERING_KEY: _payload("shrinkage")})
    shrink = policy.hooks[0][1]
    assert coast is not shrink
    assert type(coast) is ConceptorSteeringHook
    assert type(shrink) is ShrinkageSteeringHook


# ── 2. M = 0.9 I ──────────────────────────────────────────────────────────────


def test_shrinkage_builds_exactly_point_nine_identity(npz_path):
    """(2) At β = 0.1 the ablation matrix is exactly 0.9·I (float32), sized to h's hidden dim."""
    _, layer, hook, _ = _hook_for(npz_path, _payload("shrinkage"))
    assert layer == _LAYER
    assert torch.equal(hook.M, torch.full((_D,), 0.9, dtype=torch.float32).diag())
    assert torch.equal(hook.M, (1.0 - 0.1) * torch.eye(_D))


def test_shrinkage_output_is_scaled_h():
    """(2) h' = 0.9·h exactly in float32 (one nonzero term per output element)."""
    hook = ShrinkageSteeringHook(beta=0.1, device="cpu")
    h = torch.randn(2, 10, _D)
    assert torch.equal(hook(None, None, h), h * torch.tensor(0.9, dtype=torch.float32))


def test_shrinkage_bf16_realized_factor():
    """In bf16 the applied factor is bf16(0.9) = 0.8984375 (documented, not adjusted)."""
    hook = ShrinkageSteeringHook(beta=0.1, device="cpu")
    h = torch.ones(1, 1, _D, dtype=torch.bfloat16)
    out = hook(None, None, h)
    assert out.dtype == torch.bfloat16
    assert torch.all(out.float() == 0.8984375)


def test_shrinkage_rebuilds_for_new_hidden_dim():
    hook = ShrinkageSteeringHook(beta=0.1, device="cpu")
    hook(None, None, torch.randn(1, 3, 4))
    hook(None, None, torch.randn(1, 3, 6))
    assert torch.equal(hook.M, 0.9 * torch.eye(6))


def test_shrinkage_ignores_alpha_in_cache_key():
    key_a = SteeredPolicyWrapper._cache_key(_payload("shrinkage", alpha=0.1))  # noqa: SLF001
    key_b = SteeredPolicyWrapper._cache_key(_payload("shrinkage", alpha=1.0))  # noqa: SLF001
    assert key_a == key_b
    assert SteeredPolicyWrapper._cache_key(_payload("global")) != key_a  # noqa: SLF001


def test_shrinkage_is_logged(npz_path, caplog):
    """The ablation is explicit in the server log: hook class at build, M at first application."""
    with caplog.at_level(logging.INFO, logger="openpi.serving.steering"):
        _hook_for(npz_path, _payload("shrinkage"))
    text = caplog.text
    assert "ShrinkageSteeringHook" in text
    assert "Shrinkage ablation: M = (1 - beta) I = 0.9 * I (d=8)" in text


def test_shrinkage_diagnostics(npz_path):
    """Diagnostics for a pure shrink: relative delta β, cosine -1, norm ratio 1-β."""
    _, _, _, result = _hook_for(npz_path, _payload("shrinkage"), record=True)
    records = result[STEERING_DIAGNOSTICS_KEY]
    assert [r["denoising_step"] for r in records] == list(range(10))
    for r in records:
        assert r["layer"] == _LAYER
        assert r["beta"] == 0.1
        assert r["mean_relative_delta"] == pytest.approx(0.1, rel=1e-5)
        assert r["mean_cosine_delta_hidden"] == pytest.approx(-1.0, abs=1e-6)
        assert r["mean_norm_ratio"] == pytest.approx(0.9, rel=1e-6)


# ── 3. β = 0 → identity for every conceptor-matrix strategy ─────────────────


@pytest.mark.parametrize("strategy", ["global", "positive_only", "random_matched", "per_step", "shrinkage"])
def test_beta_zero_is_identity_for_all_matrix_strategies(npz_path, strategy):
    """(3) β = 0: M == I exactly and h' == h. (``linear`` has no β; its no-op is α = 0, tested in test_steering.)"""
    _, _, hook, _ = _hook_for(npz_path, _payload(strategy, beta=0.0))
    matrices = hook._Ms if hook._Ms is not None else [hook.M]  # noqa: SLF001
    for M in matrices:
        assert torch.equal(M, torch.eye(_D))
    h = torch.randn(1, 10, _D)
    assert torch.equal(hook(None, None, h), h)


# ── 4. Shrinkage reads no conceptor ──────────────────────────────────────────


def test_shrinkage_reads_no_conceptor_array(npz_path):
    """(4) Building and running the shrinkage hook performs zero NPZ array reads; global performs one."""
    policy = _StubPolicy()
    wrapper = SteeredPolicyWrapper(policy, conceptor_npz_path=npz_path, device="cpu")
    counting = _CountingNpz(wrapper._npz)  # noqa: SLF001
    wrapper._npz = counting  # noqa: SLF001

    wrapper.infer({STEERING_KEY: _payload("shrinkage")})
    wrapper.infer({STEERING_KEY: _payload("shrinkage")})
    assert counting.reads == []

    wrapper.infer({STEERING_KEY: _payload("global")})
    assert counting.reads == [f"{_TASK}__L{_LAYER}__0.5__C_contrastive"]


def test_shrinkage_works_for_layer_without_conceptor(tmp_path):
    """(4) Shrinkage needs no conceptor for the requested layer; global does."""
    path = tmp_path / "other_layer.npz"
    np.savez(path, **{f"{_TASK}__L11__0.5__C_contrastive": np.eye(_D, dtype=np.float32)})
    policy = _StubPolicy()
    wrapper = SteeredPolicyWrapper(policy, conceptor_npz_path=path, device="cpu")
    wrapper.infer({STEERING_KEY: _payload("shrinkage")})
    assert isinstance(policy.hooks[0][1], ShrinkageSteeringHook)
    with pytest.raises(KeyError, match="not in NPZ"):
        wrapper.infer({STEERING_KEY: _payload("global")})


def test_hook_constructs_without_any_matrix():
    hook = ShrinkageSteeringHook(beta=0.1, device="cpu", layer=_LAYER, record_diagnostics=True)
    assert hook.M is None
    assert hook.layer == _LAYER
    assert "ShrinkageSteeringHook" in repr(hook)


def test_pi0_fast_rejects_shrinkage(npz_path):
    """pi0-fast has no shrinkage path; the fast matrix lookup rejects the strategy cleanly."""
    with pytest.raises(ValueError, match="Unknown pi0-fast steering strategy"):
        steering.build_fast_steering_stack(np.load(npz_path), _payload("shrinkage"), max_decoding_steps=6)


def test_shrinkage_payload_validates_like_other_strategies(npz_path):
    steering.validate_steering_payload(_payload("shrinkage"), [_TASK])
    with pytest.raises(ValueError, match="not found in conceptor NPZ"):
        steering.validate_steering_payload({**_payload("shrinkage"), "task": "unknown"}, [_TASK])

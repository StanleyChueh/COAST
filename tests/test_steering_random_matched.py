"""Unit tests for the Phase 2B ``random_matched`` conceptor ablation.

``random_matched`` keeps the eigenvalue spectrum of the COAST conceptor and replaces its
eigenvectors with a deterministic random orthogonal basis:
C_sym = (C + C^T)/2 = U Λ U^T  →  C_random = Q Λ Q^T,  Q from QR of a seeded Gaussian matrix,
then applies M = (1-β)I + β·C_random through the unchanged ConceptorSteeringHook. Pure-Python / torch-on-CPU.

Requirements (Phase 2B):
1. the random matched conceptor preserves the eigenvalues
2. it changes the eigenvectors
3. the same seed gives an identical matrix
4. a different seed gives different eigenvectors
5. random_matched does not alter the original COAST (``global``) mode
6. shrinkage still works
"""

# ruff: noqa: N802, N803, N806, RUF002
from __future__ import annotations

import logging
import pathlib

import numpy as np
from openpi_client.steering import STEERING_KEY
import pytest
import torch

from openpi.serving import steering
from openpi.serving.conceptors import boolean_and
from openpi.serving.conceptors import boolean_not
from openpi.serving.conceptors import conceptor
from openpi.serving.conceptors import correlation_matrix
from openpi.serving.conceptors import random_matched_conceptor
from openpi.serving.steering import ConceptorSteeringHook
from openpi.serving.steering import ShrinkageSteeringHook
from openpi.serving.steering import SteeredPolicyWrapper

_D = 64
_K = 4  # top-k eigen-subspace compared; chance overlap is _K / _D
_TASK = "taskA"
_LAYER = 5
_KEY = f"{_TASK}__L{_LAYER}__0.5__C_contrastive"


def _contrastive_conceptor(seed: int = 0) -> np.ndarray:
    """A realistic C_contrastive = C_s AND NOT C_f from anisotropic data (float32, not exactly symmetric)."""
    rng = np.random.default_rng(seed)
    scales = np.geomspace(3.0, 0.05, _D)
    basis, _ = np.linalg.qr(rng.standard_normal((_D, _D)))
    X_s = (rng.standard_normal((2000, _D)) * scales) @ basis.T
    X_f = (rng.standard_normal((2000, _D)) * scales[::-1]) @ basis.T
    C_s = conceptor(correlation_matrix(X_s), alpha=0.5)
    C_f = conceptor(correlation_matrix(X_f), alpha=0.5)
    return boolean_and(C_s, boolean_not(C_f)).astype(np.float32)


@pytest.fixture
def C_real() -> np.ndarray:
    return _contrastive_conceptor()


@pytest.fixture
def npz_path(tmp_path: pathlib.Path, C_real: np.ndarray) -> pathlib.Path:
    path = tmp_path / "mini.npz"
    np.savez(path, **{_KEY: C_real, f"{_TASK}__L{_LAYER}__0.5__C_success": 0.8 * C_real})
    return path


class _StubPolicy:
    def __init__(self):
        self.hooks = None
        self.metadata = {}

    def infer(self, obs):
        return {"actions": np.zeros((1, 4))}

    def infer_with_steering(self, obs, *, steering_hooks):
        self.hooks = steering_hooks
        return {"actions": np.ones((1, 4))}, {}


def _payload(strategy: str, beta: float = 0.1, alpha: float = 0.5) -> dict:
    return {"task": _TASK, "layer": _LAYER, "alpha": alpha, "beta": beta, "strategy": strategy}


def _hook(wrapper: SteeredPolicyWrapper, strategy: str, **kw):
    wrapper.infer({STEERING_KEY: _payload(strategy, **kw)})
    ((layer, hook),) = wrapper._policy.hooks  # noqa: SLF001
    assert layer == _LAYER
    return hook


def _wrapper(npz_path) -> SteeredPolicyWrapper:
    return SteeredPolicyWrapper(_StubPolicy(), conceptor_npz_path=npz_path, device="cpu")


def _C_from_M(M: torch.Tensor, beta: float = 0.1) -> np.ndarray:
    """Invert M = (1-β)I + βC for the conceptor actually used by the hook."""
    M = M.double().numpy()
    return (M - (1.0 - beta) * np.eye(M.shape[0])) / beta


def _eig_desc(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Cd = C.astype(np.float64)
    w, V = np.linalg.eigh(0.5 * (Cd + Cd.T))
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def _subspace_overlap(A: np.ndarray, B: np.ndarray, k: int = _K) -> float:
    """||U_k^T V_k||_F^2 / k for the top-k eigenvectors: 1 = same subspace, k/d = chance."""
    U, V = _eig_desc(A)[1], _eig_desc(B)[1]
    return float(np.linalg.norm(U[:, :k].T @ V[:, :k]) ** 2 / k)


def _wrapper_seed() -> int:
    return steering._stable_random_seed((_TASK, _LAYER, 0.5, "random_matched"))  # noqa: SLF001


def test_fixture_is_a_contrastive_conceptor_with_a_separated_top_subspace(C_real):
    w, _ = _eig_desc(C_real)
    assert 0.0 < w[0] < 1.0
    assert w[_K - 1] - w[_K] > 1e-2  # eigengap: the top-k subspace is well defined, so overlaps are meaningful
    assert not np.array_equal(C_real, C_real.T)  # like the real NPZ conceptor


# ── 1. eigenvalues preserved ─────────────────────────────────────────────────


def test_random_matched_preserves_eigenvalues(npz_path, C_real):
    """(1) The hook's C_random has exactly the spectrum of sym(C_real)."""
    hook = _hook(_wrapper(npz_path), "random_matched")
    C_rand = _C_from_M(hook.M)
    np.testing.assert_allclose(_eig_desc(C_rand)[0], _eig_desc(C_real)[0], atol=5e-6)
    # Direct generator output (before the M round trip) matches to float32 precision.
    C_direct = random_matched_conceptor(C_real, seed=_wrapper_seed())
    np.testing.assert_allclose(_eig_desc(C_direct)[0], _eig_desc(C_real)[0], atol=1e-6)
    assert np.trace(C_direct) == pytest.approx(np.trace(C_real), rel=1e-5)


def test_random_matched_is_symmetric(C_real):
    C_rand = random_matched_conceptor(C_real, seed=_wrapper_seed())
    assert np.array_equal(C_rand, C_rand.T)


# ── 2. eigenvectors changed ──────────────────────────────────────────────────


def test_random_matched_changes_eigenvectors(npz_path, C_real):
    """(2) Top-k eigen-subspace overlap with C_real drops from 1 to near chance (k/d = 1/16)."""
    hook = _hook(_wrapper(npz_path), "random_matched")
    C_rand = _C_from_M(hook.M)
    assert _subspace_overlap(C_real, C_real) == pytest.approx(1.0)
    assert _subspace_overlap(C_real, C_rand) < 0.25
    assert np.linalg.norm(C_rand - C_real) > 0.5 * np.linalg.norm(C_real)


def test_qr_sign_fix_does_not_change_C_random(C_real):
    """Flipping column signs of Q (the QR sign-ambiguity fix) cancels in Q Λ Q^T: result is identical."""
    seed = _wrapper_seed()
    Cd = C_real.astype(np.float64)
    lam = np.linalg.eigvalsh(0.5 * (Cd + Cd.T))
    Q, R = np.linalg.qr(np.random.default_rng(seed).standard_normal((_D, _D)))
    assert np.abs(Q.T @ Q - np.eye(_D)).max() < 1e-12
    Q_fixed = Q * np.sign(np.diag(R))[None, :]
    C_fixed = ((Q_fixed * lam[None, :]) @ Q_fixed.T).astype(np.float32)
    assert np.array_equal(C_fixed, random_matched_conceptor(C_real, seed=seed))


# ── 3. same seed → identical matrix ──────────────────────────────────────────


def test_same_seed_gives_identical_matrix(npz_path, C_real):
    """(3) Two independent wrappers build bit-identical M; the generator is deterministic."""
    M1 = _hook(_wrapper(npz_path), "random_matched").M
    M2 = _hook(_wrapper(npz_path), "random_matched").M
    assert torch.equal(M1, M2)
    assert np.array_equal(random_matched_conceptor(C_real, seed=11), random_matched_conceptor(C_real, seed=11))


def test_wrapper_uses_the_documented_seed_and_logs_it(npz_path, C_real, caplog):
    with caplog.at_level(logging.INFO, logger="openpi.serving.steering"):
        hook = _hook(_wrapper(npz_path), "random_matched")
    seed = _wrapper_seed()
    expected = ConceptorSteeringHook(random_matched_conceptor(C_real, seed=seed), beta=0.1, device="cpu").M
    assert torch.equal(hook.M, expected)
    assert f"random_matched: seed={seed}" in caplog.text


# ── 4. different seed → different eigenvectors ───────────────────────────────


def test_different_seed_gives_different_eigenvectors(C_real):
    """(4) Same spectrum, different eigen-subspace across seeds."""
    a = random_matched_conceptor(C_real, seed=1)
    b = random_matched_conceptor(C_real, seed=2)
    np.testing.assert_allclose(_eig_desc(a)[0], _eig_desc(b)[0], atol=1e-6)
    assert _subspace_overlap(a, b) < 0.25
    assert np.linalg.norm(a - b) > 0.5 * np.linalg.norm(a)


# ── 5. global (original COAST) unaffected ────────────────────────────────────


def test_random_matched_does_not_alter_global(npz_path, C_real):
    """(5) global's M is 0.9I + 0.1·C_real from the NPZ, bit-identical before and after random_matched is built."""
    expected = 0.9 * torch.eye(_D) + 0.1 * torch.from_numpy(C_real)

    w = _wrapper(npz_path)
    coast_before = _hook(w, "global")
    assert type(coast_before) is ConceptorSteeringHook
    assert torch.equal(coast_before.M, expected)

    rand = _hook(w, "random_matched")
    coast_after = _hook(w, "global")
    assert coast_after is coast_before  # same cached hook, not rebuilt
    assert torch.equal(coast_after.M, expected)
    assert rand is not coast_before
    assert not torch.equal(rand.M, coast_before.M)

    # A fresh wrapper that only ever builds global gives the same matrix; the NPZ array is untouched.
    assert torch.equal(_hook(_wrapper(npz_path), "global").M, expected)
    assert np.array_equal(np.load(npz_path)[_KEY], C_real)
    h = torch.randn(1, 10, _D)
    assert torch.equal(coast_after(None, None, h), torch.matmul(h, expected.T))


def test_random_matched_uses_same_hook_class_and_beta(npz_path):
    hook = _hook(_wrapper(npz_path), "random_matched")
    assert type(hook) is ConceptorSteeringHook
    assert hook.beta == 0.1
    assert torch.equal(_hook(_wrapper(npz_path), "random_matched", beta=0.0).M, torch.eye(_D))


# ── 6. shrinkage still works ─────────────────────────────────────────────────


def test_shrinkage_still_works_alongside_random_matched(npz_path):
    """(6) In the same server wrapper, shrinkage is still M = 0.9·I, h' = 0.9·h, independent of the other hooks."""
    w = _wrapper(npz_path)
    _hook(w, "random_matched")
    _hook(w, "global")
    shrink = _hook(w, "shrinkage")
    assert type(shrink) is ShrinkageSteeringHook
    h = torch.randn(1, 10, _D)
    assert torch.equal(shrink(None, None, h), h * torch.tensor(0.9, dtype=torch.float32))
    assert torch.equal(shrink.M, 0.9 * torch.eye(_D))
    assert len(w._hook_cache) == 3  # noqa: SLF001

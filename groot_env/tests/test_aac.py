"""Tests for groot_env/aac.py (pure AAC math; CPU only).

Parity: `fixtures/aac_upstream_parity.npz` holds synthetic candidate sets plus the
outputs of the OFFICIAL upstream `select_chunk_size` (Adaptive-Action-Chunking/robocasa
@ fed3e6b), produced by experiments/aac/make_upstream_parity_fixtures.py. If the
upstream clone is present under external/aac/, the live-parity test re-runs upstream
directly as well.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import aac

H = 16
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "aac_upstream_parity.npz"
UPSTREAM = (
    pathlib.Path(__file__).resolve().parents[2] / "external" / "aac" / "aac-robocasa"
)


def _cands(n=20, *, pos_std=None, rot_std=None, grip_p=None, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pos_std = np.full(H, 0.05) if pos_std is None else np.asarray(pos_std)
    rot_std = np.full(H, 0.05) if rot_std is None else np.asarray(rot_std)
    grip_p = np.zeros(H) if grip_p is None else np.asarray(grip_p)
    mean = rng.uniform(-0.5, 0.5, size=(H, 6))
    norm = np.zeros((n, H, 32), np.float32)
    norm[:, :, 0:3] = (
        mean[None, :, 0:3] + rng.normal(size=(n, H, 3)) * pos_std[None, :, None]
    )
    norm[:, :, 3:6] = (
        mean[None, :, 3:6] + rng.normal(size=(n, H, 3)) * rot_std[None, :, None]
    )
    grip = (rng.uniform(size=(n, H)) < grip_p[None]).astype(np.float32)
    norm[:, :, 6] = grip
    return {
        "action.end_effector_position": norm[:, :, 0:3] * scale,
        "action.end_effector_rotation": norm[:, :, 3:6] * scale * 0.5,
        "action.gripper_close": grip,
        "normalized_action": norm,
    }


class TestEntropyFormulas:
    def test_gaussian_entropy_matches_closed_form(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(500, 3)) * np.array([0.1, 0.2, 0.3])
        cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(3)
        expected = 0.5 * np.log((2 * np.pi * np.e) ** 3 * np.linalg.det(cov))
        assert aac.gaussian_entropy_from_samples(x) == pytest.approx(
            expected, rel=1e-10
        )

    def test_gaussian_entropy_n1_is_zero(self):
        assert aac.gaussian_entropy_from_samples(np.ones((1, 3))) == 0.0

    def test_identical_samples_hit_epsilon_floor(self):
        # cov = 0 -> logdet = 3*log(1e-6); finite, not -inf.
        h = aac.gaussian_entropy_from_samples(np.ones((20, 3)))
        assert np.isfinite(h)
        assert h == pytest.approx(
            0.5 * (3 * np.log(2 * np.pi * np.e) + 3 * np.log(1e-6))
        )

    def test_singular_covariance_is_finite(self):
        s = np.random.default_rng(2).normal(size=(20, 1))
        x = s * np.array([[1.0, 2.0, -1.0]])  # rank 1
        assert np.isfinite(aac.gaussian_entropy_from_samples(x))

    def test_bernoulli(self):
        assert aac.bernoulli_entropy_from_samples(np.ones(20)) == pytest.approx(
            0.0, abs=1e-7
        )
        assert aac.bernoulli_entropy_from_samples(np.zeros(20)) == pytest.approx(
            0.0, abs=1e-7
        )
        assert aac.bernoulli_entropy_from_samples(
            np.r_[np.ones(10), np.zeros(10)]
        ) == pytest.approx(np.log(2))
        # (N, 1) column input is ravelled.
        assert aac.bernoulli_entropy_from_samples(np.ones((20, 1))) == pytest.approx(
            0.0, abs=1e-7
        )

    def test_prefix_mean(self):
        d = _cands(pos_std=np.linspace(0.01, 0.3, H))
        br = aac.action_entropy_gausssian_bernoulli(d)
        for h in range(1, H + 1):
            assert br["chunk_mean"][h - 1] == pytest.approx(np.mean(br["total"][:h]))
        np.testing.assert_allclose(
            br["total"], br["H_pos"] + br["H_rot"] + br["H_grip"]
        )

    def test_entropy_uses_normalized_dims(self):
        # Changing the unnormalized pose must not change entropies (normalized_pose=True upstream).
        d = _cands(pos_std=np.linspace(0.01, 0.3, H))
        br1 = aac.action_entropy_gausssian_bernoulli(d)
        d2 = dict(
            d,
            **{"action.end_effector_position": d["action.end_effector_position"] * 100},
        )
        br2 = aac.action_entropy_gausssian_bernoulli(d2)
        np.testing.assert_array_equal(br1["total"], br2["total"])


class TestSelector:
    def test_identical_candidates(self):
        d = _cands(pos_std=np.zeros(H), rot_std=np.zeros(H), scale=10.0)
        r = aac.aac_select(d)
        # All Ebar_h equal up to float round-off: upstream's argmax is then decided by
        # round-off (documented in AAC_REPRO_NOTES.md). Only the invariants are asserted.
        assert np.allclose(r.chunk_mean_diffs, 0.0, atol=1e-12)
        assert r.h_star == max(r.h_entropy, r.h_min_motion)

    def test_uncertainty_jump_selects_elbow(self):
        # Low std on steps 0..5, high from step 6 -> largest Ebar increase at h=6 -> h_entropy=6.
        std = np.where(np.arange(H) < 6, 0.01, 0.4)
        r = aac.aac_select(_cands(pos_std=std, rot_std=std, scale=100.0), move_th=0.0)
        assert r.raw_argmax == 5 and r.h_entropy == 6
        assert r.h_star == max(r.h_entropy, r.h_min_motion)

    def test_uncertainty_concentrated_early_gives_small_h(self):
        # High uncertainty only on the first two steps: diffs[0] ~ 0, every later diff is
        # strongly negative -> argmax 0 -> h_entropy = max(1, 2) = 2.
        std = np.where(np.arange(H) < 2, 0.4, 0.01)
        r = aac.aac_select(_cands(pos_std=std, rot_std=std, scale=100.0), move_th=0.0)
        assert np.all(r.chunk_mean_diffs[1:] < r.chunk_mean_diffs[0])
        assert r.raw_argmax == 0 and r.h_entropy == 2

    def test_gripper_5050_vs_deterministic(self):
        det = aac.aac_select(_cands(grip_p=np.ones(H)))
        coin = aac.aac_select(_cands(grip_p=np.full(H, 0.5)))
        assert np.all(det.H_grip < 1e-6)
        assert np.all(coin.H_grip > 0.3)

    def test_min_motion_lower_bound(self):
        std = np.where(np.arange(H) < 2, 0.4, 0.01)  # entropy alone -> 2
        tiny = aac.aac_select(_cands(pos_std=std, rot_std=std, scale=1e-3), move_th=3.0)
        assert (
            tiny.h_min_motion == 16 and tiny.h_star == 16
        )  # no motion above threshold -> full chunk
        big = aac.aac_select(_cands(pos_std=std, rot_std=std, scale=1e3), move_th=3.0)
        assert big.h_min_motion == 2 and big.h_star == 2

    def test_horizon_bounds(self):
        rng = np.random.default_rng(3)
        for i in range(100):
            d = _cands(
                n=int(rng.integers(1, 30)),
                pos_std=np.abs(rng.normal(0.1, 0.1, H)),
                rot_std=np.abs(rng.normal(0.1, 0.1, H)),
                grip_p=rng.uniform(size=H),
                scale=float(10 ** rng.uniform(-2, 2)),
                seed=i,
            )
            r = aac.aac_select(d)
            assert 2 <= r.h_star <= H
            assert np.all(np.isfinite(r.H_total))

    def test_n1_reduces_to_motion_rule(self):
        r = aac.aac_select(_cands(n=1, scale=1.0))
        # Gaussian terms are 0 for N<=1; Bernoulli is the clipped floor (~2e-8), constant over t.
        assert np.all(r.H_pos == 0.0) and np.all(r.H_rot == 0.0)
        assert np.allclose(r.chunk_mean_diffs, 0.0, atol=1e-12)
        # argmax over ~0 diffs is round-off-determined (upstream behaviour, reproduced as-is).
        assert r.h_star == max(r.h_entropy, r.h_min_motion)

    def test_gripper_toggle_in_candidate0_counts(self):
        d = _cands(pos_std=np.zeros(H), rot_std=np.zeros(H), scale=0.0)
        d["action.gripper_close"][0, 4:] = 1.0
        m = aac.action_magnitude(d)
        assert m[:3] == [0.0, 0.0, 0.0]  # l = 2, 3, 4: no toggle yet
        assert m[3] == pytest.approx(
            0.2
        )  # l = 5 sees the 0->1 switch (weight 0.2 upstream)

    def test_to_json_roundtrip(self):
        import json

        r = aac.aac_select(_cands())
        js = json.loads(json.dumps(r.to_json()))
        assert (
            js["h_star"] == r.h_star
            and len(js["H_total"]) == H
            and len(js["chunk_mean_diffs"]) == H - 1
        )


class TestBuildUpstreamDict:
    def test_squeezes_trailing_singleton_only(self):
        act = {
            "action.end_effector_position": np.zeros((1, H, 3)),
            "action.gripper_close": np.zeros((1, H, 1)),
            "action.control_mode": np.zeros((1, H, 1)),
        }
        out = aac.build_upstream_pred_action_dict(act, np.zeros((1, H, 32)))
        assert out["action.gripper_close"].shape == (1, H)
        assert out["action.end_effector_position"].shape == (1, H, 3)
        assert out["normalized_action"].shape == (1, H, 32)

    def test_unsqueezed_gripper_would_hide_toggles(self):
        # Documents why the squeeze matters: np.diff on (H,1) runs along the size-1 axis.
        g = np.zeros((H, 1))
        g[5:] = 1
        assert not np.any(np.diff(g))
        assert np.any(np.diff(g[:, 0]))


def _fixture_cases():
    if not FIXTURE.exists():
        return []
    z = np.load(FIXTURE, allow_pickle=False)
    return [str(n) for n in z["__names__"]]


@pytest.mark.skipif(not FIXTURE.exists(), reason="parity fixture not generated")
class TestUpstreamParityFixture:
    @pytest.fixture(scope="class")
    def z(self):
        return np.load(FIXTURE, allow_pickle=False)

    def test_fixture_sha(self, z):
        assert str(z["__upstream_sha__"]) == aac.UPSTREAM_ROBOCASA_SHA

    @pytest.mark.parametrize("name", _fixture_cases())
    def test_case(self, z, name):
        prefix = f"{name.split('@')[0]}|in|"
        d = {k[len(prefix) :]: z[k] for k in z.files if k.startswith(prefix)}
        # Fixture stores only the 7 AAC-relevant normalized dims; re-pad to GR00T's 32.
        norm = d["normalized_action"]
        d["normalized_action"] = np.concatenate(
            [norm, np.zeros(norm.shape[:2] + (32 - norm.shape[2],), norm.dtype)],
            axis=-1,
        )
        mth = float(z[f"{name}|out|move_th"])
        h, br = aac.select_chunk_size(d, method="gaussian_bernoulli", move_th=mth)
        assert h == int(z[f"{name}|out|h_star"])
        for k in ("H_pos", "H_rot", "H_grip", "total", "chunk_mean"):
            np.testing.assert_allclose(
                np.asarray(br[k], np.float64),
                z[f"{name}|out|{k}"],
                rtol=1e-12,
                atol=1e-12,
            )
        np.testing.assert_allclose(
            aac.action_magnitude(d), z[f"{name}|out|chunk_m"], rtol=1e-12, atol=1e-12
        )
        r = aac.aac_select(d, move_th=mth)
        assert r.h_star == h


@pytest.mark.skipif(
    not (UPSTREAM / ".git").exists(), reason="upstream AAC clone not present"
)
def test_live_upstream_parity():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "aac_fixture_gen",
        pathlib.Path(__file__).resolve().parents[2]
        / "experiments"
        / "aac"
        / "make_upstream_parity_fixtures.py",
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    up, sha = gen.load_upstream(UPSTREAM)
    assert sha == aac.UPSTREAM_ROBOCASA_SHA
    rng = np.random.default_rng(123)
    for i in range(50):
        d = _cands(
            n=int(rng.choice([2, 5, 20])),
            pos_std=np.abs(rng.normal(0.1, 0.1, H)),
            rot_std=np.abs(rng.normal(0.1, 0.1, H)),
            grip_p=rng.uniform(size=H),
            scale=float(10 ** rng.uniform(-1, 1)),
            seed=1000 + i,
        )
        h_up, br_up = up.select_chunk_size(
            {k: v.copy() for k, v in d.items()}, "gaussian_bernoulli", 3.0
        )
        h_us, br_us = aac.select_chunk_size(d, "gaussian_bernoulli", 3.0)
        assert h_up == h_us
        np.testing.assert_array_equal(br_up["total"], br_us["total"])
    sys.modules.pop("upstream_action_entropy_v2", None)

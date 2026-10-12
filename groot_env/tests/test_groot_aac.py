"""Stub-based tests for the AAC server wrapper (groot_aac.py) and serve.py flags.

No model or GPU: a stub adapter returns synthetic (N, H, d) candidates with the
same keys/shapes as `GR00TAdapterPolicy.infer_multi_sample`.
"""

from __future__ import annotations

import numpy as np
import pytest

import aac
import groot_aac
import groot_adapter
import serve

H = 16
ACTION_DIMS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}


class StubAdapter:
    """Mimics GR00TAdapterPolicy's multi-sample surface."""

    def __init__(self, spread=0.05):
        self.calls = []
        self.spread = spread

    def infer_multi_sample(self, obs, *, num_samples, seed):
        self.calls.append({"obs": obs, "num_samples": num_samples, "seed": seed})
        rng = np.random.default_rng(0 if seed is None else seed)
        norm = np.zeros((num_samples, H, 32), np.float32)
        norm[:, :, :6] = (
            rng.normal(size=(num_samples, H, 6))
            * self.spread
            * np.linspace(0.2, 2, H)[None, :, None]
        )
        norm[:, :, 6] = (rng.uniform(size=(num_samples, H)) < 0.3).astype(np.float32)
        # Candidate-indexed filler for the dims AAC ignores, so candidate 0 is identifiable.
        acts = {
            k: np.broadcast_to(
                np.arange(num_samples, dtype=np.float32)[:, None, None] * 100,
                (num_samples, H, d),
            ).copy()
            for k, d in ACTION_DIMS.items()
        }
        acts["action.end_effector_position"] = norm[:, :, 0:3] * 5
        acts["action.end_effector_rotation"] = norm[:, :, 3:6]
        acts["action.gripper_close"] = norm[:, :, 6:7]
        return acts, norm

    def action_dict_to_array(self, action_dict):
        return np.concatenate(
            [np.asarray(action_dict[k], np.float32) for k in ACTION_DIMS], axis=-1
        )

    def reset(self):
        pass


class TestAACWrapper:
    def test_metadata(self):
        w = groot_aac.AACGrootPolicyWrapper(StubAdapter())
        md = w.metadata
        assert md["aac_enabled"] is True
        assert md["aac_num_samples"] == 20
        assert md["aac_seed_per_call"] == 42
        assert md["aac_move_th"] == 3.0
        assert md["aac_executed_candidate"] == 0
        assert md["aac_upstream_robocasa_sha"] == aac.UPSTREAM_ROBOCASA_SHA
        assert md["aac_upstream_gr00t_sha"] == aac.UPSTREAM_GR00T_SHA
        assert md["action_horizon"] == 16

    def test_response_schema_and_candidate0(self):
        stub = StubAdapter()
        w = groot_aac.AACGrootPolicyWrapper(stub, num_samples=7, seed=42)
        out = w.infer({"prompt": "x"})
        assert set(out) == {"actions", "aac_horizon", "aac"}
        assert out["actions"].shape == (H, 12) and out["actions"].dtype == np.float32
        acts, _ = stub.infer_multi_sample({}, num_samples=7, seed=42)
        np.testing.assert_array_equal(
            out["actions"], stub.action_dict_to_array(acts)[0]
        )
        assert 2 <= out["aac_horizon"] <= H
        d = out["aac"]
        assert d["h_star"] == out["aac_horizon"]
        assert d["num_samples"] == 7 and d["horizon"] == H
        for k in (
            "H_pos",
            "H_rot",
            "H_grip",
            "H_total",
            "chunk_mean",
            "candidate_spread",
        ):
            assert len(d[k]) == H and all(np.isfinite(d[k]))
        assert len(d["chunk_mean_diffs"]) == H - 1
        assert d["h_star"] == max(d["h_entropy"], d["h_min_motion"])
        assert stub.calls[0]["num_samples"] == 7 and stub.calls[0]["seed"] == 42

    def test_horizon_matches_direct_selector(self):
        stub = StubAdapter()
        w = groot_aac.AACGrootPolicyWrapper(stub, num_samples=20, seed=3, move_th=0.5)
        out = w.infer({})
        acts, norm = stub.infer_multi_sample({}, num_samples=20, seed=3)
        h, _ = aac.select_chunk_size(
            aac.build_upstream_pred_action_dict(acts, norm), "gaussian_bernoulli", 0.5
        )
        assert out["aac_horizon"] == h

    def test_candidates_opt_in(self):
        stub = StubAdapter()
        w = groot_aac.AACGrootPolicyWrapper(stub, num_samples=5)
        out = w.infer({groot_aac.AAC_RETURN_CANDIDATES_KEY: True, "prompt": "x"})
        assert out["aac_candidates"].shape == (5, H, 12)
        assert out["aac_normalized"].shape == (5, H, 32)
        # The magic key is not forwarded to the adapter.
        assert groot_aac.AAC_RETURN_CANDIDATES_KEY not in stub.calls[0]["obs"]
        w2 = groot_aac.AACGrootPolicyWrapper(
            StubAdapter(), num_samples=5, return_candidates=True
        )
        assert "aac_candidates" in w2.infer({})

    def test_seed_none_passthrough(self):
        stub = StubAdapter()
        groot_aac.AACGrootPolicyWrapper(stub, seed=None).infer({})
        assert stub.calls[0]["seed"] is None

    def test_n1(self):
        out = groot_aac.AACGrootPolicyWrapper(StubAdapter(), num_samples=1).infer({})
        assert out["actions"].shape == (H, 12)
        assert 2 <= out["aac_horizon"] <= H

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_bad_num_samples(self, bad):
        with pytest.raises(ValueError):
            groot_aac.AACGrootPolicyWrapper(StubAdapter(), num_samples=bad)

    def test_rejects_unported_method(self):
        with pytest.raises(ValueError):
            groot_aac.AACGrootPolicyWrapper(StubAdapter(), method="variance")


class TestServeFlags:
    def test_defaults_off(self):
        args = serve.Args()
        assert args.aac is False
        assert (
            args.aac_num_samples == 20
            and args.aac_seed == 42
            and args.aac_move_th == 3.0
        )

    @pytest.mark.parametrize(
        "other",
        [{"steer": True, "conceptor_npz": "x.npz"}, {"collect_activations": True}],
    )
    def test_aac_mutually_exclusive(self, other):
        with pytest.raises(ValueError, match="mutually exclusive"):
            serve._validate_args(serve.Args(aac=True, **other))

    def test_aac_rejects_zero_samples(self):
        with pytest.raises(ValueError):
            serve._validate_args(serve.Args(aac=True, aac_num_samples=0))

    def test_aac_ok(self):
        serve._validate_args(serve.Args(aac=True))


class TestAdapterMultiSampleSurface:
    def test_methods_exist(self):
        assert callable(groot_adapter.GR00TAdapterPolicy.infer_multi_sample)
        assert callable(groot_adapter.GR00TAdapterPolicy.action_dict_to_array)

    def test_action_dict_to_array_batched(self):
        pol = groot_adapter.GR00TAdapterPolicy(
            None,
            video_builder=groot_adapter.build_robocasa_videos,
            state_builder=groot_adapter.build_robocasa_state_dict,
            action_keys=list(groot_adapter.ROBOCASA_ACTION_KEYS),
        )
        acts = {k: np.zeros((4, H, d), np.float32) for k, d in ACTION_DIMS.items()}
        assert pol.action_dict_to_array(acts).shape == (4, H, 12)

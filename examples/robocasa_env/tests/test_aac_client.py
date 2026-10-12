"""Tests for main.py's fixed / AAC execution modes, without MuJoCo or a server.

`eval_task` runs against a fake env and a fake policy. Each fake chunk encodes
(call index, position in chunk) in its first two dims, so the test can check
exactly which predicted actions reached `env.step`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import main

H = 16


class FakeEnv:
    def __init__(self, success_at=None):
        self.actions = []
        self.success_at = success_at
        self.reset_seeds = []

    def _obs(self):
        img = np.zeros((8, 8, 3), np.uint8)
        return {
            "video.robot0_agentview_left": img,
            "video.robot0_agentview_right": img,
            "video.robot0_eye_in_hand": img,
            "state.end_effector_position_relative": np.zeros(3, np.float32),
            "state.end_effector_rotation_relative": np.zeros(4, np.float32),
            "state.base_position": np.zeros(3, np.float32),
            "state.base_rotation": np.zeros(4, np.float32),
            "state.gripper_qpos": np.zeros(2, np.float32),
            "annotation.human.task_description": "do the thing",
        }

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return self._obs(), {}

    def step(self, action):
        self.actions.append(np.asarray(action))
        success = self.success_at is not None and len(self.actions) >= self.success_at
        return self._obs(), float(success), False, False, {"success": success}

    def close(self):
        pass


class FakePolicy:
    def __init__(self, aac_horizons=None, chunk_len=H):
        self.calls = 0
        self.aac_horizons = aac_horizons
        self.chunk_len = chunk_len

    def infer(self, element):
        chunk = np.zeros((self.chunk_len, 12), np.float32)
        chunk[:, 0] = self.calls
        chunk[:, 1] = np.arange(self.chunk_len)
        out = {"actions": chunk, "server_timing": {"infer_ms": 1.0}}
        if self.aac_horizons is not None:
            h = self.aac_horizons[self.calls % len(self.aac_horizons)]
            out["aac_horizon"] = h
            out["aac"] = {"h_star": h, "H_total": [0.0] * H}
        self.calls += 1
        return out


@pytest.fixture
def patched(monkeypatch):
    holder = {}

    def make_env(env_name, split, seed):
        holder["env"] = FakeEnv(holder.get("success_at"))
        return holder["env"]

    monkeypatch.setattr(main, "make_env", make_env)
    monkeypatch.setattr(main, "get_task_horizon", lambda name: 40)  # max_steps = 60
    monkeypatch.setattr(main, "convert_action", lambda a: a)
    return holder


def _run(tmp_path, policy, **overrides):
    args = main.Args(save_video=False, log_dir=str(tmp_path / "logs"), **overrides)
    main._validate_args(args)
    return main.eval_task("FakeTask", policy, args, str(tmp_path / "out"))


def _executed(env):
    return [(int(a[0]), int(a[1])) for a in env.actions]


@pytest.mark.parametrize("h", [2, 4, 8, 12, 16])
def test_fixed_executes_exactly_h(tmp_path, patched, h):
    _run(tmp_path, FakePolicy(), execution_mode="fixed", replan_steps=h)
    ex = _executed(patched["env"])
    assert len(ex) == 60
    # Action i comes from call i // h at in-chunk position i % h: never beyond h, never stale.
    assert ex == [(i // h, i % h) for i in range(60)]


def test_aac_executes_server_horizon(tmp_path, patched):
    hs = [3, 16, 2, 7]
    pol = FakePolicy(aac_horizons=hs)
    _run(tmp_path, pol, execution_mode="aac")
    ex = _executed(patched["env"])
    expected = []
    call = 0
    while len(expected) < 60:
        h = hs[call % len(hs)]
        expected += [(call, t) for t in range(h)]
        call += 1
    assert ex == expected[:60]
    # Logs: one inference record per call with the executed horizon.
    recs = [
        json.loads(line)
        for line in open(tmp_path / "logs" / "FakeTask" / "inference.jsonl")
    ]
    assert [r["executed_h"] for r in recs] == [
        hs[i % len(hs)] for i in range(len(recs))
    ]
    assert all(r["aac_horizon"] == r["executed_h"] for r in recs)
    ep = [
        json.loads(line)
        for line in open(tmp_path / "logs" / "FakeTask" / "episodes.jsonl")
    ]
    assert (
        len(ep) == 1
        and ep[0]["num_policy_calls"] == len(recs)
        and ep[0]["env_steps"] == 60
    )
    assert ep[0]["execution_mode"] == "aac" and ep[0]["replan_steps"] is None


def test_fixed_mode_ignores_aac_horizon(tmp_path, patched):
    _run(
        tmp_path, FakePolicy(aac_horizons=[2]), execution_mode="fixed", replan_steps=16
    )
    assert _executed(patched["env"])[:16] == [(0, t) for t in range(16)]


def test_new_episode_starts_with_empty_plan(tmp_path, patched):
    patched["success_at"] = 5  # episode 0 ends mid-chunk at step 5
    pol = FakePolicy()
    _run(tmp_path, pol, execution_mode="fixed", replan_steps=16, num_episodes=1)
    assert pol.calls == 1
    ep = [
        json.loads(line)
        for line in open(tmp_path / "logs" / "FakeTask" / "episodes.jsonl")
    ]
    assert ep[0]["success"] is True and ep[0]["env_steps"] == 5


def test_aac_requires_horizon_in_response(tmp_path, patched):
    with pytest.raises(RuntimeError, match="aac_horizon"):
        _run(tmp_path, FakePolicy(aac_horizons=None), execution_mode="aac")


@pytest.mark.parametrize("bad", [0, 17])
def test_aac_rejects_illegal_horizon(tmp_path, patched, bad):
    with pytest.raises(RuntimeError, match="outside"):
        _run(tmp_path, FakePolicy(aac_horizons=[bad]), execution_mode="aac")


def test_fixed_rejects_horizon_beyond_chunk(tmp_path, patched):
    with pytest.raises(RuntimeError, match="outside"):
        _run(tmp_path, FakePolicy(chunk_len=8), execution_mode="fixed", replan_steps=12)


def test_validate_args():
    with pytest.raises(ValueError):
        main._validate_args(main.Args(execution_mode="adaptive"))
    with pytest.raises(ValueError):
        main._validate_args(main.Args(replan_steps=0))
    with pytest.raises(ValueError):
        main._validate_args(main.Args(execution_mode="aac", steer=True))
    main._validate_args(main.Args(execution_mode="aac"))


def test_seed_per_episode(tmp_path, patched):
    _run(
        tmp_path,
        FakePolicy(),
        num_episodes=3,
        seed=0,
        seed_per_episode=True,
        max_steps=2,
    )
    assert patched["env"].reset_seeds == [0, 1, 2]


def test_default_reset_has_no_seed(tmp_path, patched):
    _run(tmp_path, FakePolicy(), num_episodes=2, max_steps=2)
    assert patched["env"].reset_seeds == [None, None]


def test_eval_all_forwards_aac_flags(tmp_path):
    import eval_all

    cmd = eval_all._build_command(
        eval_all.Args(
            execution_mode="aac",
            seed_per_episode=True,
            save_video=False,
            log_dir=str(tmp_path),
        ),
        "OpenDrawer",
        str(tmp_path),
    )
    assert cmd[cmd.index("--execution_mode") + 1] == "aac"
    assert "--seed_per_episode" in cmd and "--no-save_video" in cmd
    assert cmd[cmd.index("--log_dir") + 1] == str(tmp_path)
    default = eval_all._build_command(eval_all.Args(), "OpenDrawer", str(tmp_path))
    assert "--execution_mode" not in default and "--log_dir" not in default


def test_start_episode_offsets_seeds_and_indices(tmp_path, patched):
    _run(
        tmp_path,
        FakePolicy(),
        num_episodes=2,
        start_episode=3,
        seed=0,
        seed_per_episode=True,
        max_steps=2,
    )
    assert patched["env"].reset_seeds == [3, 4]
    ep = [
        json.loads(line)
        for line in open(tmp_path / "logs" / "FakeTask" / "episodes.jsonl")
    ]
    assert [e["episode"] for e in ep] == [3, 4]


def test_run_id_links_inference_to_episodes(tmp_path, patched):
    _run(tmp_path, FakePolicy(), num_episodes=2, max_steps=4, replan_steps=2)
    ep = [
        json.loads(line)
        for line in open(tmp_path / "logs" / "FakeTask" / "episodes.jsonl")
    ]
    inf = [
        json.loads(line)
        for line in open(tmp_path / "logs" / "FakeTask" / "inference.jsonl")
    ]
    assert len({e["run_id"] for e in ep}) == 1
    assert {(r["episode"], r["run_id"]) for r in inf} == {
        (e["episode"], e["run_id"]) for e in ep
    }

# ruff: noqa: C408  (dict(...) case specs read better than literals here)
"""Generate AAC parity fixtures by running the OFFICIAL upstream selector.

Loads `action_entropy_v2.py` + `help_function.py` from a local clone of
https://github.com/Adaptive-Action-Chunking/robocasa (pinned SHA below), runs
`select_chunk_size` on synthetic candidate sets, and writes inputs + upstream
outputs to `groot_env/tests/fixtures/aac_upstream_parity.npz`.
`groot_env/tests/test_aac.py` then checks the COAST port against those outputs.

Run from groot_env/ (needs numpy/scipy/matplotlib/cv2/torch, all in that venv):

    cd groot_env
    uv run python ../experiments/aac/make_upstream_parity_fixtures.py \
        --upstream ../external/aac/aac-robocasa
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys
import types

import numpy as np

EXPECTED_SHA = "fed3e6b5eb348160dd0570f326f726758fee9056"
H = 16
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "groot_env" / "tests" / "fixtures" / "aac_upstream_parity.npz"


def load_upstream(root: pathlib.Path):
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if sha != EXPECTED_SHA:
        raise SystemExit(f"upstream clone at {sha}, expected {EXPECTED_SHA}")
    demos = root / "robocasa" / "demos"

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Stub the package path upstream imports from, without importing robocasa itself.
    for pkg in ("robocasa", "robocasa.demos"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    _load("robocasa.demos.help_function", demos / "help_function.py")
    return _load("upstream_action_entropy_v2", demos / "action_optimization" / "action_entropy_v2.py"), sha


def make_case(rng, *, n, pos_sched, rot_sched, grip_p, motion_scale, toggle_at=None, singular=False, identical=False):
    """Synthetic candidates. *_sched: per-step std in normalized space (len H)."""
    norm = np.zeros((n, H, 32), dtype=np.float32)
    base_traj = rng.uniform(-0.5, 0.5, size=(H, 7))
    for t in range(H):
        if identical:
            noise_p = np.zeros((n, 3))
            noise_r = np.zeros((n, 3))
        elif singular:
            # All samples on a line -> rank-1 covariance.
            s = rng.normal(size=(n, 1))
            noise_p = s * np.array([[1.0, 2.0, -1.0]]) * pos_sched[t]
            noise_r = s * np.array([[0.5, 0.0, 1.0]]) * rot_sched[t]
        else:
            noise_p = rng.normal(size=(n, 3)) * pos_sched[t]
            noise_r = rng.normal(size=(n, 3)) * rot_sched[t]
        norm[:, t, 0:3] = base_traj[t, 0:3] + noise_p
        norm[:, t, 3:6] = base_traj[t, 3:6] + noise_r
    grip = (rng.uniform(size=(n, H)) < grip_p[None, :]).astype(np.float32)
    if toggle_at is not None:
        grip[0, :toggle_at] = 0.0
        grip[0, toggle_at:] = 1.0
    norm[:, :, 6] = grip
    # Unnormalized deltas: a fixed affine map of the normalized ones (min-max style), scaled.
    pos = (norm[:, :, 0:3] * motion_scale).astype(np.float32)
    rot = (norm[:, :, 3:6] * motion_scale * 0.5).astype(np.float32)
    return {
        "action.end_effector_position": pos,
        "action.end_effector_rotation": rot,
        "action.gripper_close": grip,
        "action.base_motion": np.zeros((n, H, 4), dtype=np.float32),
        "action.control_mode": np.zeros((n, H), dtype=np.float32),
        "normalized_action": norm,
    }


def build_cases(rng):
    flat = np.full(H, 0.05)
    ramp_up = np.linspace(0.01, 0.4, H)
    ramp_down = np.linspace(0.4, 0.01, H)
    jump = np.where(np.arange(H) < 6, 0.02, 0.3)
    zero_p = np.zeros(H)
    half_p = np.full(H, 0.5)
    cases = {
        "identical": dict(n=20, pos_sched=flat, rot_sched=flat, grip_p=zero_p, motion_scale=1.0, identical=True),
        "increasing_late": dict(n=20, pos_sched=ramp_up, rot_sched=ramp_up, grip_p=zero_p, motion_scale=1.0),
        "concentrated_early": dict(n=20, pos_sched=ramp_down, rot_sched=ramp_down, grip_p=zero_p, motion_scale=1.0),
        "jump_at_6": dict(n=20, pos_sched=jump, rot_sched=jump, grip_p=zero_p, motion_scale=1.0),
        "gripper_deterministic": dict(n=20, pos_sched=flat, rot_sched=flat, grip_p=np.ones(H), motion_scale=1.0),
        "gripper_5050": dict(n=20, pos_sched=flat, rot_sched=flat, grip_p=half_p, motion_scale=1.0),
        "singular_cov": dict(
            n=20, pos_sched=ramp_up, rot_sched=ramp_up, grip_p=zero_p, motion_scale=1.0, singular=True
        ),
        "n1": dict(n=1, pos_sched=ramp_up, rot_sched=ramp_up, grip_p=zero_p, motion_scale=1.0),
        "n2": dict(n=2, pos_sched=ramp_up, rot_sched=ramp_up, grip_p=zero_p, motion_scale=1.0),
        "tiny_motion_full_h": dict(n=20, pos_sched=jump, rot_sched=jump, grip_p=zero_p, motion_scale=0.01),
        "huge_motion": dict(n=20, pos_sched=jump, rot_sched=jump, grip_p=zero_p, motion_scale=20.0),
        "gripper_toggle_c0": dict(n=20, pos_sched=flat, rot_sched=flat, grip_p=zero_p, motion_scale=0.2, toggle_at=5),
    }
    for i in range(60):
        n = int(rng.choice([2, 3, 5, 10, 20, 40]))
        sched = np.abs(rng.normal(0.1, 0.1, size=H)) + 1e-4
        cases[f"random_{i:03d}"] = dict(
            n=n,
            pos_sched=sched,
            rot_sched=np.abs(rng.normal(0.1, 0.1, size=H)) + 1e-4,
            grip_p=rng.uniform(0, 1, size=H) * (rng.uniform() < 0.5),
            motion_scale=float(10 ** rng.uniform(-1.5, 1.0)),
            toggle_at=int(rng.integers(1, H)) if rng.uniform() < 0.3 else None,
        )
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=pathlib.Path, default=REPO_ROOT / "external" / "aac" / "aac-robocasa")
    ap.add_argument("--move_th", type=float, nargs="+", default=[3.0, 0.5])
    args = ap.parse_args()
    up, sha = load_upstream(args.upstream)
    rng = np.random.default_rng(0)
    arrays: dict[str, np.ndarray] = {}
    names = []
    for name, spec in build_cases(rng).items():
        d = make_case(rng, **spec)
        # Inputs stored once per case: only what AAC reads (normalized dims 0:7; test re-pads to 32).
        for k in ("action.end_effector_position", "action.end_effector_rotation", "action.gripper_close"):
            arrays[f"{name}|in|{k}"] = d[k]
        arrays[f"{name}|in|normalized_action"] = d["normalized_action"][:, :, :7]
        for mth in args.move_th:
            key = f"{name}@{mth}"
            h, br = up.select_chunk_size({k: v.copy() for k, v in d.items()}, method="gaussian_bernoulli", move_th=mth)
            chunk_m = up.action_magnitude({k: v.copy() for k, v in d.items()})
            names.append(key)
            arrays[f"{key}|out|h_star"] = np.array(h)
            arrays[f"{key}|out|move_th"] = np.array(mth)
            arrays[f"{key}|out|chunk_m"] = np.asarray(chunk_m, dtype=np.float64)
            for k in ("H_pos", "H_rot", "H_grip", "total", "chunk_mean"):
                arrays[f"{key}|out|{k}"] = np.asarray(br[k], dtype=np.float64)
    arrays["__names__"] = np.array(names)
    arrays["__upstream_sha__"] = np.array(sha)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    hs = [int(arrays[f"{n}|out|h_star"]) for n in names]
    print(f"wrote {len(names)} cases to {OUT}; h_star histogram:", np.bincount(hs, minlength=17)[2:])


if __name__ == "__main__":
    main()

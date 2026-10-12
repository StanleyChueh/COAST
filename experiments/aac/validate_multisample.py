# ruff: noqa: SLF001  (validation deliberately exercises adapter internals)
"""GPU validation of multi-sample GR00T inference + baseline identity (groot_env venv).

    cd groot_env
    CUDA_VISIBLE_DEVICES=0 uv run python ../experiments/aac/validate_multisample.py \
        --out ../results/aac/validation [--obs_npz ../results/aac/validation/real_obs.npz]

Checks (results -> <out>/multisample_validation.json):
  A. baseline identity: ORIGINAL adapter (groot_adapter.py from git main) `infer` vs
     current adapter `infer`, same torch seed -> must be bit-identical.
  B. NVIDIA `Gr00tPolicy.get_action` vs `_get_action_multi_sample(N=1, seed=None)`
     under the same torch seed -> bit-identical expected.
  C. shared conditioning: with the initial noise forced identical across candidates,
     all N outputs must be identical (only noise distinguishes candidates).
  D. diversity: candidates differ; reseeding with the same seed reproduces them;
     a different seed changes them.
  E. N=20 candidate 0 vs N=1 (same seed): reported, not required to match.
  F. latency / peak VRAM vs N (sampling and AAC selection separately).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "groot_env"))

import aac  # noqa: E402
import groot_adapter  # noqa: E402

CKPT = REPO / "checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000"


def synthetic_obs(seed=0):
    rng = np.random.default_rng(seed)
    state = np.concatenate(
        [rng.uniform(-0.3, 0.3, 3), [0, 0, 0, 1], rng.uniform(-0.1, 0.1, 3), [0, 0, 0, 1], [0.04, -0.04]]
    ).astype(np.float32)
    return {
        "observation/image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation/image2": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation/state": state,
        "prompt": "open the drawer",
    }


def load_obs(path):
    z = np.load(path, allow_pickle=False)
    obs = {k: z[k] for k in z.files if k != "prompt"}
    obs["prompt"] = str(z["prompt"])
    return obs


def load_original_adapter():
    src = subprocess.check_output(["git", "show", "main:groot_env/groot_adapter.py"], cwd=REPO, text=True)
    tmp = pathlib.Path(tempfile.mkdtemp()) / "groot_adapter_main.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("groot_adapter_main", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results/aac/validation")
    ap.add_argument("--obs_npz", type=pathlib.Path)
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 5, 10, 20, 30, 40])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    obs = load_obs(args.obs_npz) if args.obs_npz else synthetic_obs()
    report = {
        "obs": str(args.obs_npz) if args.obs_npz else "synthetic(seed=0)",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    t = time.perf_counter()
    policy = groot_adapter.make_robocasa_policy(str(CKPT), device="cuda:0", denoising_steps=4)
    report["load_s"] = time.perf_counter() - t
    gp = policy._policy

    # A. original vs current adapter, plain (non-AAC) inference.
    orig = load_original_adapter()
    orig_policy = orig.GR00TAdapterPolicy(
        gp,
        video_builder=orig.build_robocasa_videos,
        state_builder=orig.build_robocasa_state_dict,
        action_keys=list(orig.ROBOCASA_ACTION_KEYS),
    )
    torch.manual_seed(1234)
    a_old = orig_policy.infer(dict(obs))["actions"]
    torch.manual_seed(1234)
    a_new = policy.infer(dict(obs))["actions"]
    report["A_baseline_identity"] = {
        "shape": list(a_new.shape),
        "dtype": str(a_new.dtype),
        "bit_identical": bool(np.array_equal(a_old, a_new)),
        "max_abs_diff": max_abs(a_old, a_new),
    }
    # Same inference twice under the same seed (GPU determinism of the plain path).
    torch.manual_seed(1234)
    a_new2 = policy.infer(dict(obs))["actions"]
    report["A_plain_repeatability"] = {
        "bit_identical": bool(np.array_equal(a_new, a_new2)),
        "max_abs_diff": max_abs(a_new, a_new2),
    }

    # B. NVIDIA get_action vs multi-sample N=1 (no reseed), same torch seed.
    groot_obs = policy._build_groot_obs(dict(obs))
    torch.manual_seed(77)
    ref = gp.get_action(dict(groot_obs))
    torch.manual_seed(77)
    ms, ms_norm = groot_adapter._get_action_multi_sample(gp, dict(groot_obs), num_samples=1, seed=None)
    ref_arr = policy.action_dict_to_array(ref)
    ms_arr = policy.action_dict_to_array({k: v[0] for k, v in ms.items()})
    report["B_n1_vs_get_action"] = {
        "bit_identical": bool(np.array_equal(ref_arr, ms_arr)),
        "max_abs_diff": max_abs(ref_arr, ms_arr),
        "multi_shapes": {k: list(v.shape) for k, v in ms.items()},
        "normalized_shape": list(ms_norm.shape),
        "normalized_dtype": str(ms_norm.dtype),
    }

    # C. shared conditioning: force identical initial noise for all candidates.
    real_randn = torch.randn

    def same_noise(*a, size=None, **kw):
        size = size if size is not None else a[0]
        one = real_randn((1, *tuple(size[1:])), **kw)
        return one.expand(tuple(size)).contiguous()

    torch.randn = same_noise
    try:
        c_act, c_norm = groot_adapter._get_action_multi_sample(gp, dict(groot_obs), num_samples=8, seed=5)
    finally:
        torch.randn = real_randn
    spread_c = float(np.max(np.abs(c_norm - c_norm[:1])))
    report["C_shared_conditioning"] = {
        "max_abs_dev_from_candidate0_with_identical_noise": spread_c,
        "identical": spread_c == 0.0,
    }

    # D. diversity + seed determinism.
    d1, n1 = groot_adapter._get_action_multi_sample(gp, dict(groot_obs), num_samples=20, seed=42)
    d2, n2 = groot_adapter._get_action_multi_sample(gp, dict(groot_obs), num_samples=20, seed=42)
    d3, n3 = groot_adapter._get_action_multi_sample(gp, dict(groot_obs), num_samples=20, seed=43)
    flat = n1[:, :, :7].reshape(20, -1)
    pd = np.linalg.norm(flat[:, None] - flat[None], axis=-1)[np.triu_indices(20, 1)]
    report["D_diversity"] = {
        "min_pairwise_l2_normalized_dims0_6": float(pd.min()),
        "mean_pairwise_l2_normalized_dims0_6": float(pd.mean()),
        "num_identical_pairs": int((pd == 0).sum()),
        "per_step_std_dims0_6": n1[:, :, :7].std(0).mean(-1).tolist(),
        "same_seed_bit_identical": bool(np.array_equal(n1, n2)),
        "same_seed_max_abs_diff": max_abs(n1, n2),
        "different_seed_max_abs_diff": max_abs(n1, n3),
    }
    sel = aac.aac_select(aac.build_upstream_pred_action_dict(d1, n1))
    report["D_aac_on_obs"] = sel.to_json()

    # E. candidate 0 at N=20 vs N=1, both reseeded with 42.
    e1, en1 = groot_adapter._get_action_multi_sample(gp, dict(groot_obs), num_samples=1, seed=42)
    report["E_cand0_N20_vs_N1"] = {
        "max_abs_diff_normalized": max_abs(n1[0], en1[0]),
        "identical": bool(np.array_equal(n1[0], en1[0])),
    }

    # F. latency / VRAM vs N.
    lat = {}
    for n in args.ns:
        for _ in range(3):
            groot_adapter._get_action_multi_sample(gp, dict(groot_obs), num_samples=n, seed=42)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts, tsel = [], []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            act, norm = policy.infer_multi_sample(dict(obs), num_samples=n, seed=42)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            aac.aac_select(aac.build_upstream_pred_action_dict(act, norm))
            t2 = time.perf_counter()
            ts.append((t1 - t0) * 1000)
            tsel.append((t2 - t1) * 1000)
        lat[n] = {
            "sample_ms_mean": float(np.mean(ts)),
            "sample_ms_p50": float(np.median(ts)),
            "sample_ms_std": float(np.std(ts)),
            "select_ms_mean": float(np.mean(tsel)),
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
        print(n, lat[n], flush=True)
    # Plain single-sample path for reference.
    ts = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.infer(dict(obs))
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    lat["plain_infer"] = {
        "sample_ms_mean": float(np.mean(ts)),
        "sample_ms_p50": float(np.median(ts)),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    report["F_latency_vs_N"] = lat
    out = args.out / "multisample_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "D_aac_on_obs"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

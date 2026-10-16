"""GPU identity checks for the intervention server's operators (in-process, no websocket).

Checks (results to --out JSON):
  noise_golden        derive_noise((100,2,15,0)) fingerprint equals the Phase-1D pinned value
  base_vs_stock       I0 (steering-path sampler, no hooks) vs stock Policy.infer(obs, noise=...)
  base_repeat         I0 twice with the same key -> bitwise equal
  capture_inert       I0 with the read-only feature CaptureHook -> bitwise equal to I0
  il_beta0_identity   COAST hook with β=0 (M = I) -> equal to I0
  ic_omega1           CAG with ω=1 -> equal to I0 up to float rounding (a_u + (a_c - a_u))
  ops_differ         IL and IC actually change the chunk (L2 distance from I0 > 0)
  features_repeat     the decision-state feature bundle is deterministic
  latency             per-op wall time

    CUDA_VISIBLE_DEVICES=1 uv run python research/adaptive_inference/validate_operators.py \
        --obs results/adaptive_inference/validation/obs_samples.npz \
        --out results/adaptive_inference/validation/operators.json
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time

import numpy as np
import torch
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import intervention_server as srv

from openpi.serving.steering import ConceptorSteeringHook
from openpi.serving.steering import get_conceptor_matrix

GOLDEN_KEY = (100, 2, 15, 0)
GOLDEN_SHA = "87ae314ae03d4e549d3bfa442c28cb94f594bff03d5c196422ae8b165ab113a0"


@dataclasses.dataclass
class Args:
    obs: str = "results/adaptive_inference/validation/obs_samples.npz"
    out: str = "results/adaptive_inference/validation/operators.json"
    conceptor_npz: str = "conceptors/aiv_pilot_conceptors.npz"
    checkpoint: str = "checkpoints/openpi-libero-2000"


def load_obs(path: str) -> dict[int, tuple[dict, str]]:
    z = np.load(path)
    out = {}
    for tid in sorted({int(k.split("|")[0][1:]) for k in z.files}):
        p = f"t{tid}|"
        obs = {k[len(p) :]: z[k] for k in z.files if k.startswith(p) and k[len(p) :] not in ("prompt", "name")}
        obs["prompt"] = str(z[p + "prompt"])
        out[tid] = (obs, str(z[p + "name"]))
    return out


def maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main(args: Args) -> None:
    res: dict = {"checks": {}, "per_task": {}}
    noise = srv.derive_noise(list(GOLDEN_KEY), (10, 32))
    res["checks"]["noise_golden"] = srv.noise_sha256(noise) == GOLDEN_SHA

    policy = srv.build_policy(srv.Args(checkpoint=args.checkpoint, conceptor_npz=args.conceptor_npz))
    ip = srv.InterventionPolicy(policy, conceptor_npz=args.conceptor_npz)
    res["metadata"] = {
        k: (v if isinstance(v, (int, float, str, list, dict, type(None))) else str(v)) for k, v in ip.metadata.items()
    }
    shape = ip._shape
    for tid, (obs, task) in load_obs(args.obs).items():
        r: dict = {"task": task}
        key = [7, tid, 15, 0]
        nz = srv.derive_noise(key, shape)
        t0 = time.monotonic()
        a0 = ip.run_op("I0", obs, nz, task)
        r["ms_I0"] = (time.monotonic() - t0) * 1000
        a0b = ip.run_op("I0", obs, nz, task)
        stock = policy.infer(dict(obs), noise=nz)["actions"]
        r["base_vs_stock_maxdiff"] = maxdiff(a0["actions"], stock)
        r["base_vs_stock_bitwise"] = bool(np.array_equal(a0["actions"], stock))
        r["base_repeat_bitwise"] = bool(np.array_equal(a0["actions"], a0b["actions"]))
        cap = srv.CaptureHook(steps=(0, 9))
        a0c = ip.run_op("I0", obs, nz, task, extra_hooks=[(11, cap)])
        r["capture_inert_bitwise"] = bool(np.array_equal(a0["actions"], a0c["actions"]))
        r["capture_recorded_steps"] = sorted(cap.records)
        # IL with beta = 0
        cfg = srv.IL_CONFIGS[task]
        c = get_conceptor_matrix(ip._npz, task=task, layer=cfg["layer"], alpha=cfg["alpha"], strategy="global")
        h0 = ConceptorSteeringHook(c, beta=0.0, device=ip._device)
        observation, state = ip._transform(obs, mask_language=False)
        a_b0 = ip._out(state, ip._sample(observation, nz, [(cfg["layer"], h0)]))
        r["il_beta0_maxdiff"] = maxdiff(a_b0, a0["actions"])
        r["il_beta0_bitwise"] = bool(np.array_equal(a_b0, a0["actions"]))
        t0 = time.monotonic()
        al = ip.run_op("IL", obs, nz, task)
        r["ms_IL"] = (time.monotonic() - t0) * 1000
        t0 = time.monotonic()
        ac = ip.run_op("IC", obs, nz, task)
        r["ms_IC"] = (time.monotonic() - t0) * 1000
        # CAG with omega = 1
        a_w1 = ip._out(state, ac["cag_uncond_model"] + 1.0 * (ac["cag_cond_model"] - ac["cag_uncond_model"]))
        r["ic_omega1_maxdiff"] = maxdiff(a_w1, a0["actions"])
        r["cag_cond_equals_I0_bitwise"] = bool(np.array_equal(ac["cag_cond_model"], a0["actions_model"]))
        r["IL_minus_I0_l2"] = float(np.linalg.norm(al["actions"] - a0["actions"]))
        r["IC_minus_I0_l2"] = float(np.linalg.norm(ac["actions"] - a0["actions"]))
        r["IL_step0_minus_I0"] = float(np.linalg.norm(al["actions"][0] - a0["actions"][0]))
        t0 = time.monotonic()
        f1 = ip.features(obs, key, task)
        r["ms_features"] = (time.monotonic() - t0) * 1000
        f2 = ip.features(obs, key, task)
        r["features_repeat_bitwise"] = all(
            (f1[k] is None and f2[k] is None) or np.array_equal(np.asarray(f1[k]), np.asarray(f2[k])) for k in f1
        )
        r["feature_chunk_I0_equals_op"] = bool(np.array_equal(f1["chunk_I0"], a0["actions"]))
        r["feature_summary"] = {
            "cag_contrast_model_norm": f1["cag_contrast_model_norm"],
            "il_displacement_model_norm": f1["il_displacement_model_norm"],
            "base_sample_spread": float(np.mean(np.std(f1["base_samples"], axis=0))),
            "latent_dim": None if f1["latent_step0"] is None else int(f1["latent_step0"].shape[0]),
        }
        res["per_task"][tid] = r
        print(tid, json.dumps(r, default=str))
    pt = res["per_task"].values()
    res["checks"].update(
        {
            "base_vs_stock_bitwise_all": all(r["base_vs_stock_bitwise"] for r in pt),
            "base_vs_stock_maxdiff_max": max(r["base_vs_stock_maxdiff"] for r in pt),
            "base_repeat_bitwise_all": all(r["base_repeat_bitwise"] for r in pt),
            "capture_inert_all": all(r["capture_inert_bitwise"] for r in pt),
            "il_beta0_bitwise_all": all(r["il_beta0_bitwise"] for r in pt),
            "il_beta0_maxdiff_max": max(r["il_beta0_maxdiff"] for r in pt),
            "ic_omega1_maxdiff_max": max(r["ic_omega1_maxdiff"] for r in pt),
            "cag_cond_equals_I0_all": all(r["cag_cond_equals_I0_bitwise"] for r in pt),
            "ops_differ_all": all(r["IL_minus_I0_l2"] > 0 and r["IC_minus_I0_l2"] > 0 for r in pt),
            "features_repeat_all": all(r["features_repeat_bitwise"] for r in pt),
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
    )
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["checks"], indent=1))


if __name__ == "__main__":
    main(tyro.cli(Args))

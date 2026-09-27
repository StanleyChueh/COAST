"""Phase 7B pre-rollout GPU checks (synthetic observation only; no LIBERO rollout, no outcome).

For each task's paper Global configuration, through the exact Phase 7B serving stack
(PairedNoiseWrapper -> candidate SteeredPolicyWrapper -> candidate Policy):
  1. Same noise key -> the tensor reaching the sampler has the same SHA-256 under all four conditions,
     equal to the Phase 1D derived noise; a different master seed / rollout_step gives different noise.
  2. Determinism: the same key twice under the same condition gives bit-identical actions.
  3. Baseline path equivalence: candidate `Policy.infer(noise=)` (BASELINE path) vs the steering sampler with
     an identity hook (beta = 0, M = I) under the same noise -> max |diff| of actions.
  4. PURE_SHRINKAGE hook verification (fp32 exact, bf16 bit-identical to bf16 scalar multiply).
  5. Sanity: action differences between conditions under identical noise (nonzero for steered conditions).

Usage (root venv, same env as the driver):
    python phase7b_gpu_checks.py <out_json>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase7b_paired_driver as drv  # noqa: E402

cs = drv.cs


def synthetic_obs(prompt: str) -> dict:
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": prompt,
    }


def main() -> None:
    import os

    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    out_path = pathlib.Path(sys.argv[1])
    ckpt = str(drv.MAIN_ROOT / "checkpoints/openpi-libero-2000")
    ensure_pytorch_checkpoint(ckpt, "pi05_libero")
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), ckpt)
    device = str(policy._pytorch_device)  # noqa: SLF001
    odh_root = pathlib.Path(os.environ["PHASE7B_ODH_ROOT"])
    report = {"golden": drv.NOISE.noise_fingerprint(drv.NOISE.derive_noise(drv.GOLDEN_KEY, 10, 32)) == drv.GOLDEN_SHA}
    log = pathlib.Path(out_path.parent / "gpu_check_noise_log.jsonl")
    log.unlink(missing_ok=True)

    for key, (task, layer, alpha, beta) in drv.TASKS.items():
        cs.CONCEPTOR_NPZ = odh_root / key / "libero_conceptors.npz"
        npz = cs.load_npz()
        hooks, _ = drv.build_hooks(key, npz, device)
        steered = cs.SteeredPolicyWrapper(policy, steering_hooks=None)
        wrapper = drv.PairedNoiseWrapper(steered, policy, log)
        prompt = task.split("_", 2)[-1].replace("_", " ")
        obs = synthetic_obs(prompt)
        task_id = cs.LIBERO_TASK_IDS[task]
        k = {"master_seed": 100, "task_id": task_id, "init_state": 15, "rollout_step": 0}
        res = {"layer": layer, "alpha": alpha, "beta": beta}
        actions, shas = {}, {}
        for cond in drv.CONDITIONS:
            wrapper.set_condition(cond, log)
            steered.update_hooks(hooks[cond])
            r1 = wrapper.infer({**obs, "__noise_control__": dict(k)})
            r2 = wrapper.infer({**obs, "__noise_control__": dict(k)})
            actions[cond] = np.asarray(r1["actions"])
            shas[cond] = r1["noise_control"]["sampler_sha256"]
            res[f"{cond}_repeat_max_abs_diff"] = float(np.abs(actions[cond] - np.asarray(r2["actions"])).max())
            res[f"{cond}_sampler"] = r1["noise_control"]["sampler"]
        res["sampler_sha_identical_all_conditions"] = len(set(shas.values())) == 1
        res["sampler_sha"] = next(iter(shas.values()))
        # different keys -> different noise
        wrapper.set_condition("BASELINE", log)
        steered.update_hooks(None)
        other = {}
        for name, kk in (("seed200", {**k, "master_seed": 200}), ("step5", {**k, "rollout_step": 5})):
            other[name] = wrapper.infer({**obs, "__noise_control__": kk})["noise_control"]["sha256"]
        res["different_keys_give_different_noise"] = all(v != res["sampler_sha"] for v in other.values())
        # identity hook through the steering sampler vs baseline path
        identity = [(layer, cs.ConceptorSteeringHook(np.zeros((1024, 1024), np.float32), beta=0.0, device=device))]
        wrapper.set_condition("IDENTITY", log)
        steered.update_hooks(identity)
        a_id = np.asarray(wrapper.infer({**obs, "__noise_control__": dict(k)})["actions"])
        res["baseline_path_vs_identity_hook_max_abs_diff"] = float(np.abs(a_id - actions["BASELINE"]).max())
        for a, b in (
            ("ORIGINAL_COAST", "BASELINE"),
            ("PURE_SHRINKAGE", "BASELINE"),
            ("RANDOM_CONTROL", "BASELINE"),
            ("ORIGINAL_COAST", "PURE_SHRINKAGE"),
            ("ORIGINAL_COAST", "RANDOM_CONTROL"),
        ):
            res[f"action_max_abs_diff_{a}_vs_{b}"] = float(np.abs(actions[a] - actions[b]).max())
        res["shrinkage_verification"] = drv.verify_shrinkage_hook(hooks, beta, device)
        report[key] = res
        print(key, json.dumps(res, indent=1))
    out_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

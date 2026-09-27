"""Phase 7B (exploratory, after the primary records are frozen): action-level comparison at identical observations.

For each task, held-out state s in 15..44 and master seed in {100, 200}: the saved rollout_step-0 observation
(phase7b_probe_observations.py) and the key (seed, task_id, s, 0) -- exactly the first request of that episode in
the primary experiment -- are sent through the Phase 7B serving stack (PairedNoiseWrapper -> candidate
SteeredPolicyWrapper -> candidate Policy) under all four conditions. Recorded:
  - the action chunk of each condition (output-transformed, what the env would receive);
  - the BASELINE hidden state at the hooked action-expert layer, all 10 denoising steps (a read-only forward
    hook registered on the same layer; it returns None, so the output is not modified).

Only actions at the same observation and the same noise are compared.

Usage (root venv, same env as the driver; PHASE7B_ODH_ROOT=<scratch>/odh):
    python phase7b_action_probe.py <obs_dir> <out_dir>
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase7b_paired_driver as drv

cs = drv.cs
SEEDS = (100, 200)


def main() -> None:
    import torch

    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    obs_dir, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(drv.MAIN_ROOT / "checkpoints/openpi-libero-2000")
    ensure_pytorch_checkpoint(ckpt, "pi05_libero")
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), ckpt)
    device = str(policy._pytorch_device)  # noqa: SLF001
    odh_root = pathlib.Path(os.environ["PHASE7B_ODH_ROOT"])
    layers = policy._model.paligemma_with_expert.gemma_expert.model.layers  # noqa: SLF001

    for key, (task, layer, _alpha, _beta) in drv.TASKS.items():
        cs.CONCEPTOR_NPZ = odh_root / key / "libero_conceptors.npz"
        hooks, _ = drv.build_hooks(key, cs.load_npz(), device)
        steered = cs.SteeredPolicyWrapper(policy, steering_hooks=None)
        log = out_dir / f"{key}_probe_noise_log.jsonl"
        log.unlink(missing_ok=True)
        wrapper = drv.PairedNoiseWrapper(steered, policy, log)
        z = np.load(obs_dir / f"{key}_obs.npz")
        # KS4 re-randomizes a fixed fixture's pose on every env.reset() (not part of the MuJoCo state). Its pose
        # depends only on the reset count, not on intervening steps (checked), so the extra check-reset differs
        # while the saved sequence (one reset per state, fresh env, seed 7) matches the rollouts.
        assert bool(z["order_invariant"]) or key == "KS4", key
        task_id = cs.LIBERO_TASK_IDS[task]
        n = len(z["init_states"])
        actions = np.zeros((len(SEEDS), n, len(drv.CONDITIONS), 10, 7), np.float64)
        hidden = np.zeros((len(SEEDS), n, 10, 10, 1024), np.float32)  # seed, state, denoise step, token, d
        captured: list = []

        def capture(_module, _inp, out, sink=captured):
            h = out[0] if isinstance(out, tuple) else out
            sink.append(h.detach().float().cpu().numpy()[0])

        for si, seed in enumerate(SEEDS):
            for k in range(n):
                obs = {
                    "observation/image": z["image"][k],
                    "observation/wrist_image": z["wrist_image"][k],
                    "observation/state": z["state"][k],
                    "prompt": str(z["prompt"]),
                }
                nk = {
                    "master_seed": seed,
                    "task_id": task_id,
                    "init_state": int(z["init_states"][k]),
                    "rollout_step": 0,
                }
                for ci, cond in enumerate(drv.CONDITIONS):
                    wrapper.set_condition(cond, log)
                    steered.update_hooks(hooks[cond])
                    handle = layers[layer].register_forward_hook(capture) if cond == "BASELINE" else None
                    captured.clear()
                    try:
                        with torch.no_grad():
                            res = wrapper.infer({**obs, "__noise_control__": dict(nk)})
                    finally:
                        if handle is not None:
                            handle.remove()
                    actions[si, k, ci] = np.asarray(res["actions"], dtype=np.float64)
                    if cond == "BASELINE":
                        assert len(captured) == 10, len(captured)
                        hidden[si, k] = np.stack(captured)
        np.savez_compressed(
            out_dir / f"{key}_probe.npz",
            actions=actions,
            hidden=hidden,
            init_states=z["init_states"],
            seeds=np.array(SEEDS),
            conditions=np.array(drv.CONDITIONS),
            layer=layer,
        )
        print(key, "done", actions.shape, hidden.shape)


if __name__ == "__main__":
    main()

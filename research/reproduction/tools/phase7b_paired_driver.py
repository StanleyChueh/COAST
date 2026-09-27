"""Phase 7B: paired-noise mechanism decomposition driver (server side, root venv).

Evaluation-only helper. It runs, for ONE task and ONE master noise seed, the four pre-registered conditions
against a single in-process policy server, in this fixed order:

    BASELINE        no hook                                   h' = h
    ORIGINAL_COAST  candidate hook, C = NPZ C_contrastive     h' = h @ ((1-b)I + bC).T
    PURE_SHRINKAGE  candidate hook, C = 0 (d x d zeros)       h' = h @ ((1-b)I).T = (1-b) h
    RANDOM_CONTROL  candidate hook, C = candidate random      h' = h @ ((1-b)I + bC_rand).T,
                    compute_random_conceptor(seed = 100*layer + int(10*b))

Everything model-side comes from the **unmodified** historical candidate driver
`experiments/pi05_libero/src/conceptor_steering.py` @ 29059a5 (miranda-v2), imported as a module:
policy loading, `load_npz`, `get_global_contrastive`, `ConceptorSteeringHook` (the only hidden-state
transformation used by all three steered conditions), `compute_random_conceptor`, `SteeredPolicyWrapper`
(routes hooks=None to `Policy.infer` and hooks to `Policy.infer_with_steering`), and the websocket server.
PURE_SHRINKAGE is the candidate hook built from a zero matrix, so it acts at exactly the same point as COAST
(same layer, same action-expert forward hook, same suffix/action tokens, all 10 denoising steps, same
`M.to(h.dtype)` cast and `h @ M.T` matmul).

Paired policy noise (Phase 1D scheme, reused verbatim): the client attaches
obs["__noise_control__"] = {master_seed, task_id, init_state, rollout_step}. `PairedNoiseWrapper` (outermost)
pops it, derives the noise with Phase 1D's `derive_noise` (numpy SeedSequence -> PCG64 -> standard_normal
(10, 32) float32; loaded by file path from the main repo), and injects it at the sampler: for the duration of
the call it shadows `policy._sample_actions` (baseline path) and `policy._model.sample_actions_with_steering`
(steered path) with shims that pass `noise=` exactly as `Policy.infer` converts it
(`torch.from_numpy(noise).to(device)[None]`). Each shim fingerprints the tensor it actually forwards to the
sampler. Exactly one sampler call per request is asserted. The response echoes the key, the derived-noise
SHA-256 and the sampler-side SHA-256; the server also appends one JSON line per request to
<output_dir>/server_noise_log.jsonl. A request without a noise key is an error (no silent fallback).

Usage (root venv; PYTHONPATH=<candidate>/src:<candidate>/packages/openpi-client/src;
       OPENPI_DATA_HOME=<dir containing the task's libero_conceptors.npz>; TORCH_COMPILE_DISABLE=1):
    python phase7b_paired_driver.py --task-key KS4 --master-seed 100 --repeat 1 --port 8810 \
        --output-dir <dir> [--state-offset 15 --num-episodes 30] [--conditions ...] [--max-steps N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

import numpy as np

CANDIDATE_ROOT = pathlib.Path(
    os.environ.get("PHASE7A_CANDIDATE_ROOT", "/home/stanley/Stanley_ws/COAST-research/COAST-paper-candidate")
)
MAIN_ROOT = pathlib.Path(__file__).resolve().parents[3]
CLIENT_WRAPPER = pathlib.Path(__file__).resolve().parent / "phase7b_paired_client.py"

sys.path.insert(0, str(CANDIDATE_ROOT / "experiments" / "pi05_libero" / "src"))
import conceptor_steering as cs  # noqa: E402  (unmodified candidate driver)

logger = logging.getLogger("phase7b_driver")

TASKS = {  # key -> (full name, layer, alpha, beta); paper Global configurations (Table 4 = Table 17)
    "LR2a": ("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", 11, 0.5, 0.3),
    "KS4": ("KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", 5, 1.0, 0.1),
    "LR1": ("LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", 11, 0.5, 0.3),
}
CONDITIONS = ("BASELINE", "ORIGINAL_COAST", "PURE_SHRINKAGE", "RANDOM_CONTROL")
GOLDEN_KEY = {"master_seed": 100, "task_id": 2, "init_state": 15, "rollout_step": 0}
GOLDEN_SHA = "87ae314ae03d4e549d3bfa442c28cb94f594bff03d5c196422ae8b165ab113a0"  # Phase 1D, pinned


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_phase1d_noise_modules():
    """Load the Phase 1D wire protocol + server noise code from the main repo, unchanged.

    The candidate's `openpi_client` package has no `noise_control` submodule, so the main repo's stdlib-only
    wire module is registered under its canonical name before the server module (which imports it) is loaded.
    """
    wire = _load_by_path(
        "openpi_client.noise_control", MAIN_ROOT / "packages/openpi-client/src/openpi_client/noise_control.py"
    )
    server = _load_by_path("phase1d_noise_control", MAIN_ROOT / "src/openpi/serving/noise_control.py")
    return wire, server


NOISE_WIRE, NOISE = load_phase1d_noise_modules()


def tensor_sha256(t) -> str:
    arr = t.detach().to("cpu").float().numpy()
    return hashlib.sha256(np.ascontiguousarray(arr, dtype="<f4").tobytes()).hexdigest()


class PairedNoiseWrapper:
    """Outermost wrapper: derive paired noise from the request key and inject it at the sampler."""

    def __init__(self, inner, policy, log_path: pathlib.Path):
        self._inner = inner  # cs.SteeredPolicyWrapper (unmodified)
        self._policy = policy
        self._shape = NOISE.noise_shape_from_policy(policy)
        self._log_path = log_path
        self._lock = threading.Lock()
        self.condition = None

    def set_condition(self, condition: str, log_path: pathlib.Path) -> None:
        self.condition = condition
        self._log_path = log_path

    def infer(self, obs: dict) -> dict:
        if NOISE_WIRE.NOISE_CONTROL_KEY not in obs:
            raise ValueError("Phase 7B requires a noise-control key on every request")
        payload = obs[NOISE_WIRE.NOISE_CONTROL_KEY]
        NOISE.validate_noise_control_payload(payload)
        clean_obs = {k: v for k, v in obs.items() if k != NOISE_WIRE.NOISE_CONTROL_KEY}
        noise = NOISE.derive_noise(payload, *self._shape)
        derived_sha = NOISE.noise_fingerprint(noise)

        import torch

        policy, model = self._policy, self._policy._model  # noqa: SLF001
        device = policy._pytorch_device  # noqa: SLF001
        seen = []

        def to_sampler_tensor():
            # Identical to Policy.infer: torch.from_numpy(noise).to(device), then add the batch dim.
            t = torch.from_numpy(noise).to(device)
            return t[None, ...]

        orig_base = policy._sample_actions  # noqa: SLF001
        orig_steer = model.sample_actions_with_steering
        steer_was_instance_attr = "sample_actions_with_steering" in vars(model)

        def base_shim(device_, observation, **kw):
            assert "noise" not in kw, "unexpected noise kwarg on baseline path"
            t = to_sampler_tensor()
            seen.append(("sample_actions", tensor_sha256(t), tuple(t.shape), str(t.dtype)))
            return orig_base(device_, observation, noise=t, **kw)

        def steer_shim(device_, observation, **kw):
            assert "noise" not in kw, "unexpected noise kwarg on steering path"
            t = to_sampler_tensor()
            seen.append(("sample_actions_with_steering", tensor_sha256(t), tuple(t.shape), str(t.dtype)))
            return orig_steer(device_, observation, noise=t, **kw)

        with self._lock:
            policy._sample_actions = base_shim  # noqa: SLF001
            model.sample_actions_with_steering = steer_shim
            try:
                result = self._inner.infer(clean_obs)
            finally:
                policy._sample_actions = orig_base  # noqa: SLF001
                if steer_was_instance_attr:
                    model.sample_actions_with_steering = orig_steer
                else:
                    del model.sample_actions_with_steering  # back to the bound class method
        if len(seen) != 1:
            raise RuntimeError(f"expected exactly one sampler call, got {seen}")
        path, sampler_sha, shape, dtype = seen[0]
        if sampler_sha != derived_sha:
            raise RuntimeError(f"sampler noise {sampler_sha} != derived {derived_sha}")
        expect_path = "sample_actions" if self.condition == "BASELINE" else "sample_actions_with_steering"
        if path != expect_path:
            raise RuntimeError(f"condition {self.condition} used sampler path {path}")
        key = {f: int(payload[f]) for f in NOISE_WIRE.NOISE_KEY_FIELDS}
        echo = {"sha256": derived_sha, "sampler_sha256": sampler_sha, "sampler": path, "condition": self.condition}
        echo.update(key)
        result[NOISE_WIRE.NOISE_CONTROL_ECHO_KEY] = echo
        with open(self._log_path, "a") as f:
            f.write(json.dumps({**echo, "shape": list(shape), "dtype": dtype}) + "\n")
        return result

    @property
    def metadata(self) -> dict:
        return {
            **(self._inner.metadata or {}),
            "noise_control_enabled": True,
            "noise_control_shape": list(self._shape),
            "noise_control_scheme": NOISE.NOISE_SCHEME,
            "phase7b_paired_noise": True,
        }


def build_hooks(task_key: str, npz, device: str):
    """The three steered hooks; every one is the candidate ConceptorSteeringHook."""
    task, layer, alpha, beta = TASKS[task_key]
    c_global = cs.get_global_contrastive(npz, task, layer, alpha)
    d = c_global.shape[0]
    c_zero = np.zeros((d, d), dtype=np.float32)
    c_rand = cs.compute_random_conceptor(seed=layer * 100 + int(beta * 10))
    return {
        "BASELINE": None,
        "ORIGINAL_COAST": [(layer, cs.ConceptorSteeringHook(c_global, beta=beta, device=device))],
        "PURE_SHRINKAGE": [(layer, cs.ConceptorSteeringHook(c_zero, beta=beta, device=device))],
        "RANDOM_CONTROL": [(layer, cs.ConceptorSteeringHook(c_rand, beta=beta, device=device))],
    }, {"C_global": c_global, "C_zero": c_zero, "C_random": c_rand}


def verify_shrinkage_hook(hooks: dict, beta: float, device: str) -> dict:
    """Mathematical check of PURE_SHRINKAGE on the hook object that is served.

    fp32: M == (1-b) I exactly, and hook(h) == (1-b) h exactly.
    bf16 (the dtype the action expert feeds this hook): hook(h) == bf16(bf16(1-b) * h) bit for bit,
    and |hook(h) - (1-b) h| / |(1-b) h| stays within bf16 precision.
    """
    import torch

    hook = hooks["PURE_SHRINKAGE"][0][1]
    d = hook.M.shape[0]
    m_ref = (1 - beta) * torch.eye(d, dtype=torch.float32, device=device)
    out = {"M_equals_(1-b)I_fp32": bool(torch.equal(hook.M, m_ref))}
    g = torch.Generator(device="cpu").manual_seed(0)
    h32 = (torch.randn(1, 10, d, generator=g) * 150.0).to(device)
    # The candidate model sets float32_matmul_precision("high") (TF32), so an fp32 matmul is exact only at
    # "highest". The hidden state at this hook is bf16 (Phase 1E), so the TF32 path is never used at runtime.
    prev = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        y32 = hook(None, None, h32)
    finally:
        torch.set_float32_matmul_precision(prev)
    out["fp32_max_abs_diff"] = float((y32 - (1 - beta) * h32).abs().max())
    y_tf32 = hook(None, None, h32)
    out["fp32_tf32_max_rel_err"] = float(((y_tf32 - (1 - beta) * h32).norm(dim=-1) / ((1 - beta) * h32).norm(dim=-1)).max())
    hb = h32.to(torch.bfloat16)
    yb = hook(None, None, (hb, None))[0]
    scale_bf16 = torch.tensor(1 - beta, dtype=torch.float32).to(torch.bfloat16)
    out["bf16_equals_bf16_scalar_multiply"] = bool(torch.equal(yb, hb * scale_bf16.to(device)))
    exact = (1 - beta) * hb.double()
    rel = ((yb.double() - exact).norm(dim=-1) / exact.norm(dim=-1)).max()
    out["bf16_applied_factor"] = float(scale_bf16)
    out["bf16_max_rel_err_vs_exact"] = float(rel)
    hook.reset_logs()
    ok = out["M_equals_(1-b)I_fp32"] and out["fp32_max_abs_diff"] == 0.0 and out["bf16_equals_bf16_scalar_multiply"]
    ok = ok and out["bf16_max_rel_err_vs_exact"] < 2 ** -7
    if not ok:
        raise RuntimeError(f"shrinkage verification failed: {out}")
    return out


def run_client(task_key, num_episodes, port, cond_dir, master_seed, state_offset, max_steps):
    task = TASKS[task_key][0]
    task_id = cs.LIBERO_TASK_IDS[task]
    libero_env_dir = cs.REPO_ROOT / "examples" / "libero_env"
    cmd = [
        str(libero_env_dir / ".venv" / "bin" / "python"),
        str(CLIENT_WRAPPER),
        "--state_offset", str(state_offset),
        "--policy_noise_seed", str(master_seed),
        "--noise_log", str(cond_dir / "client_noise_log.jsonl"),
        "--task_suite_name", "libero_10",
        "--task_id", str(task_id),
        "--num_episodes", str(num_episodes),
        "--port", str(port),
        "--output_dir", str(cond_dir),
    ]  # fmt: skip
    if max_steps is not None:
        cmd += ["--max_steps", str(max_steps)]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    with open(cond_dir / "client_cmd.json", "w") as f:
        json.dump({"cmd": cmd, "cwd": str(libero_env_dir)}, f, indent=2)
    logger.info("Eval: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(libero_env_dir), env=env, capture_output=True, text=True, timeout=7200)
    log_text = (proc.stdout or "") + (proc.stderr or "")
    (cond_dir / "client.log").write_text(log_text)
    if proc.returncode != 0:
        raise RuntimeError(f"client failed rc={proc.returncode}:\n{log_text[-3000:]}")
    matches = cs.SUCCESS_RATE_RE.findall(log_text)
    if not matches:
        raise RuntimeError("no success_rate in client output")
    return float(matches[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-key", required=True, choices=sorted(TASKS))
    ap.add_argument("--master-seed", type=int, required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--state-offset", type=int, default=15)
    ap.add_argument("--num-episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--checkpoint-dir", default=str(MAIN_ROOT / "checkpoints/openpi-libero-2000"))
    args = ap.parse_args()

    task, layer, alpha, beta = TASKS[args.task_key]
    out = pathlib.Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    golden = NOISE.noise_fingerprint(NOISE.derive_noise(GOLDEN_KEY, 10, 32))
    if golden != GOLDEN_SHA:
        raise RuntimeError(f"Phase 1D golden noise fingerprint mismatch: {golden}")

    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, "pi05_libero")
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001
    npz = cs.load_npz()
    hooks, mats = build_hooks(args.task_key, npz, device)
    shrink_check = verify_shrinkage_hook(hooks, beta, device)
    logger.info("PURE_SHRINKAGE verification: %s", shrink_check)

    c = mats["C_global"].astype(np.float64)
    provenance = {
        "task_key": args.task_key,
        "task": task,
        "task_id": cs.LIBERO_TASK_IDS[task],
        "layer": layer,
        "alpha": alpha,
        "beta": beta,
        "master_seed": args.master_seed,
        "repeat": args.repeat,
        "port": args.port,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "state_offset": args.state_offset,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "conditions": args.conditions,
        "candidate_module": cs.__file__,
        "conceptor_npz": str(cs.CONCEPTOR_NPZ),
        "npz_key": f"{task}__L{layer}__{alpha}__C_contrastive",
        "C_global_sha256": hashlib.sha256(np.ascontiguousarray(mats["C_global"]).tobytes()).hexdigest(),
        "C_random_sha256": hashlib.sha256(np.ascontiguousarray(mats["C_random"]).tobytes()).hexdigest(),
        "C_random_seed": layer * 100 + int(beta * 10),
        "C_asymmetry_fro_rel": float(np.linalg.norm(c - c.T) / np.linalg.norm(c)),
        "noise_scheme": NOISE.NOISE_SCHEME,
        "golden_fingerprint_ok": True,
        "shrinkage_verification": shrink_check,
        "torch_compile_disable": os.environ.get("TORCH_COMPILE_DISABLE"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    steered = cs.SteeredPolicyWrapper(policy, steering_hooks=None)
    wrapper = PairedNoiseWrapper(steered, policy, out / "server_noise_log.jsonl")
    cs.start_server_background(wrapper, args.port)

    results = []
    for cond in args.conditions:
        cond_dir = out / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        wrapper.set_condition(cond, cond_dir / "server_noise_log.jsonl")
        steered.update_hooks(hooks[cond])
        t0 = time.time()
        sr = run_client(
            args.task_key, args.num_episodes, args.port, cond_dir, args.master_seed, args.state_offset, args.max_steps
        )
        results.append({"condition": cond, "success_rate": sr, "wall_s": round(time.time() - t0, 1)})
        logger.info("  %s: SR=%.3f", cond, sr)
        (out / "summary.json").write_text(json.dumps({"task": task, "conditions": results}, indent=2))
    logger.info("done: %s", results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()

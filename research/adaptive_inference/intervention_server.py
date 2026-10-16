"""Intervention-operator policy server for the same-state fork pilot (π0.5 PyTorch, LIBERO).

One frozen π0.5 serves every inference operator, selected per request:

    I0  base           frozen VLA, flow sampler with explicit noise
    IL  latent         COAST global conceptor gate h' = h((1-β)I + βC)^T on one action-expert
                       layer, all tokens, all denoising steps (src/openpi/serving/steering.py)
    IC  counterfactual CAG-TF (LIBERO-CF, serve_policy_TF.py): a second pass with the language
                       tokens and mask zeroed, same noise; a = a_u + ω (a_c - a_u) in model space
    (IH, the temporal operator, is client-side: execution horizon; it uses I0 inference.)

All operators go through `PI0Pytorch.sample_actions_with_steering` (eager), so they
share one numerical path; I0 is that path with no hooks. Noise is derived on the
server from an integer key (Phase-1D construction: SeedSequence(key) -> PCG64 ->
standard_normal((H, D), float32)), so branches that query at the same coordinate get
identical noise (common random numbers) whatever operator they run.

Request: the normal LIBERO policy inputs plus
    obs["__aiv__"] = {"op": "I0"|"IL"|"IC", "key": [ints...], "task": <libero task name>,
                      "features": bool}
Response: {"actions", "aiv": {"op", "noise_sha256", "infer_ms", ["features"]}}.

`features` (decision states only) are computed from the pre-intervention state and are
the only router inputs: the proposed first chunk of every operator under the same noise,
K extra base samples, the CAG conditional/unconditional contrast, and the mean
action-expert hidden state at one layer (denoising steps 0 and last).

Run from the repo root (root venv), one GPU:
    CUDA_VISIBLE_DEVICES=0 uv run python research/adaptive_inference/intervention_server.py \
        --port 8600 --conceptor_npz conceptors/aiv_pilot_conceptors.npz
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
import time
from typing import Any

import jax
import numpy as np
import torch
import tyro

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.serving.steering import ConceptorSteeringHook
from openpi.serving.steering import get_conceptor_matrix
from openpi.serving.steering import load_conceptor_npz
from openpi.training import config as _config

logger = logging.getLogger(__name__)

AIV_KEY = "__aiv__"
OPS = ("I0", "IL", "IC")
IMPL_ID = "aiv-intervention-server-v1"

# Frozen operator configuration (INTERVENTION_SPEC.md). IL configs are the COAST paper's
# Table 4 "Global" entries for these LIBERO-10 tasks; CAG ω is the sibling project's frozen 1.5.
IL_CONFIGS: dict[str, dict[str, Any]] = {
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": {
        "layer": 11,
        "alpha": 0.5,
        "beta": 0.3,
    },
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": {"layer": 5, "alpha": 0.5, "beta": 0.1},
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": {
        "layer": 5,
        "alpha": 1.0,
        "beta": 0.1,
    },
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": {
        "layer": 11,
        "alpha": 0.5,
        "beta": 0.3,
    },
}
CAG_OMEGA = 1.5


def derive_noise(key: list[int] | tuple[int, ...], shape: tuple[int, int]) -> np.ndarray:
    if not key or any((not isinstance(k, (int, np.integer))) or k < 0 for k in key):
        raise ValueError(f"noise key must be non-negative ints, got {key!r}")
    ss = np.random.SeedSequence([int(k) for k in key])
    return np.random.Generator(np.random.PCG64(ss)).standard_normal(shape, dtype=np.float32)


def noise_sha256(noise: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(noise, dtype="<f4").tobytes()).hexdigest()


class CaptureHook:
    """Read-only forward hook: records the token-mean hidden state at given denoising steps."""

    def __init__(self, steps: tuple[int, ...]):
        self.steps = steps
        self.current = 0
        self.records: dict[int, np.ndarray] = {}

    def set_denoise_step(self, t: int) -> None:
        self.current = int(t)

    def __call__(self, module, inputs, output):
        if self.current in self.steps:
            h = output[0] if isinstance(output, tuple) else output
            self.records[self.current] = h.detach().float().mean(dim=1)[0].cpu().numpy()
        # returning None leaves the output untouched


class InterventionPolicy:
    def __init__(
        self,
        policy,
        *,
        conceptor_npz: str | None,
        feature_layer: int = 11,
        n_feature_samples: int = 8,
    ):
        self._policy = policy
        self._model = policy._model
        self._device = str(policy._pytorch_device)
        self._sample_kwargs = dict(policy._sample_kwargs)
        self._shape = (int(self._model.config.action_horizon), int(self._model.config.action_dim))
        self._npz = load_conceptor_npz(conceptor_npz) if conceptor_npz else None
        self._il_hooks: dict[str, tuple[int, ConceptorSteeringHook]] = {}
        self._feature_layer = feature_layer
        self._n_feature_samples = n_feature_samples
        self._metadata = dict(policy.metadata)
        self._metadata.update(
            {
                "aiv_impl": IMPL_ID,
                "aiv_ops": list(OPS),
                "aiv_noise_shape": list(self._shape),
                "aiv_il_configs": IL_CONFIGS,
                "aiv_cag_omega": CAG_OMEGA,
                "aiv_feature_layer": feature_layer,
                "aiv_n_feature_samples": n_feature_samples,
                "aiv_sample_kwargs": {k: v for k, v in self._sample_kwargs.items() if isinstance(v, (int, float))},
                "aiv_conceptor_npz": conceptor_npz,
            }
        )

    @property
    def metadata(self) -> dict:
        return self._metadata

    # ------------------------------------------------------------------ helpers
    def _il_hook(self, task: str) -> tuple[int, ConceptorSteeringHook]:
        if task not in self._il_hooks:
            if self._npz is None:
                raise ValueError("IL requested but the server has no --conceptor_npz")
            cfg = IL_CONFIGS[task]
            c = get_conceptor_matrix(self._npz, task=task, layer=cfg["layer"], alpha=cfg["alpha"], strategy="global")
            self._il_hooks[task] = (cfg["layer"], ConceptorSteeringHook(c, beta=cfg["beta"], device=self._device))
        layer, hook = self._il_hooks[task]
        hook.reset_logs()
        return layer, hook

    def _transform(self, obs: dict, *, mask_language: bool) -> tuple[_model.Observation, np.ndarray]:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._policy._input_transform(inputs)
        if mask_language:  # exactly CAG-TF's _mask_language_tokens, applied after the transforms
            inputs = dict(inputs)
            for k in ("tokenized_prompt", "tokenized_prompt_mask"):
                if inputs.get(k) is not None:
                    inputs[k] = np.zeros_like(inputs[k])
        state = np.asarray(inputs["state"])
        tens = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._device)[None, ...], inputs)
        return _model.Observation.from_dict(tens), state

    def _sample(self, observation, noise: np.ndarray, hooks) -> np.ndarray:
        """Model-space (normalized) action chunk (H, D) float32."""
        nt = torch.from_numpy(noise).to(self._device)[None, ...]
        with torch.no_grad():
            x, _ = self._model.sample_actions_with_steering(
                self._device, observation, noise=nt, steering_hooks=hooks or None, **self._sample_kwargs
            )
        return x[0].detach().float().cpu().numpy()

    def _out(self, state: np.ndarray, actions_model: np.ndarray) -> np.ndarray:
        return np.asarray(self._policy._output_transform({"state": state, "actions": actions_model})["actions"])

    def run_op(self, op: str, obs: dict, noise: np.ndarray, task: str, extra_hooks=()) -> dict[str, np.ndarray]:
        observation, state = self._transform(obs, mask_language=False)
        out: dict[str, np.ndarray] = {}
        if op == "I0":
            a = self._sample(observation, noise, list(extra_hooks))
        elif op == "IL":
            layer, hook = self._il_hook(task)
            a = self._sample(observation, noise, [(layer, hook), *extra_hooks])
            out["il_intervention_norm_mean"] = np.asarray(np.mean(hook.intervention_norms), np.float32)
        elif op == "IC":
            cond = self._sample(observation, noise, list(extra_hooks))
            masked_obs, _ = self._transform(obs, mask_language=True)
            uncond = self._sample(masked_obs, noise, [])
            a = uncond + CAG_OMEGA * (cond - uncond)
            out["cag_cond_model"] = cond
            out["cag_uncond_model"] = uncond
        else:
            raise ValueError(f"unknown op {op!r}")
        out["actions_model"] = a
        out["actions"] = self._out(state, a)
        return out

    def features(self, obs: dict, key: list[int], task: str) -> dict[str, Any]:
        noise = derive_noise(key, self._shape)
        cap = CaptureHook(steps=(0, int(self._sample_kwargs.get("num_steps", 10)) - 1))
        base = self.run_op("I0", obs, noise, task, extra_hooks=[(self._feature_layer, cap)])
        il = self.run_op("IL", obs, noise, task) if task in IL_CONFIGS and self._npz is not None else None
        ic = self.run_op("IC", obs, noise, task)
        samples = []
        for j in range(self._n_feature_samples):
            nj = derive_noise([*key, 1_000_000 + j], self._shape)
            samples.append(self.run_op("I0", obs, nj, task)["actions"])
        feats = {
            "latent_step0": cap.records.get(0),
            "latent_last": cap.records.get(max(cap.steps)),
            "chunk_I0": base["actions"],
            "chunk_IC": ic["actions"],
            "chunk_IL": None if il is None else il["actions"],
            "cag_contrast_model_norm": float(np.linalg.norm(ic["cag_cond_model"] - ic["cag_uncond_model"])),
            "il_displacement_model_norm": None
            if il is None
            else float(np.linalg.norm(il["actions_model"] - base["actions_model"])),
            "il_intervention_norm_mean": None if il is None else float(il["il_intervention_norm_mean"]),
            "base_samples": np.stack(samples, 0),
        }
        return feats

    # -------------------------------------------------------------------- serve
    def infer(self, obs: dict) -> dict:
        payload = obs.get(AIV_KEY)
        if payload is None:
            raise ValueError(f"request without {AIV_KEY}: this server only serves keyed intervention requests")
        clean = {k: v for k, v in obs.items() if k != AIV_KEY}
        op = payload["op"]
        key = [int(k) for k in payload["key"]]
        task = str(payload["task"])
        noise = derive_noise(key, self._shape)
        t0 = time.monotonic()
        res = self.run_op(op, clean, noise, task)
        infer_ms = (time.monotonic() - t0) * 1000
        aiv: dict[str, Any] = {"op": op, "key": key, "noise_sha256": noise_sha256(noise), "infer_ms": infer_ms}
        if payload.get("features"):
            t1 = time.monotonic()
            aiv["features"] = self.features(clean, key, task)
            aiv["features_ms"] = (time.monotonic() - t1) * 1000
        if torch.cuda.is_available():
            aiv["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        return {"actions": res["actions"], "aiv": aiv}


@dataclasses.dataclass
class Args:
    port: int = 8600
    config: str = "pi05_libero"
    checkpoint: str = "checkpoints/openpi-libero-2000"
    conceptor_npz: str | None = "conceptors/aiv_pilot_conceptors.npz"
    feature_layer: int = 11
    n_feature_samples: int = 8


def build_policy(args: Args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint, use_pytorch=True)
    return policy


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    policy = build_policy(args)
    ip = InterventionPolicy(
        policy,
        conceptor_npz=args.conceptor_npz,
        feature_layer=args.feature_layer,
        n_feature_samples=args.n_feature_samples,
    )
    logger.info("AIV server metadata: %s", json.dumps({k: str(v) for k, v in ip.metadata.items()})[:2000])
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=ip, host="0.0.0.0", port=args.port, metadata=ip.metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))

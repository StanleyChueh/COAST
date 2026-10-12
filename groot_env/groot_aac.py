"""Server-side Adaptive Action Chunking (AAC) wrapper for the GR00T N1.5 adapter.

`serve.py --aac` wraps `GR00TAdapterPolicy` with `AACGrootPolicyWrapper`. Every
`infer` call:
  1. draws N candidate chunks for the observation in ONE batched pass
     (`GR00TAdapterPolicy.infer_multi_sample`, port of gr00t-multi-sample),
     reseeding with `seed` first when set (upstream `seed_per_step=True`, 42);
  2. runs the upstream AAC selector (`aac.aac_select`) on those candidates;
  3. returns candidate 0's full (H, 12) chunk under "actions" (upstream
     `--chunk_id_selector 0`) plus "aac_horizon" = h* and diagnostics.

The client decides how many actions to execute: `--execution_mode aac` uses
"aac_horizon"; `--execution_mode fixed --replan_steps h` ignores it. That mirrors
upstream, where the fixed-h baselines and AAC both ran against the same N=20,
seed-42 server and executed candidate 0. The N x H x D candidates stay on the
server unless the request carries AAC_RETURN_CANDIDATES_KEY or the server was
started with --aac-return-candidates.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy

import aac

AAC_RETURN_CANDIDATES_KEY = "__aac_return_candidates__"
EXECUTED_CANDIDATE = 0  # upstream chunk_id_selector "0"


class AACGrootPolicyWrapper(_base_policy.BasePolicy):
    def __init__(
        self,
        policy,
        *,
        num_samples: int = aac.DEFAULT_NUM_SAMPLES,
        seed: int | None = aac.DEFAULT_SEED,
        move_th: float = aac.DEFAULT_MOVE_TH,
        method: str = aac.DEFAULT_METHOD,
        return_candidates: bool = False,
        device: str | None = None,
    ) -> None:
        if num_samples < 1:
            raise ValueError(f"--aac-num-samples must be >= 1, got {num_samples}")
        if method != "gaussian_bernoulli":
            raise ValueError(
                f"only the upstream 'gaussian_bernoulli' selector is ported, got {method!r}"
            )
        self._policy = policy
        self._num_samples = int(num_samples)
        self._seed = seed
        self._move_th = float(move_th)
        self._method = method
        self._return_candidates = return_candidates
        self._device = device
        self._num_calls = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "aac_enabled": True,
            "aac_impl": aac.AAC_IMPL_ID,
            "aac_upstream_robocasa_sha": aac.UPSTREAM_ROBOCASA_SHA,
            "aac_upstream_gr00t_sha": aac.UPSTREAM_GR00T_SHA,
            "aac_num_samples": self._num_samples,
            "aac_seed_per_call": self._seed,
            "aac_move_th": self._move_th,
            "aac_method": self._method,
            "aac_executed_candidate": EXECUTED_CANDIDATE,
            "action_horizon": self._action_horizon(),
        }

    def _action_horizon(self) -> int:
        try:
            return int(self._policy._policy.model.action_head.config.action_horizon)
        except AttributeError:
            return 16

    def _cuda(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is always present in groot_env
            return None
        return torch if torch.cuda.is_available() else None

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs = dict(obs)
        want_candidates = (
            bool(obs.pop(AAC_RETURN_CANDIDATES_KEY, False)) or self._return_candidates
        )
        torch = self._cuda()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        action_dict, normalized = self._policy.infer_multi_sample(
            obs, num_samples=self._num_samples, seed=self._seed
        )
        if torch is not None:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred = aac.build_upstream_pred_action_dict(action_dict, normalized)
        result = aac.aac_select(pred, method=self._method, move_th=self._move_th)
        t2 = time.perf_counter()

        candidates = self._policy.action_dict_to_array(action_dict)  # (N, H, 12)
        chunk = candidates[EXECUTED_CANDIDATE]
        horizon = chunk.shape[0]
        if not 1 <= result.h_star <= horizon:
            raise RuntimeError(
                f"AAC selected h*={result.h_star} outside [1, {horizon}]"
            )

        # Candidate disagreement per future step, in the space AAC uses
        # (normalized dims 0:7 = pos, rot, gripper): mean over dims of the std
        # across candidates. Logging only.
        spread = normalized[:, :, :7].std(axis=0).mean(axis=-1)

        diag = result.to_json()
        diag["candidate_spread"] = [float(x) for x in spread]
        diag["sample_ms"] = (t1 - t0) * 1000.0
        diag["select_ms"] = (t2 - t1) * 1000.0
        if torch is not None:
            diag["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        diag["call_index"] = self._num_calls
        self._num_calls += 1

        out: dict[str, Any] = {
            "actions": np.ascontiguousarray(chunk, dtype=np.float32),
            "aac_horizon": int(result.h_star),
            "aac": diag,
        }
        if want_candidates:
            out["aac_candidates"] = np.ascontiguousarray(candidates, dtype=np.float32)
            out["aac_normalized"] = np.ascontiguousarray(normalized, dtype=np.float32)
        return out

    def reset(self) -> None:
        self._policy.reset()

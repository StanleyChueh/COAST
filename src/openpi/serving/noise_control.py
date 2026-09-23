"""Controlled (paired) initial flow noise for pi0 / pi0.5 policy serving.

Research/control mechanism used to compare policy conditions (e.g. baseline vs
conceptor steering) under an identical flow-matching noise schedule. It is
opt-in: the server only installs ``NoiseControlledPolicyWrapper`` when started
with ``--noise_control``, and even then a request without the magic key passes
through untouched.

Protocol (see ``openpi_client.noise_control``): the client attaches
``obs["__noise_control__"] = {master_seed, task_id, init_state, rollout_step}``.
The wrapper

    1. pops the key (downstream wrappers / input transforms never see it),
    2. derives the noise deterministically from the key:

           ss    = np.random.SeedSequence([master_seed, task_id, init_state, rollout_step])
           noise = np.random.Generator(np.random.PCG64(ss)).standard_normal(
                       (action_horizon, action_dim), dtype=np.float32)

       with ``(action_horizon, action_dim)`` read from the loaded model,
    3. calls ``inner.infer(clean_obs, noise=noise)``. ``Policy.infer`` and
       ``Policy.infer_with_steering`` convert it identically and hand it to
       ``sample_actions`` / ``sample_actions_with_steering``, replacing the
       sampler's own ``sample_noise()`` draw (i.i.d. N(0, 1) either way),
    4. echoes ``{"sha256": ..., **key}`` in the response under
       ``"noise_control"`` so clients can verify pairing after the fact.

Only the initial flow noise is controlled. The global torch RNG is not reseeded,
and GPU kernels are not forced to be deterministic.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from openpi_client.noise_control import NOISE_CONTROL_ECHO_KEY
from openpi_client.noise_control import NOISE_CONTROL_KEY
from openpi_client.noise_control import NOISE_KEY_FIELDS

NOISE_SCHEME = "numpy.SeedSequence(master_seed,task_id,init_state,rollout_step)->PCG64->standard_normal(float32)"


def validate_noise_control_payload(payload: Any) -> None:
    """Raise ``ValueError`` if the client-supplied noise-control dict is malformed."""
    if not isinstance(payload, dict):
        raise ValueError(f"{NOISE_CONTROL_KEY} payload must be a dict, got {type(payload).__name__}")
    if set(payload) != set(NOISE_KEY_FIELDS):
        raise ValueError(f"{NOISE_CONTROL_KEY} payload must have exactly {NOISE_KEY_FIELDS}, got {sorted(payload)}")
    for field in NOISE_KEY_FIELDS:
        value = payload[field]
        # bool is a subclass of int; reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"{NOISE_CONTROL_KEY}.{field} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{NOISE_CONTROL_KEY}.{field} must be non-negative, got {value}")


def derive_noise(payload: dict, action_horizon: int, action_dim: int) -> np.ndarray:
    """Deterministic float32 N(0, 1) noise of shape (action_horizon, action_dim) for a noise key."""
    validate_noise_control_payload(payload)
    seed_sequence = np.random.SeedSequence([int(payload[field]) for field in NOISE_KEY_FIELDS])
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)


def noise_fingerprint(noise: np.ndarray) -> str:
    """SHA-256 over the little-endian float32 bytes of ``noise`` (C order)."""
    return hashlib.sha256(np.ascontiguousarray(noise, dtype="<f4").tobytes()).hexdigest()


def noise_shape_from_policy(policy: Any) -> tuple[int, int]:
    """Read ``(action_horizon, action_dim)`` from a ``Policy``'s loaded model.

    PyTorch ``PI0Pytorch`` stores them on ``model.config``; JAX models expose
    them as attributes on the model itself.
    """
    model = getattr(policy, "_model", None)
    for source in (getattr(model, "config", None), model):
        horizon = getattr(source, "action_horizon", None)
        dim = getattr(source, "action_dim", None)
        if isinstance(horizon, int) and isinstance(dim, int):
            return horizon, dim
    raise ValueError(f"Could not determine action_horizon/action_dim from policy model {type(model).__name__}")


class NoiseControlledPolicyWrapper:
    """Wrap a policy so it honors an obs[\"__noise_control__\"] key.

    Must be the outermost policy wrapper so the key is removed before any other
    wrapper (e.g. ``SteeredPolicyWrapper``) or input transform sees the obs.
    The inner policy's ``infer`` must accept a ``noise=`` keyword.
    """

    def __init__(self, policy: Any, *, action_horizon: int, action_dim: int) -> None:
        self._policy = policy
        self._noise_shape = (int(action_horizon), int(action_dim))

    def infer(self, obs: dict) -> dict:
        if not isinstance(obs, dict) or NOISE_CONTROL_KEY not in obs:
            return self._policy.infer(obs)
        payload = obs[NOISE_CONTROL_KEY]
        validate_noise_control_payload(payload)
        clean_obs = {k: v for k, v in obs.items() if k != NOISE_CONTROL_KEY}
        state = clean_obs.get("observation/state")
        if state is not None and np.ndim(state) != 1:
            raise ValueError(f"{NOISE_CONTROL_KEY} supports single (unbatched) observations only")

        noise = derive_noise(payload, *self._noise_shape)
        result = self._policy.infer(clean_obs, noise=noise)
        result[NOISE_CONTROL_ECHO_KEY] = {
            "sha256": noise_fingerprint(noise),
            "shape": list(noise.shape),
            **{field: int(payload[field]) for field in NOISE_KEY_FIELDS},
        }
        return result

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def metadata(self) -> dict:
        underlying = getattr(self._policy, "metadata", {}) or {}
        return {
            **underlying,
            "noise_control_enabled": True,
            "noise_control_shape": list(self._noise_shape),
            "noise_control_scheme": NOISE_SCHEME,
        }

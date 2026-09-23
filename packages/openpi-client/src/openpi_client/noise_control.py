"""Client-side protocol for controlled (paired) policy flow noise.

Research/control mechanism: a client attaches ``obs[NOISE_CONTROL_KEY]`` with an
integer key identifying the decision point. A server started with
``--noise_control`` (``openpi.serving.noise_control.NoiseControlledPolicyWrapper``)
pops the key, deterministically derives the initial flow-matching noise from it,
and passes that noise explicitly to the sampler. Two runs (e.g. baseline vs
steered) that request inference at the same key therefore start denoising from
the identical noise tensor.

Pure stdlib (no openpi / torch / numpy) so it imports from the root venv and the
Python 3.8 LIBERO sub-venv alike. The server imports ``NOISE_CONTROL_KEY`` and
``NOISE_KEY_FIELDS`` from here so the two sides of the wire can't drift.

Example client use:

    from openpi_client.noise_control import NOISE_CONTROL_KEY, build_noise_control_payload

    element[NOISE_CONTROL_KEY] = build_noise_control_payload(
        master_seed=100, task_id=2, init_state=15, rollout_step=0
    )
    result = policy.infer(element)
    result[NOISE_CONTROL_ECHO_KEY]["sha256"]  # fingerprint of the noise actually used
"""

from __future__ import annotations

from typing import Dict

# On-wire magic key, same pattern as ``__steering__`` / ``__collect__``.
NOISE_CONTROL_KEY = "__noise_control__"

# Response key under which the server echoes the key and noise fingerprint.
NOISE_CONTROL_ECHO_KEY = "noise_control"

# Ordered fields of the noise key. The server seeds the noise generator with
# these integers in exactly this order, so the order is part of the protocol.
NOISE_KEY_FIELDS = ("master_seed", "task_id", "init_state", "rollout_step")


def build_noise_control_payload(
    master_seed: int,
    task_id: int,
    init_state: int,
    rollout_step: int,
) -> Dict[str, int]:
    """Construct the obs[NOISE_CONTROL_KEY] dict with the correct schema."""
    return {
        "master_seed": int(master_seed),
        "task_id": int(task_id),
        "init_state": int(init_state),
        "rollout_step": int(rollout_step),
    }

"""Exact snapshot / restore of a LIBERO (robosuite 1.4, MuJoCo 3.x) environment.

LIBERO's own ``get_sim_state()`` / ``set_init_state()`` serialise only (time, qpos,
qvel). That drops act, ctrl, qacc_warmstart (the constraint-solver warm start),
mocap targets, applied forces and the python-side state of the robosuite env, its
controllers and grippers (e.g. the Panda gripper's ``current_action``), so a
"restored" state does not continue bit-identically (measured: divergence on the first
step). This module saves MuJoCo's full integration state plus that python-side state
and the observation, and hashes them so callers can verify identity before branching.

Python 3.8 compatible (runs in examples/libero_env/.venv).
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

_SCALAR_TYPES = (int, float, bool, np.generic)


def _robosuite_env(env):
    """OffScreenRenderEnv -> the wrapped robosuite env."""
    return env.env if hasattr(env, "env") and hasattr(env.env, "sim") else env


def _plain_attrs(obj) -> Dict[str, Any]:
    """Copy of the numeric python-side state of a robosuite object (no sim refs)."""
    out = {}
    for k, v in vars(obj).items():
        if isinstance(v, np.ndarray) or isinstance(v, _SCALAR_TYPES) or v is None:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (np.ndarray,) + _SCALAR_TYPES) for x in v):
            out[k] = copy.deepcopy(v)
        elif type(v).__name__ in ("DeltaBuffer", "RingBuffer"):
            out[k] = copy.deepcopy(v)
    return out


def _stateful_objects(renv) -> List[Tuple[str, Any]]:
    objs = []
    for i, robot in enumerate(renv.robots):
        objs.append(("robot{}".format(i), robot))
        grippers = getattr(robot, "gripper", None)
        if isinstance(grippers, dict):
            for arm, g in grippers.items():
                objs.append(("robot{}.gripper.{}".format(i, arm), g))
        elif grippers is not None:
            objs.append(("robot{}.gripper".format(i), grippers))
        ctrl = getattr(robot, "controller", None)
        if ctrl is not None:
            objs.append(("robot{}.controller".format(i), ctrl))
            interp = getattr(ctrl, "interpolator_pos", None)
            if interp is not None:
                objs.append(("robot{}.controller.interpolator_pos".format(i), interp))
            interp = getattr(ctrl, "interpolator_ori", None)
            if interp is not None:
                objs.append(("robot{}.controller.interpolator_ori".format(i), interp))
    for name, obs in getattr(renv, "_observables", {}).items():
        objs.append(("observable:" + name, obs))
    return objs


class Snapshot:
    """mjSTATE_INTEGRATION state vector + robosuite python-side state + observation."""

    def __init__(
        self,
        data: np.ndarray,
        env_attrs: Dict[str, Any],
        obj_attrs: Dict[str, Dict[str, Any]],
        obs: Dict[str, Any],
    ):
        self.data = data
        self.env_attrs = env_attrs
        self.obj_attrs = obj_attrs
        self.obs = obs


_ENV_ATTRS = ("timestep", "cur_time", "done")


def capture(env, obs: Dict[str, Any]) -> Snapshot:
    """`obs` is the observation the env returned at this state (from step/reset).

    It is stored verbatim: robosuite computes step-time observations inside the
    substep loop, which a forced recomputation after restore does not reproduce
    bitwise, so the decision-state observation is part of the snapshot.
    """
    renv = _robosuite_env(env)
    m = renv.sim.model._model
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    dst = np.empty(mujoco.mj_stateSize(m, spec), dtype=np.float64)
    mujoco.mj_getState(m, renv.sim.data._data, dst, spec)
    env_attrs = {k: copy.deepcopy(getattr(renv, k)) for k in _ENV_ATTRS if hasattr(renv, k)}
    obj_attrs = {name: _plain_attrs(o) for name, o in _stateful_objects(renv)}
    return Snapshot(dst, env_attrs, obj_attrs, copy.deepcopy(obs))


def restore(env, snap: Snapshot) -> Dict[str, np.ndarray]:
    """Restore `snap` in place and return (a copy of) the observation at that state.

    mjSTATE_INTEGRATION is everything MuJoCo needs for exact integration: time, qpos,
    qvel, act, qacc_warmstart, ctrl, qfrc_applied, xfrc_applied, mocap, eq_active,
    userdata and plugin state. Derived quantities are not copied: robosuite calls
    ``sim.forward()`` before every substep, which recomputes them from this state, and
    the decision-state observation itself is stored in the snapshot.
    """
    renv = _robosuite_env(env)
    m = renv.sim.model._model
    mujoco.mj_setState(m, renv.sim.data._data, snap.data, mujoco.mjtState.mjSTATE_INTEGRATION)
    for k, v in snap.env_attrs.items():
        setattr(renv, k, copy.deepcopy(v))
    objs = dict(_stateful_objects(renv))
    for name, attrs in snap.obj_attrs.items():
        o = objs[name]
        for k, v in attrs.items():
            setattr(o, k, copy.deepcopy(v))
    return copy.deepcopy(snap.obs)


def state_hash(env) -> str:
    """SHA-256 over the dynamic mjData state that determines the future (bitwise)."""
    renv = _robosuite_env(env)
    d = renv.sim.data._data
    h = hashlib.sha256()
    for arr in (
        np.array([d.time]),
        d.qpos,
        d.qvel,
        d.act,
        d.ctrl,
        d.qacc_warmstart,
        d.mocap_pos,
        d.mocap_quat,
        d.qfrc_applied,
        d.xfrc_applied,
    ):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(repr([getattr(renv, k, None) for k in _ENV_ATTRS]).encode())
    return h.hexdigest()


def obs_hash(policy_inputs: Dict[str, Any]) -> str:
    """SHA-256 over the exact arrays sent to the policy (images + state), in key order."""
    h = hashlib.sha256()
    for k in sorted(policy_inputs):
        v = policy_inputs[k]
        if isinstance(v, np.ndarray):
            h.update(k.encode())
            h.update(str(v.dtype).encode())
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()

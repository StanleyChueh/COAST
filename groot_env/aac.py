"""Adaptive Action Chunking (AAC) chunk-size selector, ported from the official code.

Paper: Liang et al., "Adaptive Action Chunking at Inference-time for
Vision-Language-Action Models", CVPR 2026, arXiv:2604.04161.

Provenance (ported nearly verbatim, numerics unchanged):
  https://github.com/Adaptive-Action-Chunking/robocasa @ fed3e6b5eb348160dd0570f326f726758fee9056
    robocasa/demos/action_optimization/action_entropy_v2.py
      _safe_logdet_cov, gaussian_entropy_from_samples, bernoulli_entropy_from_samples,
      action_entropy_gausssian_bernoulli, action_magnitude, find_first_idx, select_chunk_size
    robocasa/demos/help_function.py
      axisangle2quat, quat_multiply, quat2axisangle, compose_rotations

Upstream runs this on the CLIENT, on the dict returned by the multi-sample GR00T
server (gr00t-multi-sample @ 11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a). COAST runs it
on the SERVER (so the N x H x D candidate tensor never crosses the WebSocket), on a
dict built to have exactly the upstream layout -- see `build_upstream_pred_action_dict`.

Upstream `pred_action_dict` layout for RoboCasa (N candidates, H=16):
  "action.end_effector_position": (N, H, 3)   unnormalized
  "action.end_effector_rotation": (N, H, 3)   unnormalized, axis-angle
  "action.gripper_close":         (N, H)      unnormalized, binary {0, 1}
  "action.base_motion":          (N, H, 4)   (unused by AAC)
  "action.control_mode":         (N, H)      (unused by AAC)
  "normalized_action":            (N, H, 32)  model output before unnormalization

Do not "fix" anything in the ported functions without recording it in
experiments/aac/AAC_REPRO_NOTES.md: they are the reproduction reference.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
from scipy.stats import entropy

AAC_IMPL_ID = "coast-aac-port-v1"
UPSTREAM_ROBOCASA_SHA = "fed3e6b5eb348160dd0570f326f726758fee9056"
UPSTREAM_GR00T_SHA = "11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a"

# Upstream defaults (inference_client_action_entropy.py, AAC_evaluation.sh,
# policy_server_robocasa.py).
DEFAULT_METHOD = "gaussian_bernoulli"
DEFAULT_MOVE_TH = 3.0
DEFAULT_NUM_SAMPLES = 20
DEFAULT_SEED = 42


# --------------------------------------------------------------------------- #
# Rotation helpers -- verbatim from upstream robocasa/demos/help_function.py.  #
# --------------------------------------------------------------------------- #


def axisangle2quat(vec):
    """Converts scaled axis-angle to (x, y, z, w) quat."""
    angle = np.linalg.norm(vec)
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def quat_multiply(quaternion1, quaternion0):
    """Return multiplication of two (x, y, z, w) quaternions (q1 * q0)."""
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )


def quat2axisangle(quat):
    """Converts (x, y, z, w) quaternion to axis-angle."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def compose_rotations(delta_rs):
    q_total = np.array([0.0, 0.0, 0.0, 1.0])
    for r in delta_rs:
        q = axisangle2quat(r)
        q_total = quat_multiply(q, q_total)
    return quat2axisangle(q_total)


# --------------------------------------------------------------------------- #
# Entropy + selector -- verbatim from upstream action_entropy_v2.py.           #
# --------------------------------------------------------------------------- #


def _safe_logdet_cov(X, eps=1e-6):
    X = np.asarray(X)
    N, d = X.shape
    if N <= 1:
        return -np.inf
    Sigma = np.cov(X, rowvar=False)
    Sigma = Sigma + eps * np.eye(d)
    try:
        L = np.linalg.cholesky(Sigma)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return -np.inf
    return logdet


def gaussian_entropy_from_samples(X):
    X = np.asarray(X)
    N, d = X.shape
    if N <= 1:
        return 0.0
    logdet = _safe_logdet_cov(X)
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)


def bernoulli_entropy_from_samples(g):
    g = np.asarray(g).ravel()
    p1 = float(np.clip(np.mean(g), 1e-9, 1 - 1e-9))
    p0 = 1.0 - p1
    return entropy([p0, p1], base=np.e)


def action_entropy_gausssian_bernoulli(
    pred_action_dict, w_pos=1.0, w_rot=1.0, w_grip=1.0, normalized_pose=True
):
    # (sic: upstream spelling "gausssian" kept for grep-ability against upstream.)
    if normalized_pose:
        end_effector_position = pred_action_dict["normalized_action"][:, :, :3]
        end_effector_rotation = pred_action_dict["normalized_action"][:, :, 3:6]
    else:
        end_effector_position = pred_action_dict["action.end_effector_position"]
        end_effector_rotation = pred_action_dict["action.end_effector_rotation"]

    gripper_close = pred_action_dict["action.gripper_close"]

    N, T, D = end_effector_position.shape
    step_H_pos, step_H_rot, step_H_grip, step_H_total = [], [], [], []
    for t in range(T):
        H_pos = gaussian_entropy_from_samples(end_effector_position[:, t, :])
        H_rot = gaussian_entropy_from_samples(end_effector_rotation[:, t, :])
        H_grip = bernoulli_entropy_from_samples(gripper_close[:, t])
        H_total = w_pos * H_pos + w_rot * H_rot + w_grip * H_grip
        step_H_pos.append(H_pos)
        step_H_rot.append(H_rot)
        step_H_grip.append(H_grip)
        step_H_total.append(H_total)

    step_H_pos = np.array(step_H_pos)
    step_H_rot = np.array(step_H_rot)
    step_H_grip = np.array(step_H_grip)
    step_H_total = np.array(step_H_total)

    H_chunk_mean = [float(np.mean(step_H_total[:k])) for k in range(1, T + 1)]

    return {
        "H_pos": step_H_pos,
        "H_rot": step_H_rot,
        "H_grip": step_H_grip,
        "total": step_H_total,
        "chunk_mean": H_chunk_mean,
    }


def action_magnitude(pred_action_dict, end_to_end=True, chunk_id=0):
    # NOTE: upstream ignores `chunk_id` and always uses candidate 0 ([0,]).
    end_effector_position = np.asarray(
        pred_action_dict["action.end_effector_position"][0,]
    )
    end_effector_rotation = np.asarray(
        pred_action_dict["action.end_effector_rotation"][0,]
    )
    gripper_close = np.asarray(pred_action_dict["action.gripper_close"][0,]).astype(int)
    chunk_size = list(range(2, 17))
    chunk_m = []
    wpos = 1
    wrot = 1
    wgrip = 0.2
    for k in chunk_size:
        if end_to_end:
            dpos = np.linalg.norm(np.sum(end_effector_position[:k,], axis=0))
            drot = np.linalg.norm(compose_rotations(end_effector_rotation[:k,]))
        else:
            dpos_t = np.linalg.norm(end_effector_position[:k,], axis=1)
            dpos = np.sum(dpos_t)
            drot_t = np.linalg.norm(end_effector_rotation[:k,], axis=1)
            drot = np.sum(drot_t)
        gripper_toggle = int(np.any(np.diff(gripper_close[:k,])))
        m = wpos * dpos + wrot * drot + wgrip * gripper_toggle
        chunk_m.append(m)
    return chunk_m


def find_first_idx(chunk_m, move_th):
    for i, m in enumerate(chunk_m):
        if m > move_th:
            return i + 2
    return 16


def select_chunk_size(pred_action_dict: dict, method, move_th):
    """Upstream entry point. Returns (h_star, breakdown)."""
    if method == "gaussian_bernoulli":
        breakdown = action_entropy_gausssian_bernoulli(pred_action_dict)
    else:
        raise ValueError

    vals = breakdown["chunk_mean"]
    diffs = np.diff(vals)
    best_k = int(np.argmax(diffs)) + 1
    best_k_entropy = max(best_k, 2)

    chunk_m = action_magnitude(pred_action_dict)
    min_id = find_first_idx(chunk_m, move_th)

    best_k = max(min_id, best_k_entropy)
    return best_k, breakdown


# --------------------------------------------------------------------------- #
# COAST-side glue (not upstream).                                              #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class AACResult:
    """Everything needed to audit one AAC decision."""

    h_star: int
    # Entropy diagnostics (length H each, except diffs: H-1).
    H_pos: np.ndarray
    H_rot: np.ndarray
    H_grip: np.ndarray
    H_total: np.ndarray
    chunk_mean: np.ndarray  # Ebar_h for h = 1..H
    chunk_mean_diffs: np.ndarray  # Ebar_{h+1} - Ebar_h for h = 1..H-1
    raw_argmax: int  # argmax over diffs (0-based)
    h_entropy: int  # max(raw_argmax + 1, 2)
    # Minimum-motion lower bound (xi).
    chunk_magnitude: np.ndarray  # m(l) for l = 2..16
    h_min_motion: int  # find_first_idx(m, move_th)
    move_th: float
    num_samples: int
    horizon: int

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                out[f.name] = [float(x) for x in v.tolist()]
            elif isinstance(v, (np.integer,)):
                out[f.name] = int(v)
            elif isinstance(v, (np.floating,)):
                out[f.name] = float(v)
            else:
                out[f.name] = v
        return out


def aac_select(
    pred_action_dict: dict,
    *,
    method: str = DEFAULT_METHOD,
    move_th: float = DEFAULT_MOVE_TH,
) -> AACResult:
    """Run the upstream selector and return its intermediate quantities.

    `h_star` is taken from the upstream `select_chunk_size` call itself; the
    other fields recompute the same intermediate values for logging only.
    """
    h_star, breakdown = select_chunk_size(
        pred_action_dict, method=method, move_th=move_th
    )
    chunk_mean = np.asarray(breakdown["chunk_mean"], dtype=np.float64)
    diffs = np.diff(chunk_mean)
    raw_argmax = int(np.argmax(diffs))
    chunk_m = np.asarray(action_magnitude(pred_action_dict), dtype=np.float64)
    h_min_motion = find_first_idx(chunk_m, move_th)
    n, horizon = np.asarray(pred_action_dict["normalized_action"]).shape[:2]
    result = AACResult(
        h_star=int(h_star),
        H_pos=np.asarray(breakdown["H_pos"], dtype=np.float64),
        H_rot=np.asarray(breakdown["H_rot"], dtype=np.float64),
        H_grip=np.asarray(breakdown["H_grip"], dtype=np.float64),
        H_total=np.asarray(breakdown["total"], dtype=np.float64),
        chunk_mean=chunk_mean,
        chunk_mean_diffs=diffs,
        raw_argmax=raw_argmax,
        h_entropy=max(raw_argmax + 1, 2),
        chunk_magnitude=chunk_m,
        h_min_motion=int(h_min_motion),
        move_th=float(move_th),
        num_samples=int(n),
        horizon=int(horizon),
    )
    assert result.h_star == max(result.h_min_motion, result.h_entropy)
    return result


def build_upstream_pred_action_dict(
    action_dict: dict[str, np.ndarray], normalized_action: np.ndarray
) -> dict[str, np.ndarray]:
    """Shape a COAST multi-sample output like upstream `Gr00tPolicy.get_action`.

    Upstream applies `np.squeeze` to every output, which turns the (N, H, 1)
    gripper/control-mode arrays into (N, H). That matters: `action_magnitude`
    calls `np.diff` on the gripper along the LAST axis, so an (H, 1) array would
    silently yield "no toggle". We drop only trailing singleton dims so the
    candidate axis survives for N=1 too (upstream's global squeeze breaks N=1).
    """
    out: dict[str, np.ndarray] = {}
    for k, v in action_dict.items():
        arr = np.asarray(v)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        out[k] = arr
    out["normalized_action"] = np.asarray(normalized_action)
    return out

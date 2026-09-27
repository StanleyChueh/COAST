"""Phase 7B: operator decomposition of the paper Global configurations (offline; no rollouts).

Implemented transformation (candidate ConceptorSteeringHook, row vectors): h' = h @ M.T, M = (1-b)I + bC.
Column form: h' = (1-b)h + b C h. Decomposition, in the implemented row form:
    delta_coast  = h @ M.T - h
    delta_shrink = h @ ((1-b)I).T - h = -b h
    r_C          = h' _coast - h'_shrink = b h @ C.T            (column form: b C h)
    check        : delta_coast - delta_shrink - r_C == 0
C is used exactly as stored (it is not symmetric; ||C - C.T||_F / ||C||_F is reported).

Two variants:
  exact_fp64    M in float64, as written;
  realized_bf16 M cast to bf16 first (as the hook does, M.to(h.dtype) with h bf16), both for COAST
                (bf16(M_coast)) and PURE_SHRINKAGE (bf16((1-b)I)); r_C = h @ (bf16(M_coast) - bf16(M_shrink)).T.
                The matmul itself is done in float64, so only the matrix rounding is modelled.

Hidden-state sets (per task, at its layer):
  fit      the authors' fit token vectors (every 10th inference step, all 10 denoising steps; Phase 7A set);
  heldout  BASELINE hidden states at the first observation of held-out states 15-44, both master seeds, all 10
           denoising steps (phase7b_action_probe.py output), if <probe_dir> is given.

Metrics are per-token values averaged over tokens (mean; min/max also kept for the cosines).
The same metrics are reported for RANDOM_CONTROL (C = candidate random conceptor at the same layer and b).

Usage (repo root): .venv/bin/python research/reproduction/tools/phase7b_operator_decomposition.py \
    <odh_root> <out_csv> [<probe_dir>]
"""

# ruff: noqa: N803, N806  (matrix names follow the math: H, C, M)
from __future__ import annotations

import csv
import glob
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/stanley/Stanley_ws/COAST-research/COAST-paper-candidate/experiments/pi05_libero/src")
from conceptor_steering import compute_random_conceptor  # candidate code, for the random control

ACT = pathlib.Path("activations/pi05-libero-activations-v1-2000-15env/openpi-libero-2000")
LAYER_IDX = {0: 0, 5: 1, 11: 2, 17: 3}
TASKS = {
    "LR2a": ("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", 11, 0.5, 0.3),
    "KS4": ("KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", 5, 1.0, 0.1),
    "LR1": ("LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", 11, 0.5, 0.3),
}
EPS = 1e-12


def bf16(A: np.ndarray) -> np.ndarray:
    return torch.from_numpy(A.astype(np.float32)).to(torch.bfloat16).to(torch.float64).numpy()


def cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + EPS)


def decompose(H: np.ndarray, C: np.ndarray, beta: float, *, realized: bool) -> dict:
    d = C.shape[0]
    M_coast = (1 - beta) * np.eye(d) + beta * C
    M_shrink = (1 - beta) * np.eye(d)
    if realized:
        M_coast, M_shrink = bf16(M_coast), bf16(M_shrink)
    delta_coast = H @ M_coast.T - H
    delta_shrink = H @ M_shrink.T - H
    r_c = H @ (M_coast - M_shrink).T if realized else beta * (H @ C.T)
    nh = np.linalg.norm(H, axis=1)
    ndc = np.linalg.norm(delta_coast, axis=1)
    nds = np.linalg.norm(delta_shrink, axis=1)
    nr = np.linalg.norm(r_c, axis=1)
    recon = np.linalg.norm(delta_coast - delta_shrink - r_c, axis=1) / (ndc + EPS)
    c_dc_ds = cos_rows(delta_coast, delta_shrink)
    c_dc_h = cos_rows(delta_coast, H)
    c_r_h = cos_rows(r_c, H)
    return {
        "rel_delta_coast": float(np.mean(ndc / nh)),
        "rel_delta_shrink": float(np.mean(nds / nh)),
        "rel_r_C": float(np.mean(nr / nh)),
        "r_C_over_delta_coast": float(np.mean(nr / ndc)),
        "cos_delta_coast_delta_shrink": float(np.mean(c_dc_ds)),
        "cos_delta_coast_delta_shrink_min": float(np.min(c_dc_ds)),
        "cos_delta_coast_h": float(np.mean(c_dc_h)),
        "cos_delta_coast_h_max": float(np.max(c_dc_h)),
        "cos_r_C_h": float(np.mean(c_r_h)),
        "hCh": float(np.mean(np.einsum("ij,jk,ik->i", H, C, H) / nh**2)),
        "quota": float(np.trace(C) / d),
        "rel_decomp_error": float(np.max(recon)),
        "shrink_applied_factor": float(M_shrink[0, 0]),
    }


def fit_vectors(task: str, layer: int) -> np.ndarray:
    files = sorted(glob.glob(str(ACT / task / "episode_*" / "step_*" / "suffix_residual.npz")))[::10]
    return np.concatenate(
        [np.load(f)["all_suffix_residual"][:, LAYER_IDX[layer]].reshape(-1, 1024).astype(np.float64) for f in files]
    )


def main() -> None:
    odh, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    probe = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
    rows = []
    for key, (task, layer, alpha, beta) in TASKS.items():
        z = np.load(odh / key / "libero_conceptors.npz")
        pre = f"{task}__L{layer}__{alpha}__"
        C = z[pre + "C_contrastive"].astype(np.float32).astype(np.float64)  # the hook builds M from float32 C
        Cs = z[pre + "C_success"].astype(np.float64)
        Cf = z[pre + "C_failure"].astype(np.float64)
        overlap = float(np.trace(Cs @ Cf) / np.sqrt(np.trace(Cs @ Cs) * np.trace(Cf @ Cf)))
        eig = np.linalg.eigvals(C).real
        Cr = compute_random_conceptor(seed=layer * 100 + int(beta * 10)).astype(np.float64)
        sets = {"fit": fit_vectors(task, layer)}
        if probe is not None and (probe / f"{key}_probe.npz").exists():
            sets["heldout"] = np.load(probe / f"{key}_probe.npz")["hidden"].reshape(-1, 1024).astype(np.float64)
        for set_name, H in sets.items():
            for op, mat in (("ORIGINAL_COAST", C), ("RANDOM_CONTROL", Cr)):
                for variant in ("exact_fp64", "realized_bf16"):
                    m = decompose(H, mat, beta, realized=variant == "realized_bf16")
                    rows.append(
                        {
                            "task": key,
                            "hidden_set": set_name,
                            "operator": op,
                            "variant": variant,
                            "layer": layer,
                            "alpha": alpha,
                            "beta": beta,
                            "n_token_vectors": len(H),
                            **{k: (f"{v:.3e}" if k == "rel_decomp_error" else round(v, 6)) for k, v in m.items()},
                            "C_asym_fro_rel": round(float(np.linalg.norm(mat - mat.T) / np.linalg.norm(mat)), 6),
                            "C_eig_min": round(float(eig.min()), 6) if op == "ORIGINAL_COAST" else "",
                            "C_eig_max": round(float(eig.max()), 6) if op == "ORIGINAL_COAST" else "",
                            "overlap_s_f": round(overlap, 4) if op == "ORIGINAL_COAST" else "",
                        }
                    )
                    print(rows[-1])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

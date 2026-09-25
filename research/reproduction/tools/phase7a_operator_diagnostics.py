"""Phase 7A (exploratory, offline): operator geometry of the paper Global configurations.

Run only after the Phase 7A success-rate records are frozen. No rollouts.

For each task, at its paper Global configuration (layer L, alpha a, beta b), using the per-task candidate
NPZ built by for_subin/build_conceptors.py @ 29059a5, and the authors' released fit activations
(token-level vectors at layer L, all 10 denoising steps, every 10th inference step):
  - rel_dh    mean ||dh|| / ||h||, with dh = h @ M.T - h and M = (1-b)I + bC   (C = C_contrastive)
  - cos_dh_h  mean cos(dh, h)                                          (-1 = pure shrink toward 0)
  - quota     tr(C) / d
  - hCh       mean h^T C h / ||h||^2                                    (energy fraction inside C)
  - overlap   tr(Cs Cf) / sqrt(tr(Cs^2) tr(Cf^2)) at alpha a
The same numbers are reported for the candidate's random control at (L, b).

Usage (repo root): .venv/bin/python research/reproduction/tools/phase7a_operator_diagnostics.py <odh_root> <out_csv>
   <odh_root>/<KEY>/libero_conceptors.npz; KS3 is read from <KS3_NPZ> if it exists (Phase 6C reference).
"""

# ruff: noqa: N803, N806  (matrix names follow the math: H, C, M)
from __future__ import annotations

import csv
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/stanley/Stanley_ws/COAST-research/COAST-paper-candidate/experiments/pi05_libero/src")
from conceptor_steering import compute_random_conceptor  # candidate code, for the random control

ACT = pathlib.Path("activations/pi05-libero-activations-v1-2000-15env/openpi-libero-2000")
LAYER_IDX = {0: 0, 5: 1, 11: 2, 17: 3}
TASKS = {
    "KS3": ("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", 5, 0.5, 0.1),
    "LR2a": ("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", 11, 0.5, 0.3),
    "KS4": ("KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", 5, 1.0, 0.1),
    "LR1": ("LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", 11, 0.5, 0.3),
}
KS3_NPZ = pathlib.Path(
    "/tmp/claude-1000/-home-stanley-Stanley-ws-COAST-research-COAST/dd2dd40c-4801-4748-8fa9-5fc7ffed3e09/"
    "scratchpad/phase6c/openpi_data_home/libero_conceptors.npz"
)


def metrics(H: np.ndarray, C: np.ndarray, beta: float) -> dict:
    d = C.shape[0]
    M = (1 - beta) * np.eye(d) + beta * C
    dh = H @ M.T - H
    nh = np.linalg.norm(H, axis=1)
    ndh = np.linalg.norm(dh, axis=1)
    return {
        "rel_dh": float(np.mean(ndh / nh)),
        "cos_dh_h": float(np.mean(np.sum(dh * H, 1) / (ndh * nh))),
        "quota": float(np.trace(C) / d),
        "hCh": float(np.mean(np.einsum("ij,jk,ik->i", H, C, H) / nh**2)),
    }


def main() -> None:
    odh, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    rows = []
    for key, (task, layer, alpha, beta) in TASKS.items():
        npz_path = KS3_NPZ if key == "KS3" else odh / key / "libero_conceptors.npz"
        if not npz_path.exists():
            continue
        z = np.load(npz_path)
        pre = f"{task}__L{layer}__{alpha}__"
        C = z[pre + "C_contrastive"].astype(np.float64)
        Cs = z[pre + "C_success"].astype(np.float64)
        Cf = z[pre + "C_failure"].astype(np.float64)
        overlap = float(np.trace(Cs @ Cf) / np.sqrt(np.trace(Cs @ Cs) * np.trace(Cf @ Cf)))
        files = sorted(glob.glob(str(ACT / task / "episode_*" / "step_*" / "suffix_residual.npz")))[::10]
        H = np.concatenate(
            [np.load(f)["all_suffix_residual"][:, LAYER_IDX[layer]].reshape(-1, 1024).astype(np.float64) for f in files]
        )
        Cr = compute_random_conceptor(seed=layer * 100 + int(beta * 10)).astype(np.float64)
        for name, mat in (("PAPER_GLOBAL", C), ("RANDOM_CONTROL", Cr)):
            m = metrics(H, mat, beta)
            rows.append(
                {
                    "task": key,
                    "operator": name,
                    "layer": layer,
                    "alpha": alpha,
                    "beta": beta,
                    "n_token_vectors": len(H),
                    **{k: round(v, 5) for k, v in m.items()},
                    "overlap_s_f": round(overlap, 4) if name == "PAPER_GLOBAL" else "",
                }
            )
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

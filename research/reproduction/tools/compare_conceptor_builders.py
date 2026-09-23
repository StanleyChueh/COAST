"""Phase 1B: offline comparison of repository-faithful vs paper-faithful COAST conceptors.

Research-only analysis script. It does NOT modify or monkeypatch the production code in
``src/openpi/serving/conceptors.py``. The repository functions are imported and called
unchanged for the repository-style variants (V0-V2). The paper-faithful variant (V3) is
implemented independently from arXiv:2605.17144 (Sec. 3.1-3.2, App. A.9.1).

Variants (all on the same Phase 1A activations, same task, same layers/alphas):
    V0  repository behaviour : token flattening + uncentered R + repository AND
    V1  + paper pooling      : token mean-pooling + uncentered R + repository AND
    V2  + paper centering    : token mean-pooling + centered R   + repository AND
    V3  paper-faithful       : token mean-pooling + centered R   + canonical pinv AND

"repository AND" = openpi.serving.conceptors.boolean_and:  A @ inv(A + B - A@B) @ B
"canonical AND"  = paper Eq. (3)/(7):                       pinv(pinv(A) + pinv(B) - I)

V0-V2 mirror compute_task_conceptors() exactly, including its float32 casts of C_success/
C_failure before the Boolean AND. V3 computes in float64 and casts to float32 only for storage.

Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python \
        research/reproduction/tools/compare_conceptor_builders.py
"""

# ruff: noqa: E741, N802, N803, N806, RUF001, RUF002, RUF003
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import resource
import time

import numpy as np

from openpi.serving import conceptors as repo

TASK = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
COLLECT_LAYERS = (0, 5, 11, 17)
LAYERS = (5, 11)
ALPHAS = (0.1, 0.5, 1.0, 2.0, 10.0)
PER_STEP_ALPHA = 1.0  # repository bakes alpha=1.0 into per-step conceptors
PER_STEP_INDICES = tuple(range(10))
D = 1024
VARIANTS = ("V0", "V1", "V2", "V3")
PINV_RCONDS = (1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)  # 1e-15 = numpy 1.26 default
DEFAULT_RCOND = 1e-15


# ─────────────────────────────── data loading ───────────────────────────────


def load_task(task_dir: pathlib.Path) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Return (success_eps, failure_eps, info). Each episode: (T, 10 denoise, 2 layers, 10 tokens, 1024) fp32.

    Only the layer slots for LAYERS are kept. Episode order and success labels follow the
    repository (sorted dirs; success = any(rewards.npz:success_at_step)); metadata.json is cross-checked.
    """
    slots = [COLLECT_LAYERS.index(layer) for layer in LAYERS]
    succ, fail, info = [], [], {"episodes": []}
    for ep in sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")):
        with np.load(ep / "rewards.npz") as r:
            success = bool(np.any(r["success_at_step"]))
        meta = json.loads((ep / "metadata.json").read_text())
        assert success == bool(meta["episode_success"]), f"label mismatch in {ep}"
        steps = sorted(s for s in ep.iterdir() if s.is_dir() and s.name.startswith("step_"))
        arr = np.stack([np.load(s / "suffix_residual.npz")["all_suffix_residual"][:, slots] for s in steps])
        (succ if success else fail).append(arr)
        info["episodes"].append({"episode": ep.name, "success": success, "inference_steps": len(steps)})
    return succ, fail, info


def rows(eps: list[np.ndarray], layer: int, *, pooled: bool, step: int | None) -> np.ndarray:
    """Build the (N, 1024) float64 sample matrix for one class.

    flattened (repository): every (inference call, denoise step, token) is a row.
    pooled    (paper)     : tokens are mean-pooled first -> one row per (inference call, denoise step).
    step=None -> global (all 10 denoise steps); step=t -> only denoise step t.
    """
    li = LAYERS.index(layer)
    out = []
    for h in eps:
        x = h[:, :, li] if step is None else h[:, step : step + 1, li]  # (T, S, 10, 1024)
        x = x.astype(np.float64)
        x = x.mean(axis=2) if pooled else x
        out.append(x.reshape(-1, D))
    return np.concatenate(out, axis=0)


# ─────────────────────────── conceptor constructions ────────────────────────


def centered_R(X: np.ndarray) -> np.ndarray:
    """Paper Eq. (1)/A.9.1: R = X~^T X~ / N with class-specific mean centering."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / X.shape[0]


def paper_conceptor(R: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """C = R (R + alpha^-2 I)^-1 via eigendecomposition (exactly symmetric). Returns (C, min raw eigenvalue of R)."""
    lam, U = np.linalg.eigh(R)
    lam_min = float(lam.min())
    lam = np.clip(lam, 0.0, None)  # R is PSD; clip round-off negatives
    mu = lam / (lam + alpha**-2)
    return (U * mu) @ U.T, lam_min


def paper_and(A: np.ndarray, B: np.ndarray, rcond: float = DEFAULT_RCOND) -> np.ndarray:
    """Paper Eq. (3): A AND B = pinv(pinv(A) + pinv(B) - I) (Moore-Penrose, numpy defaults unless rcond given)."""
    I = np.eye(A.shape[0])
    return np.linalg.pinv(np.linalg.pinv(A, rcond=rcond) + np.linalg.pinv(B, rcond=rcond) - I, rcond=rcond)


def build_variant(variant: str, Xs: np.ndarray, Xf: np.ndarray, alphas, *, keep_R: bool = False) -> dict:
    """Return {alpha: (C_s, C_f, C_c)} as float32 matrices (the storage dtype), plus float64 extras for V3."""
    out = {}
    if variant in ("V0", "V1"):
        Rs, Rf = repo.correlation_matrix(Xs), repo.correlation_matrix(Xf)
    else:
        Rs, Rf = centered_R(Xs), centered_R(Xf)
    for a in alphas:
        if variant in ("V0", "V1", "V2"):
            # mirrors repo.compute_task_conceptors exactly
            Cs = repo.conceptor(Rs, a).astype(np.float32)
            Cf = repo.conceptor(Rf, a).astype(np.float32)
            Cc = repo.contrastive_conceptor(Cs, Cf).astype(np.float32)
            out[a] = {"C_success": Cs, "C_failure": Cf, "C_contrastive": Cc}
        else:
            Cs64, lmin_s = paper_conceptor(Rs, a)
            Cf64, lmin_f = paper_conceptor(Rf, a)
            Cc64 = paper_and(Cs64, np.eye(D) - Cf64)
            out[a] = {
                "C_success": Cs64.astype(np.float32),
                "C_failure": Cf64.astype(np.float32),
                "C_contrastive": Cc64.astype(np.float32),
                "_f64": (Cs64, Cf64, Cc64),
                "_lam_min": (lmin_s, lmin_f),
                # equivalence check: eigh-based conceptor vs repository solve-based formula
                "_eigh_vs_solve": float(
                    max(np.abs(Cs64 - repo.conceptor(Rs, a)).max(), np.abs(Cf64 - repo.conceptor(Rf, a)).max())
                ),
            }
    if keep_R:
        out["_R"] = (Rs, Rf)
    return out


# ──────────────────────────────── diagnostics ───────────────────────────────


def diag(C: np.ndarray) -> dict:
    C = C.astype(np.float64)
    fro = float(np.linalg.norm(C))
    sym_err = float(np.linalg.norm(C - C.T) / fro) if fro > 0 else 0.0
    sv = np.linalg.svd(C, compute_uv=False)
    p = sv / sv.sum() if sv.sum() > 0 else sv
    erank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    d = {
        "shape": list(C.shape),
        "finite": bool(np.isfinite(C).all()),
        "sym_err": sym_err,
        "trace": float(np.trace(C)),
        "quota": float(np.trace(C) / C.shape[0]),
        "fro": fro,
        "erank": erank,
        "n_sv_gt_0.01": int((sv > 0.01).sum()),
        "n_sv_gt_0.5": int((sv > 0.5).sum()),
    }
    if sym_err < 1e-5:
        ev = np.linalg.eigvalsh(0.5 * (C + C.T))
        d.update(eig_min=float(ev.min()), eig_max=float(ev.max()), eig_imag_max=0.0, eig_kind="eigvalsh(sym)")
    else:
        ev = np.linalg.eigvals(C)
        d.update(
            eig_min=float(ev.real.min()),
            eig_max=float(ev.real.max()),
            eig_imag_max=float(np.abs(ev.imag).max()),
            eig_kind="eigvals(general)",
        )
    evs = np.linalg.eigvalsh(0.5 * (C + C.T))
    d.update(symPart_eig_min=float(evs.min()), symPart_eig_max=float(evs.max()))
    return d


def rel_fro(A: np.ndarray, B: np.ndarray) -> float:
    """||A - B||_F / ||B||_F."""
    B64 = B.astype(np.float64)
    return float(np.linalg.norm(A.astype(np.float64) - B64) / np.linalg.norm(B64))


def top_eigvecs(C: np.ndarray, k: int) -> np.ndarray:
    ev, U = np.linalg.eigh(0.5 * (C.astype(np.float64) + C.astype(np.float64).T))
    return U[:, np.argsort(ev)[::-1][:k]]


def subspace_alignment(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """Mean cos^2 of the k principal angles between the top-k eigenspaces (1 = identical, ~k/d = random)."""
    Ua, Ub = top_eigvecs(A, k), top_eigvecs(B, k)
    return float(np.linalg.norm(Ua.T @ Ub) ** 2 / k)


def overlap(Cs: np.ndarray, Cf: np.ndarray) -> float:
    """Paper Eq. (11): tr(Cs Cf) / sqrt(tr(Cs^2) tr(Cf^2))."""
    Cs, Cf = Cs.astype(np.float64), Cf.astype(np.float64)
    return float(np.trace(Cs @ Cf) / np.sqrt(np.trace(Cs @ Cs) * np.trace(Cf @ Cf)))


# ─────────────────────────────────── main ───────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation-root", default="activations/libero/openpi-libero-2000")
    ap.add_argument("--repo-npz", default="conceptors/phase1b_repo_task02.npz")
    ap.add_argument("--out-dir", default="conceptors")
    ap.add_argument("--results-json", default="conceptors/phase1b_results.json")
    ap.add_argument("--metrics-csv", default="research/reproduction/experiments/phase1b_conceptor_metrics.csv")
    args = ap.parse_args()
    t0 = time.time()
    timings = {}

    succ, fail, info = load_task(pathlib.Path(args.activation_root) / TASK)
    timings["load_s"] = time.time() - t0
    res: dict = {"task": TASK, "n_success_eps": len(succ), "n_failure_eps": len(fail), "episodes": info["episodes"]}
    res["inference_steps"] = {
        "success": int(sum(h.shape[0] for h in succ)),
        "failure": int(sum(h.shape[0] for h in fail)),
    }

    repo_npz = np.load(args.repo_npz)
    store = {v: {} for v in ("V1", "V2", "V3")}
    mats: dict = {}  # (variant, layer, strategy_key) -> {kind: fp32 matrix}
    csv_rows = []
    res["sample_counts"], res["centering"], res["V0_repro_check"], res["paper_numerics"] = {}, {}, {}, {}

    def strategy_keys():
        for a in ALPHAS:
            yield f"global_a{a}", None, a
        for t in PER_STEP_INDICES:
            yield f"per_step_{t}_a{PER_STEP_ALPHA}", t, PER_STEP_ALPHA

    t1 = time.time()
    for layer in LAYERS:
        for sk_step in [None, *PER_STEP_INDICES]:
            alphas = ALPHAS if sk_step is None else (PER_STEP_ALPHA,)
            tag = "global" if sk_step is None else f"per_step_{sk_step}"
            Xs_flat, Xf_flat = (
                rows(succ, layer, pooled=False, step=sk_step),
                rows(fail, layer, pooled=False, step=sk_step),
            )
            Xs_pool, Xf_pool = (
                rows(succ, layer, pooled=True, step=sk_step),
                rows(fail, layer, pooled=True, step=sk_step),
            )
            res["sample_counts"][f"L{layer}_{tag}"] = {
                "flattened": {"success": Xs_flat.shape[0], "failure": Xf_flat.shape[0]},
                "pooled": {"success": Xs_pool.shape[0], "failure": Xf_pool.shape[0]},
            }
            if sk_step in (None, 0, 9):
                cstats = {}
                for name, X in (
                    ("success_pooled", Xs_pool),
                    ("failure_pooled", Xf_pool),
                    ("success_flat", Xs_flat),
                    ("failure_flat", Xf_flat),
                ):
                    mu = X.mean(axis=0)
                    tr_unc = float(np.einsum("ij,ij->", X, X) / X.shape[0])
                    cstats[name] = {
                        "mean_norm": float(np.linalg.norm(mu)),
                        "mean_row_norm": float(np.linalg.norm(X, axis=1).mean()),
                        "frac_uncentered_trace_from_mean": float(mu @ mu / tr_unc),
                    }
                cstats["success_minus_failure_mean_norm_pooled"] = float(
                    np.linalg.norm(Xs_pool.mean(0) - Xf_pool.mean(0))
                )
                res["centering"][f"L{layer}_{tag}"] = cstats

            built = {
                "V0": build_variant("V0", Xs_flat, Xf_flat, alphas),
                "V1": build_variant("V1", Xs_pool, Xf_pool, alphas),
                "V2": build_variant("V2", Xs_pool, Xf_pool, alphas),
                "V3": build_variant("V3", Xs_pool, Xf_pool, alphas, keep_R=True),
            }
            del Xs_flat, Xf_flat
            Rs3, Rf3 = built["V3"]["_R"]
            for a in alphas:
                sk = f"global_a{a}" if sk_step is None else f"per_step_{sk_step}_a{a}"
                npz_key = f"{TASK}__L{layer}__{a}" if sk_step is None else f"{TASK}__L{layer}__per_step_{sk_step}"
                # V0 reproduction check against the official NPZ
                res["V0_repro_check"][f"L{layer}_{sk}"] = {
                    kind: float(np.abs(built["V0"][a][kind].astype(np.float64) - repo_npz[f"{npz_key}__{kind}"]).max())
                    for kind in ("C_success", "C_failure", "C_contrastive")
                }
                for v in VARIANTS:
                    mats[(v, layer, sk)] = {k: built[v][a][k] for k in ("C_success", "C_failure", "C_contrastive")}
                for v in ("V1", "V2", "V3"):
                    for kind in ("C_success", "C_failure", "C_contrastive"):
                        store[v][f"{npz_key}__{kind}"] = built[v][a][kind]
                # paper-faithful numerics: float64 symmetry, eigen range, pinv tolerance sensitivity
                Cs64, Cf64, Cc64 = built["V3"][a]["_f64"]
                ev = np.linalg.eigvalsh(0.5 * (Cc64 + Cc64.T))
                sens = {}
                for rc in PINV_RCONDS:
                    Cc_rc = paper_and(Cs64, np.eye(D) - Cf64, rcond=rc)
                    evr = np.linalg.eigvalsh(0.5 * (Cc_rc + Cc_rc.T))
                    sens[f"{rc:g}"] = {
                        "rel_fro_vs_default": rel_fro(Cc_rc, Cc64),
                        "eig_min": float(evr.min()),
                        "eig_max": float(evr.max()),
                        "quota": float(np.trace(Cc_rc) / D),
                    }
                Cc_herm = np.linalg.pinv(
                    np.linalg.pinv(Cs64, hermitian=True) + np.linalg.pinv(np.eye(D) - Cf64, hermitian=True) - np.eye(D),
                    hermitian=True,
                )
                sens["hermitian_default"] = {"rel_fro_vs_default": rel_fro(Cc_herm, Cc64)}
                res["paper_numerics"][f"L{layer}_{sk}"] = {
                    "R_success_rank": int(np.linalg.matrix_rank(Rs3, hermitian=True)),
                    "R_failure_rank": int(np.linalg.matrix_rank(Rf3, hermitian=True)),
                    "R_min_raw_eig": list(built["V3"][a]["_lam_min"]),
                    "C_success_rank_gt1e-12": int((np.linalg.eigvalsh(Cs64) > 1e-12).sum()),
                    "eigh_vs_solve_max_abs": built["V3"][a]["_eigh_vs_solve"],
                    "C_contrastive_f64_sym_err": float(np.linalg.norm(Cc64 - Cc64.T) / np.linalg.norm(Cc64)),
                    "C_contrastive_f64_eig_min": float(ev.min()),
                    "C_contrastive_f64_eig_max": float(ev.max()),
                    "C_contrastive_f64_n_eig_gt_1": int((ev > 1 + 1e-9).sum()),
                    "pinv_sensitivity": sens,
                }
            del built
    timings["build_s"] = time.time() - t1

    # ─── diagnostics & comparisons ───
    t2 = time.time()
    res["diagnostics"], res["comparisons"], res["overlap"] = {}, {}, {}
    for layer in LAYERS:
        for sk, _, _ in strategy_keys():
            for v in VARIANTS:
                m = mats[(v, layer, sk)]
                for kind, C in m.items():
                    dg = diag(C)
                    res["diagnostics"][f"{v}_L{layer}_{sk}_{kind}"] = dg
                    csv_rows.append(
                        {
                            "record": "diagnostic",
                            "variant": v,
                            "layer": layer,
                            "strategy": sk,
                            "matrix": kind,
                            **{
                                k: dg[k]
                                for k in (
                                    "sym_err",
                                    "eig_min",
                                    "eig_max",
                                    "eig_imag_max",
                                    "trace",
                                    "quota",
                                    "fro",
                                    "erank",
                                    "n_sv_gt_0.01",
                                    "n_sv_gt_0.5",
                                )
                            },
                        }
                    )
                res["overlap"][f"{v}_L{layer}_{sk}"] = {
                    "sim_Cs_Cf_eq11": overlap(m["C_success"], m["C_failure"]),
                    "top10_align_Cs_Cf": subspace_alignment(m["C_success"], m["C_failure"], 10),
                }
            pairs = [("V1", "V0"), ("V2", "V1"), ("V3", "V2"), ("V0", "V3")]
            for a_, b_ in pairs:
                for kind in ("C_success", "C_failure", "C_contrastive"):
                    A, B = mats[(a_, layer, sk)][kind], mats[(b_, layer, sk)][kind]
                    rec = {
                        "rel_fro": rel_fro(A, B),
                        "quota_a": float(np.trace(A) / D),
                        "quota_b": float(np.trace(B) / D),
                        "top10_align": subspace_alignment(A, B, 10),
                        "top32_align": subspace_alignment(A, B, 32),
                    }
                    res["comparisons"][f"{a_}_vs_{b_}_L{layer}_{sk}_{kind}"] = rec
                    csv_rows.append(
                        {
                            "record": f"{a_}_vs_{b_}",
                            "variant": f"{a_}|{b_}",
                            "layer": layer,
                            "strategy": sk,
                            "matrix": kind,
                            **rec,
                        }
                    )
    timings["diagnostics_s"] = time.time() - t2

    # ─── offline gate diagnostic (NOT a behavioural result) ───
    t3 = time.time()
    res["gate"] = {}
    sample_eps = {
        "success_ep": info["episodes"].index(next(e for e in info["episodes"] if e["success"])),
        "failure_ep": info["episodes"].index(next(e for e in info["episodes"] if not e["success"])),
    }
    s_h, f_h = succ[0], fail[0]  # first success / failure episode in sorted order
    for layer, a in ((5, 0.5), (11, 0.5), (11, 1.0)):
        li = LAYERS.index(layer)
        H = np.concatenate([s_h[::10, :, li].reshape(-1, D), f_h[::10, :, li].reshape(-1, D)]).astype(np.float64)
        for beta in (0.1, 0.3):
            for v in VARIANTS:
                C = mats[(v, layer, f"global_a{a}")]["C_contrastive"].astype(np.float64)
                M = (1 - beta) * np.eye(D) + beta * C
                r = np.linalg.norm(H @ M.T - H, axis=1) / np.linalg.norm(H, axis=1)
                res["gate"][f"L{layer}_a{a}_b{beta}_{v}"] = {
                    "n_rows": int(H.shape[0]),
                    "mean": float(r.mean()),
                    "median": float(np.median(r)),
                    "max": float(r.max()),
                    "M_transpose_vs_M_rel": float(np.linalg.norm(H @ M.T - H @ M) / np.linalg.norm(H @ M.T - H))
                    if np.linalg.norm(H @ M.T - H) > 0
                    else 0.0,
                }
    res["gate_sample"] = {
        "episodes": [
            info["episodes"][sample_eps["success_ep"]]["episode"],
            info["episodes"][sample_eps["failure_ep"]]["episode"],
        ],
        "rows": "every 10th inference call x 10 denoise steps x 10 tokens (in-sample: these states were used for fitting)",
    }
    timings["gate_s"] = time.time() - t3

    # ─── write outputs ───
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "phase1b_paper_task02.npz", **store["V3"])
    np.savez(out_dir / "phase1b_variant_V1_task02.npz", **store["V1"])
    np.savez(out_dir / "phase1b_variant_V2_task02.npz", **store["V2"])
    timings["total_s"] = time.time() - t0
    res["timings"] = timings
    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    res["numpy_version"] = np.__version__
    pathlib.Path(args.results_json).write_text(json.dumps(res, indent=1))
    keys = sorted(
        {k for r in csv_rows for k in r},
        key=lambda k: (k not in ("record", "variant", "layer", "strategy", "matrix"), k),
    )
    with open(args.metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps({"timings": timings, "peak_rss_gb": res["peak_rss_gb"]}, indent=1))


if __name__ == "__main__":
    main()

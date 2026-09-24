"""Phase 2B spectral validation: real COAST conceptor vs its random matched counterpart.

Builds C_random exactly as the server does for strategy ``random_matched``
(SteeredPolicyWrapper seed derivation -> get_conceptor_matrix -> conceptors.random_matched_conceptor),
then reports:
  1. seed, dimension, NPZ key / sha256;
  2. symmetry error and eigenvalue spectrum of C_real and C_random;
  3. ||eig(C_real) - eig(C_random)||;
  4. orthogonality error of the random basis Q, and that fixing the QR sign ambiguity
     (Q <- Q diag(sign(diag R))) leaves C_random unchanged (column signs cancel in Q L Q^T);
  5. eigenvector similarity (top-k subspace overlap, chance level k/d, and a second seed);
  6. M = (1-beta)I + beta*C for real / random / shrinkage, in fp32 and after the hook's bf16 cast;
  7. descriptive hidden-state check on the Phase 1A layer-5 activations (the conceptor's own training data):
     Rayleigh quotient h^T C h / ||h||^2 and predicted ||h'||/||h||, cos(dh, h) for each M.

Descriptive only; no tuning. Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/validate_phase2b_random_conceptor.py
"""

# ruff: noqa: N802, N803, N806
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import torch

from openpi.serving import conceptors
from openpi.serving import steering

ROOT = pathlib.Path(__file__).resolve().parents[3]
NPZ = ROOT / "conceptors/phase1b_repo_task02.npz"
ACTIVATIONS = ROOT / "activations/libero/openpi-libero-2000"
TASK = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
LAYER, ALPHA, BETA = 5, 0.5, 0.1
TOP_K = (1, 5, 10, 20, 50, 100)
ROLLOUT_STRIDE = 4  # subsample rollout steps of the activation tree (descriptive check only)


def spectrum_summary(eig: np.ndarray) -> dict:
    eig = np.sort(eig)[::-1]
    return {
        "max": float(eig[0]),
        "top10": [round(float(x), 6) for x in eig[:10]],
        "median": float(np.median(eig)),
        "min": float(eig[-1]),
        "trace": float(eig.sum()),
        "trace_over_d": float(eig.mean()),
        "n_gt_0.1": int((eig > 0.1).sum()),
        "n_gt_0.01": int((eig > 0.01).sum()),
        "n_gt_0.001": int((eig > 0.001).sum()),
        "n_negative": int((eig < 0).sum()),
        "all_in_[0,1)_within_1e-6": bool(eig[-1] > -1e-6 and eig[0] < 1.0),
    }


def asymmetry_summary(C: np.ndarray) -> dict:
    Cd = C.astype(np.float64)
    S = 0.5 * (Cd - Cd.T)
    ev = np.linalg.eigvals(Cd)
    ev_sym = np.sort(np.linalg.eigvalsh(0.5 * (Cd + Cd.T)))[::-1]
    re = np.sort(ev.real)[::-1]
    return {
        "skew_part_frob_over_C_frob": float(np.linalg.norm(S) / np.linalg.norm(Cd)),
        "general_eigvals_max_abs_imag": float(np.abs(ev.imag).max()),
        "general_eigvals_real_top5": [round(float(x), 6) for x in re[:5]],
        "general_vs_symmetric_part_eig_max_abs_diff": float(np.abs(re - ev_sym).max()),
    }


def sym_eig(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Cd = C.astype(np.float64)
    w, V = np.linalg.eigh(0.5 * (Cd + Cd.T))
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def subspace_overlap(U: np.ndarray, V: np.ndarray, k: int) -> float:
    """||U_k^T V_k||_F^2 / k: mean squared cosine of the principal angles (1 = same subspace, k/d = chance)."""
    return float(np.linalg.norm(U[:, :k].T @ V[:, :k]) ** 2 / k)


def eigvec_similarity(U: np.ndarray, V: np.ndarray) -> dict:
    d = U.shape[0]
    return {
        "top_k_subspace_overlap": {k: subspace_overlap(U, V, k) for k in TOP_K},
        "chance_k_over_d": {k: k / d for k in TOP_K},
        "abs_cos_leading_eigvec": [round(float(abs(U[:, i] @ V[:, i])), 6) for i in range(5)],
        "max_abs_cos_top1_vs_any_top100": float(np.abs(U[:, 0] @ V[:, :100]).max()),
    }


def frob_cos_offdiag(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius cosine of the trace-free parts (the directional part of C; the isotropic part is shared)."""
    d = A.shape[0]
    a = A - np.trace(A) / d * np.eye(d)
    b = B - np.trace(B) / d * np.eye(d)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_M(C: np.ndarray | None, d: int) -> torch.Tensor:
    """M exactly as ConceptorSteeringHook / ShrinkageSteeringHook build it (fp32, device cpu)."""
    if C is None:
        return steering.ShrinkageSteeringHook(beta=BETA, device="cpu")._build_M(np.zeros((d, d), np.float32))  # noqa: SLF001
    return steering.ConceptorSteeringHook(C, beta=BETA, device="cpu").M


def load_layer5_hiddens() -> tuple[np.ndarray, list[bool]]:
    layer_axis = steering_layer_axis()
    chunks, outcome = [], []
    for ep_dir in sorted((ACTIVATIONS / TASK).iterdir()):
        if not ep_dir.name.startswith("episode_"):
            continue
        outcome.append(conceptors.episode_is_success(ep_dir))
        steps = sorted(d for d in ep_dir.iterdir() if d.name.startswith("step_"))[::ROLLOUT_STRIDE]
        for sd in steps:
            with np.load(sd / "suffix_residual.npz") as z:
                chunks.append(np.asarray(z["all_suffix_residual"])[:, layer_axis].reshape(-1, 1024))
    return np.concatenate(chunks).astype(np.float32), outcome


def steering_layer_axis() -> int:
    return conceptors.DEFAULT_COLLECT_LAYERS.index(LAYER)


def hidden_state_check(H: np.ndarray, Cs: dict[str, np.ndarray | None]) -> dict:
    h32 = torch.from_numpy(H)
    hn = torch.linalg.vector_norm(h32, dim=-1)
    out = {"n_vectors": int(H.shape[0]), "mean_hidden_norm": float(hn.mean())}
    for name, C in Cs.items():
        row = {}
        if C is not None:
            Ct = torch.from_numpy(C)
            rq = ((h32 @ Ct) * h32).sum(-1) / hn**2
            row["rayleigh_mean"] = float(rq.mean())
            row["rayleigh_p5_p95"] = [float(rq.quantile(0.05)), float(rq.quantile(0.95))]
        M = build_M(C, H.shape[1])
        for tag, h, Mx in (
            ("fp32", h32, M),
            ("bf16", h32.to(torch.bfloat16), M.to(torch.bfloat16)),  # the hook casts M to h.dtype (bf16 on GPU)
        ):
            hs = (h @ Mx.T).float()
            hf = h.float()
            delta = hs - hf
            ratio = torch.linalg.vector_norm(hs, dim=-1) / torch.linalg.vector_norm(hf, dim=-1)
            cos = (delta * hf).sum(-1) / (
                torch.linalg.vector_norm(delta, dim=-1) * torch.linalg.vector_norm(hf, dim=-1)
            )
            row[f"{tag}_norm_ratio"] = float(ratio.mean())
            row[f"{tag}_cos_delta_h"] = float(cos.mean())
            row[f"{tag}_rel_delta"] = float((torch.linalg.vector_norm(delta, dim=-1) / hn).mean())
        out[name] = row
    # Paired difference real vs random on identical h (fp32): how far apart the two steered states are.
    Mr, Mx = build_M(Cs["real"], 1024), build_M(Cs["random"], 1024)
    diff = torch.linalg.vector_norm(h32 @ (Mr - Mx).T, dim=-1) / hn
    out["real_vs_random_rel_diff_of_steered_h_fp32"] = float(diff.mean())
    Ms = build_M(None, 1024)
    out["real_vs_shrinkage_rel_diff_of_steered_h_fp32"] = float(
        (torch.linalg.vector_norm(h32 @ (Mr - Ms).T, dim=-1) / hn).mean()
    )
    out["random_vs_shrinkage_rel_diff_of_steered_h_fp32"] = float(
        (torch.linalg.vector_norm(h32 @ (Mx - Ms).T, dim=-1) / hn).mean()
    )
    return out


def main() -> None:
    npz = steering.load_conceptor_npz(NPZ)
    key = f"{TASK}__L{LAYER}__{ALPHA}__C_contrastive"
    C_real = np.asarray(npz[key])
    d = C_real.shape[0]

    # Seed exactly as SteeredPolicyWrapper._get_or_build_hook derives it for random_matched.
    seed_key = (TASK, LAYER, ALPHA, "random_matched")
    seed = steering._stable_random_seed(seed_key)  # noqa: SLF001
    C_rand = steering.get_conceptor_matrix(npz, TASK, LAYER, ALPHA, "random_matched", random_seed=seed)

    # Wrapper path must produce the same hook matrix M.
    wrapper = steering.SteeredPolicyWrapper.__new__(steering.SteeredPolicyWrapper)
    wrapper._npz, wrapper._device, wrapper._hook_cache, wrapper._record_diagnostics = npz, "cpu", {}, False  # noqa: SLF001
    payload = {"task": TASK, "layer": LAYER, "alpha": ALPHA, "beta": BETA, "strategy": "random_matched"}
    _, hook = wrapper._get_or_build_hook(payload)  # noqa: SLF001
    assert torch.equal(hook.M, build_M(C_rand, d)), "wrapper M != (1-beta)I + beta*C_random"

    # Independent re-derivation with the QR sign fix; column signs cancel in Q L Q^T.
    Cd = C_real.astype(np.float64)
    eig_asc = np.linalg.eigvalsh(0.5 * (Cd + Cd.T))
    Q, R = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    Q_fixed = Q * np.sign(np.diag(R))[None, :]
    C_fixed = ((Q_fixed * eig_asc[None, :]) @ Q_fixed.T).astype(np.float32)

    w_real, U = sym_eig(C_real)
    w_rand, V = sym_eig(C_rand)
    C_rand2 = conceptors.random_matched_conceptor(C_real, seed=seed + 1)
    _, V2 = sym_eig(C_rand2)

    summary = {
        "npz": str(NPZ.relative_to(ROOT)),
        "npz_sha256": hashlib.sha256(NPZ.read_bytes()).hexdigest(),
        "key": key,
        "dimension": d,
        "seed_key": list(map(str, seed_key)),
        "seed": seed,
        "generator": "numpy default_rng(seed).standard_normal((d,d)) -> np.linalg.qr -> C = Q diag(eigvalsh(C_sym)) Q^T",
        "wrapper_M_equals_expected": True,
        "symmetry_error": {
            "real_max_abs": float(np.abs(C_real - C_real.T).max()),
            "real_rel_frob": float(np.linalg.norm(C_real - C_real.T) / np.linalg.norm(C_real)),
            "random_max_abs": float(np.abs(C_rand - C_rand.T).max()),
            "random_rel_frob": float(np.linalg.norm(C_rand - C_rand.T) / np.linalg.norm(C_rand)),
        },
        # C_contrastive = C_s AND NOT C_f is not symmetric; random_matched uses the spectrum of its symmetric part,
        # so the skew part S = (C - C^T)/2 is dropped as well (h^T S h = 0: S only adds components orthogonal to h).
        "asymmetry_real": asymmetry_summary(C_real),
        "spectrum_real": spectrum_summary(w_real),
        "spectrum_random": spectrum_summary(w_rand),
        "eig_diff": {
            "l2": float(np.linalg.norm(w_real - w_rand)),
            "max_abs": float(np.abs(w_real - w_rand).max()),
            "rel_l2": float(np.linalg.norm(w_real - w_rand) / np.linalg.norm(w_real)),
        },
        "frobenius_norm": {"real": float(np.linalg.norm(C_real)), "random": float(np.linalg.norm(C_rand))},
        "orthogonality_error_Q": {
            "max_abs_QtQ_minus_I": float(np.abs(Q.T @ Q - np.eye(d)).max()),
            "frob_QtQ_minus_I": float(np.linalg.norm(Q.T @ Q - np.eye(d))),
        },
        "qr_sign_fix": {
            "n_negative_diag_R": int((np.diag(R) < 0).sum()),
            "max_abs_diff_C_signfixed_vs_production": float(np.abs(C_fixed - C_rand).max()),
        },
        "eigvec_similarity_real_vs_random": eigvec_similarity(U, V),
        "eigvec_similarity_real_vs_real": {"top_k_subspace_overlap": {k: subspace_overlap(U, U, k) for k in TOP_K}},
        "eigvec_similarity_random_seed_vs_seed_plus_1": eigvec_similarity(V, V2),
        "frobenius_cosine_trace_free": {
            "real_vs_random": frob_cos_offdiag(C_real, C_rand),
            "real_vs_real": frob_cos_offdiag(C_real, C_real),
            "random_vs_random_seed_plus_1": frob_cos_offdiag(C_rand, C_rand2),
        },
        "diag_of_C": {
            "real_mean_std": [float(np.diag(C_real).mean()), float(np.diag(C_real).std())],
            "random_mean_std": [float(np.diag(C_rand).mean()), float(np.diag(C_rand).std())],
        },
    }

    # M matrices and their bf16 realization (the hook casts M to the activation dtype).
    Ms = {"real": build_M(C_real, d), "random": build_M(C_rand, d), "shrinkage": build_M(None, d)}
    summary["M"] = {}
    for name, M in Ms.items():
        Mb = M.to(torch.bfloat16).float()
        diag = torch.diagonal(Mb)
        vals, counts = torch.unique(diag, return_counts=True)
        top = torch.argsort(counts, descending=True)[:4]
        summary["M"][name] = {
            "fp32_diag_mean": float(torch.diagonal(M).mean()),
            "bf16_diag_mean": float(diag.mean()),
            "bf16_diag_top_values": {f"{float(vals[i]):.7f}": int(counts[i]) for i in top},
            "bf16_offdiag_nonzero": int((Mb - torch.diag(diag)).count_nonzero()),
            "bf16_round_error_frob": float(torch.linalg.norm(Mb - M)),
        }

    H, outcome = load_layer5_hiddens()
    summary["hidden_state_check_phase1a_layer5"] = {
        "source": f"{ACTIVATIONS.relative_to(ROOT)}/{TASK} ({len(outcome)} episodes, {sum(outcome)} success; "
        f"every {ROLLOUT_STRIDE}th rollout step, all 10 denoising steps, 10 tokens)",
        **hidden_state_check(H, {"real": C_real, "random": C_rand, "shrinkage": None}),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

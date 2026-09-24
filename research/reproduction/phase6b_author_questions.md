# Questions for the COAST Authors (draft — not sent)

**Status:** draft for internal review. Nothing has been sent. Accompanies `phase6b_reproduction_summary.md`.

**Context for the recipient:**
- We are reproducing the KS3 result in Table 4 of arXiv:2605.17144 v1 with the released repository (`COAST-VLA/COAST` @ `2afa10e`), π0.5 and `openpi-libero-2000`.
- With the paper's stated configuration (global, L5, α 0.5, β 0.1) we measured 19/30 on 30 test rollouts.
- Following the paper's oracle-selection procedure we measured 21/30, compared with 28/30 reported.
- Our unsteered fit-set rate (8/15) matches the paper's KS3 Base.
- So the released implementation, run under our documented protocol, shows a reproduction gap. We would like to find which protocol details differ. We would be grateful for any clarification.

## Questions (in priority order)

**1. Evaluation states and seeds.**
Could you clarify which LIBERO initial-state indices and seeds were used for KS3, for:
- (a) the 15 fitting rollouts;
- (b) the 30 test rollouts?

We used init states 0–14 (`--seed 0`) and 15–44 (`--seed 15`). Were the runs made before or after the change to `--seed` handling in PR #48 (2026-04-23)?

**2. How the Table 4 value was obtained.**
Could you confirm how the 28/30 for KS3 global was produced? Specifically, which of these applies:
- (a) one configuration selected on the 15 fitting rollouts, then evaluated once on 30 separate test rollouts;
- (b) the best of several evaluation runs of that configuration;
- (c) the best success rate across the configuration grid, measured on the same 30 rollouts that are reported?

**3. Number of runs.**
How many stochastic rollouts or evaluation runs were performed for each configuration during (a) selection on the fitting rollouts and (b) test evaluation? Was the flow-matching sampling noise seeded, and if so, how?

**4. Code version.**
Could you confirm the repository commit or branch that generated Tables 1 and 4?

**5. Checkpoint.**
Could you confirm that the results used `brandonyang/openpi-libero-2000` at revision `aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e`, run through the repository's PyTorch conversion? If possible, could you share a hash of the weights used?

**6. Conceptor construction.**
Were the Table 4 conceptors built with the released `compute_conceptors.py`? That builder uses every token as a row, an uncentered correlation matrix, and AND computed as A·(A + B − AB)⁻¹·B. Or were they built with the construction in App. A.9.1: mean-pooled tokens, centered, and AND = (A⁻¹ + B⁻¹ − I)⁻¹? Would it be possible to make `brandonyang/libero-conceptors` public? It currently returns HTTP 401 to anonymous requests.

**7. Per-step α.**
The released code builds per-step conceptors only at α = 1.0. Could you clarify how the per-step results with α = 10 (KS3) were generated, and whether that code is available?

**8. Base column.**
Could you confirm whether the π0.5 LIBERO "Base" values in Table 1 are the success rates of the 15 fitting rollouts? Or do they come from a separate unsteered evaluation on the 30 test states? (They match the Table 21 fit rates for all ten tasks.)

---

*Optional closing line:* "We are happy to share our scripts, per-episode results and configuration files if that would help. Thank you for releasing the code and checkpoints, which made this reproduction possible."

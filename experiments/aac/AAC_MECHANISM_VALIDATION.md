# AAC mechanism validation

> Update after the Phase A pilot (23,348 AAC calls, 7 tasks × 50 seeds, `results/aac/phaseA_pilot/`). All conclusions below hold at scale:
> - 15 distinct h* (2–16), mean 8.46, 20% h* = 16, 0 non-finite values.
> - The entropy elbow decides h* in **9.5%** of calls; the ξ bound in **90.5%**.
> - Spearman(h*, m(16)) = **−0.91**.
> - H_total rises along the horizon in 81% of calls (mean −11.9 → −9.6 nats).
> - Mean h* by task ranges from 6.0 (PickPlaceCounterToStove) to 13.3 (CoffeeSetupMug).
> - On 922/934 matched-scene first calls, AAC diagnostics are bit-identical across arms. The other 12 differ by ≤ 0.19 nats with the same h*.
> - The N ablation (`results/aac/phaseA_nablation/`) confirms that at N=1 the elbow is pure round-off: h_entropy ∈ {2, 15} in 99% of calls.

Run `results/aac/mechanism/` (gitignored; `manifest.json` pins every SHA). Tasks: OpenDrawer (articulated drawer), PickPlaceCounterToCabinet (pick-and-place), TurnOnElectricKettle (appliance button). 5 paired episodes each (`--seed 0 --seed_per_episode`, scenes 0–4). Arms: `aac` and `fixed16` against two identical `serve.py --aac` servers (N=20, reseed 42, move_th 3.0, 4 denoising steps, one per RTX 4090). Images native 256². COAST 12-D action execution.

Command:
```bash
experiments/aac/serve_aac.sh 0 8000 &   # and: experiments/aac/serve_aac.sh 1 8001 &
RUN=mechanism TASKS="OpenDrawer PickPlaceCounterToCabinet TurnOnElectricKettle" EPISODES=5 \
  ARMS="aac fixed16" PORTS="8000 8001" WORKERS=3 experiments/aac/run_arms.sh
uv run python experiments/aac/analyze_aac.py results/aac/mechanism
```

Every call is logged to `inference.jsonl` with these fields: task, episode, step, inference index, N, H, h*, H_pos/H_rot/H_grip/H_total per future step, Ē_h, Ē_{h+1}−Ē_h, raw argmax, h_entropy, m(l), ξ, candidate spread, sample/select/server latency, and peak VRAM. Per-episode success goes to `episodes.jsonl`. Fixed arms log the h* that AAC *would* have chosen on their own observations.

## Criteria

| Criterion | Result |
|---|---|
| Candidates genuinely differ | yes. Mean candidate std (normalized dims 0–6) is 0.06 at step 1 and rises to 0.09 at step 16. GPU check D: 0/190 identical pairs |
| AAC produces more than one horizon | yes. 15 distinct values (2..16) over 966 AAC calls; mean h* 7.02 |
| h* within legal bounds | yes. All h* ∈ [2, 16]; the client asserts 1 ≤ h ≤ 16 and an empty queue at replan |
| h* follows the upstream rule | yes. Every logged call satisfies h* = max(ξ, max(argmax+1, 2)), which `aac_select` asserts. 144 synthetic cases match the upstream code exactly |
| No NaN/inf entropy | yes (0 non-finite values) |
| No simulator exceptions | yes (0 tracebacks in the 6 task logs) |
| Fixed mode unaffected | yes. Plain inference is bit-identical to `main` (`AAC_BASELINE_IDENTITY.md`). Fixed arms execute exactly h actions per call (client tests) |
| Pairing | 15/15 AAC–fixed16 pairs have identical initial-scene hashes |

**Mechanism status: PASS.**

## What AAC actually does on this checkpoint

1. **Entropy rises along the predicted horizon.** Mean H_total goes from −11.4 nats at step 1 to −9.2 at step 16, and the last step exceeds the first in 86% of calls. Translation entropy rises −3.1 → −2.2, rotation −8.3 → −7.0. Gripper entropy is small (0.03–0.08) because the 20 candidates almost always agree on gripper open/closed.
2. **The rise is front-loaded**, so the elbow argmax_h(Ē_{h+1} − Ē_h) sits at its floor. h_entropy = 2 in 51% of calls and 3 in 40%; it exceeds 4 in under 3%.
3. **ξ decides most choices.** The minimum-motion bound exceeds the entropy elbow in **92%** of AAC calls (89% in fixed16's counterfactual decisions). ξ = 16 (no l reaches α = 3) in 6% of calls. In practice, on this checkpoint and these tasks, AAC behaves as "execute until candidate 0's predicted motion exceeds α = 3, but at least 2–3 steps".
4. The decile means of h* over episode progress lie between 5.6 and 7.9. They are lowest at the very start and end of episodes, but there is no strong phase structure in this small sample.

5. **h* is close to a function of predicted speed.** The Spearman correlation between h* and candidate 0's 16-step motion m(16) is **ρ = −0.90** over 966 calls. Mean h* by m(16) bin: < 3 → 16.0 (n=50); 3–6 → 10.5 (206); 6–10 → 7.4 (247); 10–20 → 4.5 (388); ≥ 20 → 3.3 (75).

Points 3 and 5 are properties of the released algorithm, not of our port: α, the 0.2-weighted gripper term and the ξ = 16 fallback are all upstream values. They mean that chunk sizing here is mostly motion-driven, not entropy-driven. Fast predicted motion gives short chunks, and slow, fine motion gives *long* chunks. That is the reverse of the paper's Fig. 3 narrative (small chunks at the critical manipulation stage). Whether the paper's checkpoint and action scale put ξ in a different regime cannot be checked without that checkpoint (U1). This is flagged in `AAC_REPRO_STATUS.md` as a caveat for using AAC as a baseline.

Plots (`results/aac/mechanism/plots/`): `h_star_hist.png`, `h_star_by_task.png`, `entropy_vs_index.png`, `h_star_vs_progress.png`, `candidate_spread_vs_index.png`, `success_vs_h.png`. Latency vs N is in `AAC_BASELINE_IDENTITY.md`.

## Small-sample outcome (not a performance claim)

| Arm | Success | mean h | calls/ep | server ms/query |
|---|---|---:|---:|---:|
| fixed16 | 5/15 | 16 | 30.1 | 144 |
| AAC | 6/15 | 7.02 | 64.4 | 156 |

Paired: 1 episode where only AAC succeeds, 0 where only fixed16 does. McNemar p = 1.0.

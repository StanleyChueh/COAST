# Pilot preregistration, amendment 5 (2026-10-18, after Gate 0 round 1, before any analysis-seed rollout)

Gate 0 round 1 ran on the calibration block only. That is 10 seeds per task, which are never analysed:
- planners: 320 runs;
- π0.5: 200 rollouts.

No analysis-seed planner or VLA rollout exists. The outcome is **GATE0 FAIL — FIXABLE**; the full report is in the experiment repo at `research/GATE0_REPORT.md`. This amendment only applies remedies that `PILOT_PREREG.md` §4 already fixed. No threshold, estimand, N, R or gate changes.

## 1. Round-1 results that trigger a fixed remedy

| check | result | criterion | fixed remedy applied |
|---|---|---|---|
| G0.8 (P1 = place_cans_plasticbox) | π0.5 V2 SR 20/20 = 1.00 | within [0.10, 0.95] | P1 → **place_burger_fries** (next P1 reserve, `TASK_AUDIT.md` §3; named in amendment 3). Gate 0 is rerun for it |
| G0.5 (C1 = lift_pot) | agreement 26/30 = 0.867 (E2 10/10, V2 16/20) | ≥ 0.95 | the **one** allowed predicate revision (§2); if it fails again, C1 → **grab_roller** |
| G0.3 | identical action-trajectory hash 14/20 = 0.70; same-key **success** agreement 20/20 | hash ≥ 0.90; CRN is void only if success disagrees on > 10% | CRN pairing is **retained** (the fixed rule). The divergence is reported. Same-key replicates are not bitwise reproducible across processes, so `D_same` includes this numerical non-determinism, which makes it conservative |
| G0.4 | TOPP fallback per action: V2 ≤ 0.0005, VCI = 1.00 (the clamped arm's zero-length path takes the fixed 50-step fallback) | V2/VCI difference > 5 pp ⇒ drop VLA physics time | VLA physics time is **dropped entirely**. VLA time = executed policy actions only, which was already the primary measure |
| G0.2 | 1/80 identical (success + T) | ≥ 0.95 | already applied before round 1: every planner condition is run 3×, with majority success and median T |

Passed: G0.1 (80/80), G0.2b (10/10 + 10/10), G0.3b/G0.7 (max clamp deviation 0.0 rad), G0.6 (20/20), G0.9b (20/20 equal observation and input hashes), G0.10 / G0.9 (projected ≈ 11 h on 2× 4090, below the 30 h cap; no reductions).

The G0.8 values of the other gated slots are R1 0.80, R2 0.90, P2 0.80, S1 0.90 and L1 0.15. H1 (0.50) and C1 (0.90) are exempt.

## 2. lift_pot: the one predicate revision (calibration data only)

**Diagnosis.** Calibration seeds were rerun with the official predicate's clauses logged. Every disagreement is a case where:
- the pot is at z > 0.82 and upright;
- one TCP is within 3 cm of its handle contact point;
- the other TCP is 3.5–4.3 cm from its handle, just outside the official 3 cm clause.

Continued under the official termination rule, two of these episodes never satisfied the official predicate within 800 actions, and the pot ended at z ≈ 0.78. All disagreements therefore sit in the two-TCP clause that the object-centric predicate deliberately drops (`TASK_AUDIT.md` §1: that clause is definitional).

**Revision (physically motivated, arm-agnostic, fixed before re-evaluation).** The object-centric predicate becomes "the pot is lifted, upright **and held by the robot**":

    pot z > 0.82  ∧  upright (face product) > 0.8
                  ∧  ∃ arm: |TCP − its handle contact point| < 0.03 m  ∧  that gripper closed (< 0.5)

The new clause excludes a pot that is up but not held. It does not reintroduce the second-arm requirement.

**Re-evaluation.**
- **Data:** the lift_pot calibration block is rerun with the revised predicate as the terminating predicate:
  - E2 on the 10 calibration seeds;
  - π0.5 V2 on the same 10 seeds × masters 1, 2, with the same keys.
- **Criterion:** unchanged (≥ 0.95 agreement).
- **Consequence:** this is the only revision. If agreement is still below 0.95, C1 = **grab_roller**, with the object-centric predicate already fixed in `TASK_AUDIT.md` §2 (roller z > 0.8; the both-closed clause dropped). Gate 0 is then run for it; C1 is exempt from G0.8.

## 3. Replacement-task programs (written from code only, before any of their rollouts)

- **place_burger_fries (P1).**
  - **Layout:** hamburger at x ∈ [−0.30, −0.25] (left), fries at x ∈ [0.20, 0.30] (right), static tray at x = 0.
  - **Official predicate:** hamburger at tray functional point 0, fries at point 1 (xy within 0.08), both grippers open. It is arm-agnostic, since an idle open gripper satisfies it, so it is the primary predicate.
  - **E2:** the official expert (simultaneous grasp + lift, staggered placement, overlapping return-home).
  - **E2ser:** the generic serialization.
  - **E1L / E1R:** the near object first. Each object is grasped, lifted (adaptive height, same helper as the other tasks), placed at its own tray point, and the arm lifts clear. Then back to origin.
  - Feasibility is scene-dependent (the far object is at |x| ≥ 0.2). The IK probe classifies unreachable scenes as for the other tasks.
- **grab_roller (C1, only if §2 fails).**
  - **E1L / E1R:** grasp the roller at the contact point RoboTwin's own selection picks for that arm, then lift by the official displacement (adaptive height).

## 4. What does not change

- The other six tasks keep their calibration data and Gate 0 results. Their analysis seeds are frozen in the manifest.
- Thresholds, N (P1: 30 scenes), R, conditions, gates A–G, analysis code and decision rules are unchanged. The slot→task map is configuration (`rt.PILOT`).
- place_cans_plasticbox and (if replaced) lift_pot calibration data are kept for the record and never analysed.

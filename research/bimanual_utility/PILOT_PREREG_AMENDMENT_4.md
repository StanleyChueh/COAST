# Pilot preregistration, amendment 4 (2026-10-17, before any analysis-set or VLA data)

Amends the definition of `T_crit` in [`PILOT_PREREG_AMENDMENT_1.md`](PILOT_PREREG_AMENDMENT_1.md) §1, Layer A.

## Defect

Amendment 1 defines `T_crit` as the critical path of a DAG with one node per (primitive, arm). Its edges are "same arm → sequential" and "same object → sequential".

When both arms manipulate one object *simultaneously* in the demonstration (lift_pot: both grippers on the pot), this rule serializes them. `T_crit` then **exceeds** the demonstrated time `T_E2`, which is impossible for a bound on achievable time.

Observed in the planner smoke test on two seeds (100000/100001, the first scanned seeds; no analysis-set seed, no VLA output, no gate statistic computed):
- lift_pot: `T_crit` 2676 vs `T_E2` 1488;
- place_dual_shoes: 5741 vs 4022.

## Correction

1. A primitive the demonstrator executed with **both** arms at once is one co-scheduled node spanning both arms, with duration equal to its actual length.
2. Single-arm primitives are nodes of that arm.
3. Edges are unchanged: same arm, and any shared contacted object.

The demonstrated schedule is therefore always a feasible schedule of the DAG, so `T_crit ≤ T_E2` holds by construction (verified on all smoke E2 runs).

`T_crit` remains a non-constructive estimate:
- it is optimistic, because it ignores collisions and reachability;
- it is conservative where two arms place into one container, because the shared object serializes them.

`T_crit` is recomputed from the stored per-segment logs with the corrected function for every E2 record. Nothing else changes.

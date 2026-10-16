# Pilot preregistration, amendment 3 (2026-10-17, before any rollout)

**Task substitution in slot P1: `pick_dual_bottles` → `place_cans_plasticbox`** (the first reserve for P1 in `TASK_AUDIT.md` §3, in the preregistered order).

## Reason

`pick_dual_bottles.check_success` requires **both bottles to be held aloft**: functional-point z > 0.89 at targets (±0.06, −0.105), with nothing to rest on at that height. The expert ends holding them with `is_open=False`.

A single parallel gripper cannot hold two bottles at once, so every P1 scene is two-arm-**necessary** by its goal state. That makes P1 unusable for the efficiency/parallelism slot:
- C2 is defined on certified single-arm-FEASIBLE scenes (gate B);
- gate E needs C2 direction-consistent in both P1 and P2;
- with P1 having no FEASIBLE scenes by construction, C2 could never pass, independent of any VLA behaviour.

**The error was in our task audit** (`TASK_AUDIT.md` §2 listed P1 as "likely yes if one arm reaches the far bottle"). It was found while writing the one-arm expert programs, from code only. No rollout, planner run or VLA output of any task exists.

## Replacement

`place_cans_plasticbox`:
- **Layout:** cans at x ∈ [−0.25, −0.15] and [0.15, 0.25], box at x = 0.
- **Expert:** simultaneous grasp + lift, then staggered placement with overlapping return-home.
- **Success:** both cans within 0.04 m of a box functional point, both grippers open. This is arm-agnostic, and an idle open gripper satisfies it.
- **One arm** must reach the far-side can, so feasibility is scene-dependent (category 2 + 5, as P1 was meant to be).
- **Operating range:** the π0.5 card has no public success rate for this task, so it is checked in Gate 0 G0.8 like every slot. If it fails G0.8, the next reserve is `place_burger_fries` (`TASK_AUDIT.md` §3).

`pick_dual_bottles` is not run in the pilot.

Nothing else changes: thresholds, N, R, other tasks and gates A–G.

"""Static audit of RoboTwin 2.0 scripted experts: how many arms each expert uses, and how.

Reads `envs/<task>.py` from a RoboTwin checkout (never imports or executes it) and reports, per task:

- arm rule: how the expert chooses arms (`side` = one arm chosen by object x-position,
  `side+opposite` = side arm plus the other arm, `fixed-both` = left and right hard-coded,
  `fixed-one` = a single hard-coded arm, `attr` = arm stored on `self.arm_tag` in setup);
- arms used: 1 or 2 (from ArmTag literals, `.opposite`, and helper functions);
- moves / simultaneous moves: `self.move(...)` calls, and those with >= 2 action arguments
  (both arms planned and executed together);
- success predicate: whether `check_success` references both grippers or both TCPs
  (a predicate that *defines* success as two-armed);
- eval step limit (`_eval_step_limit.yml`).

    python research/bimanual_utility/scripts/audit_robotwin_experts.py /path/to/RoboTwin --csv out.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import pathlib
import re

SKIP = {"__init__.py", "model_utils.py"}
ARM_HELPERS = ("pick_and_place_block", "move_bowl", "remove_bread", "remove")


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _move_calls(fn: ast.FunctionDef) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "move"]


def audit_task(path: pathlib.Path, step_limits: dict[str, int]) -> dict:
    src = path.read_text()
    fns = _functions(ast.parse(src))
    task = path.stem
    play = fns.get("play_once")
    if play is None:
        raise ValueError(f"{task}: no play_once")
    # Expert body plus the helpers it calls for per-object pick/place.
    bodies = [play] + [fns[h] for h in ARM_HELPERS if h in fns]
    text = "\n".join(ast.get_source_segment(src, b) or "" for b in bodies)
    moves = [m for b in bodies for m in _move_calls(b)]
    n_sim = sum(1 for m in moves if len(m.args) >= 2)

    side = bool(re.search(r"ArmTag\(\s*[\"'](left|right)[\"']\s+if\b", text))
    # `.opposite` only inside back_to_origin(...) means "send the previous arm home", not a second working arm.
    opposite = any(".opposite" in line and "back_to_origin" not in line for line in text.splitlines())
    lit_left = bool(re.search(r"ArmTag\(\s*[\"']left[\"']\s*\)", text))
    lit_right = bool(re.search(r"ArmTag\(\s*[\"']right[\"']\s*\)", text))
    attr = "self.arm_tag" in text and not side
    if side and opposite:
        rule = "side+opposite"
    elif side and (lit_left or lit_right):
        rule = "side+fixed"
    elif side:
        rule = "side"
    elif attr and opposite:
        rule = "attr+opposite"
    elif attr:
        rule = "attr"
    elif lit_left and lit_right:
        rule = "fixed-both"
    elif lit_left or lit_right:
        rule = "fixed-one"
    else:
        rule = "other"
    # Arms that move in a given scene. `side` experts that switch arm between objects
    # (multi-object helpers) can use both, depending on object positions.
    multi_object = any(h in fns for h in ARM_HELPERS[:2]) or "for i in range" in text
    if rule in ("side+opposite", "attr+opposite", "fixed-both"):
        arms = "2"
    elif rule in ("side", "side+fixed") and (multi_object or n_sim > 0 or rule == "side+fixed"):
        arms = "1-2 (scene)"
    else:
        arms = "1"

    succ = ast.get_source_segment(src, fns["check_success"]) if "check_success" in fns else ""
    # Lines that pick one arm with a conditional (`a if arm == "left" else b`) are arm-specific, not two-armed.
    plain = [line for line in succ.splitlines() if " if " not in line]
    plain_text = "\n".join(plain)
    two_arm_defined = ("get_left_tcp" in plain_text and "get_right_tcp" in plain_text) or (
        "is_left_gripper_close" in plain_text and "is_right_gripper_close" in plain_text
    )
    both_open = "is_left_gripper_open" in plain_text and "is_right_gripper_open" in plain_text
    arm_specific = bool(re.search(r"arm_tag|handover_arm|grasp_arm", succ))
    return {
        "task": task,
        "arm_rule": rule,
        "arms_used": arms,
        "moves": len(moves),
        "simultaneous_moves": n_sim,
        "success_defines_two_arms": two_arm_defined,
        "success_needs_both_open": both_open,
        "success_arm_specific": arm_specific,
        "eval_step_limit": step_limits.get(task, ""),
    }


def _step_limits(root: pathlib.Path) -> dict[str, int]:
    for cand in root.rglob("_eval_step_limit.yml"):
        out = {}
        for line in cand.read_text().splitlines():
            if ":" in line and not line.strip().startswith("#"):
                k, v = line.split(":", 1)
                if v.strip().isdigit():
                    out[k.strip()] = int(v)
        return out
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("robotwin_root", type=pathlib.Path)
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    args = ap.parse_args()
    envs = args.robotwin_root / "envs"
    limits = _step_limits(args.robotwin_root)
    rows = [
        audit_task(p, limits) for p in sorted(envs.glob("*.py")) if not p.name.startswith("_") and p.name not in SKIP
    ]
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["arm_rule"]] = counts.get(r["arm_rule"], 0) + 1
    print(f"\n{len(rows)} tasks; arm rules: {counts}")
    print(f"tasks with >=1 simultaneous move: {sum(r['simultaneous_moves'] > 0 for r in rows)}")
    arms: dict[str, int] = {}
    for r in rows:
        arms[r["arms_used"]] = arms.get(r["arms_used"], 0) + 1
    print(f"arms used by the expert: {arms}")
    print(f"success predicate defines two arms: {[r['task'] for r in rows if r['success_defines_two_arms']]}")
    print(f"success predicate is arm-specific: {[r['task'] for r in rows if r['success_arm_specific']]}")
    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

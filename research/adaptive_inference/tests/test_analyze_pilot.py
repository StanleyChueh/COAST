"""Synthetic checks of the preregistered pilot estimators (no GPU, no simulator).

uv run pytest research/adaptive_inference/tests -q
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import analyze_pilot as ap


def _synthetic(kind: str, n_states: int = 400, n_rep: int = 4, seed: int = 0):
    """Per-state success probabilities for every arm, then Bernoulli draws per replicate."""
    rng = np.random.default_rng(seed)
    states, y, task, unit = [], {}, {}, {}
    for i in range(n_states):
        g = f"{i % 4}_{i // 40}_{i}"
        t = i % 4
        base = rng.uniform(0.1, 0.9)
        p = dict.fromkeys(ap.ALL_ARMS, base)
        if kind == "state_specific":
            # every state has one arm that is much better, varying within task
            p[("IL", "IC", "IH")[i % 3]] = 0.95
            p["I0"] = p["I0P"] = 0.05
        elif kind == "task_specific":
            p[("IL", "IC", "IH", "IL")[t]] = 0.95
            p["I0"] = p["I0P"] = 0.05
        y[g] = {a: (rng.random(n_rep) < p[a]).astype(int) for a in ap.ALL_ARMS}
        states.append(g)
        task[g] = t
        unit[g] = (t, i // 40)
    return states, y, task, unit


def test_null_has_no_loo_gain_but_naive_oracle_is_inflated():
    states, y, task, _ = _synthetic("null")
    s_on = ap.always_on(states, y)
    b = max(s_on[a] for a in ap.ROUTE)
    o_state, _ = ap.loo_state_oracle(states, y)
    assert o_state - b < 0.02  # no real heterogeneity -> no honest gain
    assert ap.naive_oracle(states, y) - b > 0.15  # single-draw "any arm succeeds" harvests noise


def test_state_specific_heterogeneity_is_recovered_and_beats_task_router():
    states, y, task, _ = _synthetic("state_specific")
    s_on = ap.always_on(states, y)
    b = max(s_on[a] for a in ap.ROUTE)
    o_state, _ = ap.loo_state_oracle(states, y)
    o_task = ap.loo_task_oracle(states, y, task)
    assert o_state - b > 0.2
    assert o_state - o_task > 0.2
    rs = ap.rescue_stats(states, y, ap.INTERV)
    # other arms still succeed by chance at their base rate, so unique rescues are ~10% here
    assert all(rs["unique_rate"][a] > 0.05 for a in ap.INTERV)


def test_task_specific_heterogeneity_is_explained_by_task_router():
    states, y, task, _ = _synthetic("task_specific")
    o_state, _ = ap.loo_state_oracle(states, y)
    o_task = ap.loo_task_oracle(states, y, task)
    # no state-level structure beyond task: the 3-draw state choice can only lose to the
    # pooled task choice (selection noise), which makes gate D conservative
    assert o_state - o_task < 0.01


def test_choose_breaks_ties_by_cost_order():
    assert ap._choose({"I0": 0.5, "IL": 0.5, "IC": 0.5, "IH": 0.5}, ap.ROUTE) == "I0"  # noqa: SLF001
    assert ap._choose({"I0": 0.2, "IL": 0.5, "IC": 0.5, "IH": 0.1}, ap.ROUTE) == "IL"  # noqa: SLF001


def test_risk_gate_equals_base_when_base_always_succeeds():
    states, y, _, _ = _synthetic("null")
    for g in states:
        y[g]["I0"] = np.ones_like(y[g]["I0"])
    assert ap.loo_risk_gate(states, y) == pytest.approx(1.0)

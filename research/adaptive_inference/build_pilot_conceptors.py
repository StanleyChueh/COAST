"""Build the IL (COAST global) conceptors for the pilot tasks with the released builder.

The released builder (`openpi.serving.conceptors.compute_all_conceptors`, used by
experiments/libero/compute_conceptors.py) is called unchanged. The only adaptation is
the success label: the authors' activation dataset
(brandonyang/pi05-libero-activations-v1-2000-15env) stores it in each episode's
metadata.json (`episode_success`) and ships `rewards.npz` for only a few episodes. The
label reader prefers rewards.npz and otherwise uses metadata.json, and asserts the two
agree wherever both exist.

    CUDA_VISIBLE_DEVICES="" uv run python research/adaptive_inference/build_pilot_conceptors.py
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
from openpi.serving import conceptors as _c
import tyro


def episode_is_success(ep_dir: pathlib.Path) -> bool:
    meta = json.loads((ep_dir / "metadata.json").read_text())["episode_success"]
    rewards = ep_dir / "rewards.npz"
    if rewards.exists():
        with np.load(rewards) as data:
            r = bool(np.any(data["success_at_step"]))
        if r != bool(meta):
            raise ValueError(f"label mismatch rewards.npz={r} metadata={meta} in {ep_dir}")
        return r
    return bool(meta)


@dataclasses.dataclass
class Args:
    activation_root: pathlib.Path = pathlib.Path("activations/pi05-libero-activations-v1-2000-15env")
    output_path: pathlib.Path = pathlib.Path("conceptors/aiv_pilot_conceptors.npz")
    layers: tuple[int, ...] = (5, 11)
    alphas: tuple[float, ...] = (0.5, 1.0)


def main(args: Args) -> None:
    _c.episode_is_success = episode_is_success  # the builder looks this name up at call time
    counts = {}
    for ep in _c.iter_episode_dirs(args.activation_root):
        t = _c.episode_task_name(ep)
        counts.setdefault(t, [0, 0])[0 if episode_is_success(ep) else 1] += 1
    print("fit success/failure counts:", json.dumps(counts, indent=1))
    summary = _c.compute_all_conceptors(
        activation_root=args.activation_root,
        output_path=args.output_path,
        layers=args.layers,
        alphas=args.alphas,
        per_step_indices=(),
        collect_layers=_c.DEFAULT_COLLECT_LAYERS,
        min_episodes_per_class=2,
        task_filter=None,
    )
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main(tyro.cli(Args))

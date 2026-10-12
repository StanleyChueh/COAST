"""Write results/aac/<run>/manifest.json: everything needed to reproduce a run.

Run from examples/robocasa_env/ (it needs openpi_client to query the server):

    cd examples/robocasa_env
    uv run python ../../experiments/aac/write_manifest.py \
        --run_dir ../../results/aac/<run> --port 8000 --port 8001 -- <free-form command string>
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]


def _git(*args, cwd=REPO):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _worktree_hash():
    h = hashlib.sha256(_git("diff", "HEAD").encode())
    for rel in sorted(_git("ls-files", "--others", "--exclude-standard").splitlines()):
        path = REPO / rel
        if path.is_file():
            h.update(rel.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=pathlib.Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, action="append", default=[])
    ap.add_argument("--config", type=pathlib.Path, help="JSON file with the run config (arms, tasks, seeds, ...)")
    ap.add_argument("command", nargs="*")
    args = ap.parse_args()

    servers = {}
    for port in args.port:
        try:
            from openpi_client import websocket_client_policy

            servers[str(port)] = websocket_client_policy.WebsocketClientPolicy(args.host, port).get_server_metadata()
        except Exception as exc:
            servers[str(port)] = {"error": repr(exc)}

    ckpt = None
    for md in servers.values():
        if isinstance(md, dict) and "model_path" in md:
            ckpt = md["model_path"]

    manifest = {
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "coast_git_sha": _git("rev-parse", "HEAD"),
        "coast_git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "coast_git_dirty": _git("status", "--porcelain") != "",
        "coast_git_diff_stat": _git("diff", "--stat"),
        # Uncommitted code is part of the run's identity: hash tracked diffs + untracked AAC files.
        "coast_worktree_sha256": _worktree_hash(),
        "submodules": _git("submodule", "status"),
        "upstream": {
            "aac_robocasa": {
                "url": "https://github.com/Adaptive-Action-Chunking/robocasa",
                "sha": _git("rev-parse", "HEAD", cwd=REPO / "external/aac/aac-robocasa"),
            },
            "aac_gr00t_multi_sample": {
                "url": "https://github.com/Adaptive-Action-Chunking/gr00t-multi-sample",
                "sha": _git("rev-parse", "HEAD", cwd=REPO / "external/aac/gr00t-multi-sample"),
            },
        },
        "checkpoint": ckpt,
        "servers": servers,
        "client_env": {
            "python": platform.python_version(),
            "robocasa": _version("robocasa"),
            "robosuite": _version("robosuite"),
            "mujoco": _version("mujoco"),
            "numpy": _version("numpy"),
            "openpi-client": _version("openpi-client"),
        },
        "host": platform.node(),
        "command": " ".join(args.command),
        "config": json.loads(args.config.read_text()) if args.config else None,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

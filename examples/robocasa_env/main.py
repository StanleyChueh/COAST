"""
Evaluate a single RoboCasa task using a policy server.

For evaluating one environment:
    MUJOCO_GL=egl uv run examples/robocasa_env/main.py --env_name CloseBlenderLid

To evaluate every environment in a task set (atomic_seen, composite_seen,
composite_unseen, pretrain50, ...) instead, use eval_all.py:
    MUJOCO_GL=egl uv run examples/robocasa_env/eval_all.py --task_set atomic_seen

Note: RoboCasa does NOT support parallel envs (gym.vector.AsyncVectorEnv) because
each MuJoCo env requires its own EGL/OpenGL context, which is not multiprocess-safe.
The grid in the video is built from the multiple cameras of a single env, not from
multiple parallel envs as in the MetaWorld example.
"""

import collections
import dataclasses
import hashlib
import json
import logging
import math
import os
import time
import uuid
from typing import Optional

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import robocasa  # noqa: F401
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.collection_session import CollectionSession
from openpi_client.steering import STEERING_KEY, build_steering_payload
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from tqdm import tqdm

logger = logging.getLogger(__name__)

EXECUTION_MODES = ("fixed", "aac")

_MUTUALLY_EXCLUSIVE_MODE_ERROR = (
    "--collect and --steer are mutually exclusive; run activation collection and "
    "steering evaluation as separate passes."
)

# RoboCasa's gym wrapper hardcodes these three camera observations on every step.
# Keys are short names used in CLI args; values are the obs dict keys.
CAMERA_KEYS = {
    "agentview_left": "video.robot0_agentview_left",
    "agentview_right": "video.robot0_agentview_right",
    "eye_in_hand": "video.robot0_eye_in_hand",
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # RoboCasa task name (e.g., "CloseBlenderLid", "OpenCabinet", "TurnOnMicrowave").
    env_name: str = "CloseBlenderLid"
    # Dataset split: "pretrain" (in-distribution object instances) or "target" (held-out).
    split: str = "pretrain"
    # Number of episodes to run.
    num_episodes: int = 1
    # Index of the first episode (episodes run start_episode .. start_episode+num_episodes-1).
    # With --seed_per_episode the scene of episode k depends only on k, so a crashed run
    # can be resumed exactly; see experiments/aac/run_arms.sh.
    start_episode: int = 0
    # Override the maximum steps per episode. If None, uses 1.5 * task horizon.
    max_steps: int | None = None
    # Number of steps to execute from the model's action plan before re-planning.
    # Used when --execution_mode fixed (the default).
    replan_steps: int = 5
    # How many predicted actions to execute per policy call:
    #   fixed: exactly --replan_steps (COAST default behaviour).
    #   aac:   the server's per-call "aac_horizon" (requires `groot_env/serve.py --aac`).
    execution_mode: str = "fixed"
    # If True, reset episode k with seed (--seed + k) instead of drawing scenes
    # from one env-level RNG stream. Upstream AAC builds each trial's env with
    # seed=trial_id; this makes episode k's scene independent of what happened
    # in earlier episodes, so arms can be paired episode-by-episode.
    seed_per_episode: bool = False
    # Write per-episode video files.
    save_video: bool = True
    # If set, append machine-readable logs to <log_dir>/<env_name>/{episodes,inference}.jsonl.
    log_dir: Optional[str] = None

    # Image resize size for the policy input.
    resize_size: int = 224

    # Cameras to tile into the video output.
    render_cameras: list[str] = dataclasses.field(
        default_factory=lambda: ["agentview_left", "agentview_right", "eye_in_hand"]
    )

    fps: int = 24
    seed: int = 7

    # If True, attach activation-collection metadata to every infer call so the
    # server (started with --collect_activations) saves intermediates to its disk.
    collect: bool = False
    # The `env_id` tag attached to every `__collect__` payload — feeds into the
    # server's output path `episode_{episode_id:03d}_env_{env_id:03d}/`. Leave
    # at the default (0) in the common case. You only need to set this for a
    # specific use case:
    #
    # **When you DO NOT need this flag** (almost always):
    #   • Single main.py invocation (any --num_episodes value) — episode_id
    #     varies per rollout, env_id=0 is fine.
    #   • `eval_all.py` — dispatches one subprocess per env_name. The server
    #     keys the path on task_name, so env_id=0 never collides across
    #     subprocesses. This is the documented, "safe" parallel collection
    #     pattern.
    #
    # **When you DO need this flag** (rare, opt-in):
    #   • You're running N parallel main.py subprocesses on the *same*
    #     env_name (e.g., to collect more rollouts of one task for a per-task
    #     conceptor NPZ faster than running --num_episodes N sequentially).
    #     Then all N share `(task_name, episode_id=0)`, so you must pass a
    #     distinct `--collect_env_id` per subprocess (0, 1, 2, ...) to avoid
    #     them clobbering each other's `episode_000_env_000/` files.
    collect_env_id: int = 0

    # Override the top-level output directory (for videos / artifacts). If
    # None, defaults to ``output/`` — ``eval_task`` nests a per-env subdir, so
    # per-episode videos land at ``output/{env_name}/episode_NNN.mp4``.
    output_dir: Optional[str] = None

    # ── Steering (requires server started with --steer). ──────────────────────
    # When True, attach obs[STEERING_KEY] = {task, layer, alpha, beta, strategy}
    # to every inference call so the server applies a conceptor steering hook.
    steer: bool = False
    steering_layer: int = 11
    steering_alpha: float = 0.1
    steering_beta: float = 0.3
    steering_strategy: str = "global"
    # Override the conceptor task key (default: args.env_name).
    steering_task: Optional[str] = None


def tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange N frames into a grid image.

    Grid layout: cols = ceil(sqrt(N)), rows = ceil(N / cols).
    Empty slots are filled with black.
    """
    n = len(frames)
    h, w, c = frames[0].shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    grid = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        r, col = divmod(idx, cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame

    return grid


def make_env(env_name: str, split: str, seed: int) -> gym.Env:
    return gym.make(f"robocasa/{env_name}", split=split, seed=seed)


def build_state(obs: dict) -> np.ndarray:
    """Concatenate the proprioceptive state into the 16-dim vector expected by the policy."""
    return np.concatenate(
        [
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ],
        axis=0,
    )


def _scene_hash(env, obs: dict) -> str:
    """Hash of the initial MuJoCo state (falls back to the proprio state) for
    checking that two runs saw the same initial scene."""
    try:
        state = np.asarray(
            env.unwrapped.env.sim.get_state().flatten(), dtype=np.float64
        )
    except Exception:  # noqa: BLE001
        state = build_state(obs).astype(np.float64)
    return hashlib.sha1(np.round(state, 6).tobytes()).hexdigest()[:16]


def _model_hash(env) -> str | None:
    """Hash of static scene geometry (body positions, geom sizes). Complements
    _scene_hash, which only sees the simulation state."""
    try:
        model = env.unwrapped.env.sim.model
        geo = np.concatenate(
            [np.asarray(model.body_pos).ravel(), np.asarray(model.geom_size).ravel()]
        )
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha1(np.round(geo.astype(np.float64), 6).tobytes()).hexdigest()[:16]


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _execution_horizon(args, result: dict, chunk_len: int) -> int:
    """Number of actions of this chunk to queue, validated against the chunk."""
    if args.execution_mode == "fixed":
        h = int(args.replan_steps)
    elif args.execution_mode == "aac":
        if "aac_horizon" not in result:
            raise RuntimeError(
                "--execution_mode aac but the server response has no 'aac_horizon'; "
                "start groot_env/serve.py with --aac"
            )
        h = int(result["aac_horizon"])
    else:
        raise ValueError(
            f"Unknown execution_mode {args.execution_mode!r}; expected one of {EXECUTION_MODES}"
        )
    if not 1 <= h <= chunk_len:
        raise RuntimeError(
            f"Execution horizon {h} outside [1, {chunk_len}] ({args.execution_mode} mode)"
        )
    return h


def eval_task(
    env_name: str,
    policy: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    output_dir: str,
    collect_session: CollectionSession | None = None,
) -> dict[str, float]:
    """Evaluate a single env over args.num_episodes episodes and return per-task stats.

    The args object only needs to provide: split, num_episodes, max_steps,
    replan_steps, resize_size, render_cameras, fps. Both main.py's Args and
    eval_all.py's Args satisfy this.

    If `collect_session` is provided, every inference call has __collect__
    metadata attached and the episode boundaries are reported to the session,
    so a server started with --collect_activations writes activations to disk.
    """
    task_output_dir = os.path.join(output_dir, env_name)
    os.makedirs(task_output_dir, exist_ok=True)

    env = make_env(env_name=env_name, split=args.split, seed=args.seed)
    task_horizon = get_task_horizon(env_name)
    max_steps = (
        args.max_steps if args.max_steps is not None else int(task_horizon * 1.5)
    )

    log_dir = getattr(args, "log_dir", None)
    if log_dir is not None:
        task_log_dir = os.path.join(log_dir, env_name)
        os.makedirs(task_log_dir, exist_ok=True)
        episodes_log = open(os.path.join(task_log_dir, "episodes.jsonl"), "a")
        inference_log = open(os.path.join(task_log_dir, "inference.jsonl"), "a")
    else:
        episodes_log = inference_log = None
    execution_mode = getattr(args, "execution_mode", "fixed")
    # Identifies this client process in the logs: an episode interrupted by a crash
    # leaves inference records whose (episode, run_id) never appears in episodes.jsonl.
    run_id = uuid.uuid4().hex[:12]
    save_video = getattr(args, "save_video", True)

    successes: list[bool] = []
    start_episode = getattr(args, "start_episode", 0)
    for episode in range(start_episode, start_episode + args.num_episodes):
        episode_seed = (
            args.seed + episode if getattr(args, "seed_per_episode", False) else None
        )
        obs, info = env.reset(seed=episode_seed)
        scene_hash = _scene_hash(env, obs)
        model_hash = _model_hash(env)
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()
        success = False
        inference_index = 0
        executed_horizons: list[int] = []
        policy_ms: list[float] = []
        episode_t0 = time.perf_counter()
        step = -1

        if collect_session is not None:
            collect_session.start_episode(
                task_name=env_name,
                task_id=0,
                episode_id=episode,
                prompt=str(task_lang),
                env_id=args.collect_env_id,
            )

        video_path = os.path.join(task_output_dir, f"episode_{episode:03d}.mp4")
        video_ctx = (
            iio.imopen(video_path, "w", plugin="pyav") if save_video else _NullVideo()
        )
        with video_ctx as video:
            video.init_video_stream("h264", fps=args.fps)

            pbar = tqdm(
                range(max_steps),
                desc=f"[{env_name}] Episode {episode + 1}/{args.num_episodes}",
                leave=False,
            )
            for step in pbar:
                # Tile multiple cameras of the single env into a grid frame.
                if save_video:
                    frames = [
                        image_tools.convert_to_uint8(obs[CAMERA_KEYS[cam]])
                        for cam in args.render_cameras
                    ]
                    video.write_frame(tile_frames(frames))

                if not action_plan:
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            obs[CAMERA_KEYS["agentview_left"]],
                            args.resize_size,
                            args.resize_size,
                        )
                    )
                    # Second side-view camera. pi0's RobocasaInputs only reads
                    # `observation/image` and `observation/wrist_image`, so this
                    # extra key is silently ignored on the pi05 server. GR00T
                    # N1.5's robocasa head was trained with TWO distinct side
                    # views (agentview_left + agentview_right); the GR00T
                    # adapter picks this up and feeds it into
                    # `video.robot0_agentview_right`. Falls back to duplicating
                    # `observation/image` if the key is missing (older clients).
                    img2 = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            obs[CAMERA_KEYS["agentview_right"]],
                            args.resize_size,
                            args.resize_size,
                        )
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            obs[CAMERA_KEYS["eye_in_hand"]],
                            args.resize_size,
                            args.resize_size,
                        )
                    )
                    element = {
                        "observation/image": img,
                        "observation/image2": img2,
                        "observation/wrist_image": wrist_img,
                        "observation/state": build_state(obs),
                        "prompt": task_lang,
                    }
                    if collect_session is not None:
                        element["__collect__"] = collect_session.make_collect_metadata(
                            step
                        )
                    if args.steer:
                        element[STEERING_KEY] = build_steering_payload(
                            task=args.steering_task or env_name,
                            layer=args.steering_layer,
                            alpha=args.steering_alpha,
                            beta=args.steering_beta,
                            strategy=args.steering_strategy,
                        )
                    t_call = time.perf_counter()
                    result = policy.infer(element)
                    roundtrip_ms = (time.perf_counter() - t_call) * 1000.0
                    action_chunk = result["actions"]  # (action_horizon, action_dim=12)
                    assert action_chunk.ndim == 2, (
                        f"Model output must have shape (action_horizon, action_dim), but got {action_chunk.shape}"
                    )
                    exec_h = _execution_horizon(args, result, action_chunk.shape[0])
                    assert not action_plan, (
                        "stale actions left in the plan at replan time"
                    )
                    for t in range(exec_h):
                        action_plan.append(action_chunk[t])
                    executed_horizons.append(exec_h)
                    policy_ms.append(roundtrip_ms)
                    if inference_log is not None:
                        record = {
                            "task": env_name,
                            "episode": episode,
                            "run_id": run_id,
                            "step": step,
                            "inference_index": inference_index,
                            "execution_mode": execution_mode,
                            "executed_h": exec_h,
                            "chunk_len": int(action_chunk.shape[0]),
                            "roundtrip_ms": roundtrip_ms,
                            "server_timing": _jsonable(result.get("server_timing", {})),
                        }
                        if "aac_horizon" in result:
                            record["aac_horizon"] = int(result["aac_horizon"])
                            record["aac"] = _jsonable(result.get("aac", {}))
                        inference_log.write(json.dumps(record) + "\n")
                        inference_log.flush()
                    inference_index += 1

                action = convert_action(action_plan.popleft())
                obs, reward, terminated, truncated, info = env.step(action)
                success = bool(info.get("success", False))

                if collect_session is not None:
                    collect_session.record_step(step, float(reward), done=success)

                pbar.set_postfix(success=str(success))

                if success:
                    break

        if collect_session is not None:
            collect_session.finalize_episode()

        if episodes_log is not None:
            episodes_log.write(
                json.dumps(
                    {
                        "task": env_name,
                        "split": args.split,
                        "episode": episode,
                        "run_id": run_id,
                        "seed": args.seed,
                        "episode_seed": episode_seed,
                        "scene_hash": scene_hash,
                        "model_hash": model_hash,
                        "prompt": str(task_lang),
                        "execution_mode": execution_mode,
                        "replan_steps": int(args.replan_steps)
                        if execution_mode == "fixed"
                        else None,
                        "success": bool(success),
                        "env_steps": step + 1,
                        "max_steps": max_steps,
                        "num_policy_calls": len(executed_horizons),
                        "executed_horizons": executed_horizons,
                        "mean_executed_h": float(np.mean(executed_horizons))
                        if executed_horizons
                        else None,
                        "policy_roundtrip_ms_total": float(np.sum(policy_ms)),
                        "wall_s": time.perf_counter() - episode_t0,
                    }
                )
                + "\n"
            )
            episodes_log.flush()

        successes.append(success)
        logger.info(
            f"[{env_name}] Episode {episode + 1}/{args.num_episodes}: success={success}, video={video_path}"
        )

    env.close()
    if episodes_log is not None:
        episodes_log.close()
        inference_log.close()

    success_rate = float(np.mean(successes))
    return {"success_rate": success_rate, "num_episodes": float(len(successes))}


class _NullVideo:
    """Stand-in for the pyav writer when --save_video is off."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def init_video_stream(self, *args, **kwargs):
        pass

    def write_frame(self, frame):
        pass


def _validate_args(args: Args) -> None:
    if args.collect and args.steer:
        raise ValueError(_MUTUALLY_EXCLUSIVE_MODE_ERROR)
    if args.execution_mode not in EXECUTION_MODES:
        raise ValueError(
            f"--execution_mode must be one of {EXECUTION_MODES}, got {args.execution_mode!r}"
        )
    if args.execution_mode == "fixed" and args.replan_steps < 1:
        raise ValueError(f"--replan_steps must be >= 1, got {args.replan_steps}")
    if args.execution_mode == "aac" and (args.collect or args.steer):
        raise ValueError(
            "--execution_mode aac cannot be combined with --collect or --steer"
        )


def main(args: Args) -> None:
    _validate_args(args)
    np.random.seed(args.seed)

    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = policy.get_server_metadata()
    logger.info(f"Server metadata: {server_metadata}")
    if args.execution_mode == "aac" and not server_metadata.get("aac_enabled", False):
        raise RuntimeError(
            "--execution_mode aac requires a server started with `serve.py --aac`"
        )

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        # Bare parent — ``eval_task`` below joins ``env_name`` so the final
        # per-task tree is ``output/{env_name}/episode_NNN.mp4`` (matches the
        # README's documented "Default output" path).
        output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    collect_session = CollectionSession(policy) if args.collect else None

    result = eval_task(
        args.env_name, policy, args, output_dir, collect_session=collect_session
    )
    logger.info(
        f"[{args.env_name}/{args.split}] success_rate={result['success_rate']:.2f} "
        f"({int(result['success_rate'] * result['num_episodes'])}/{int(result['num_episodes'])})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)

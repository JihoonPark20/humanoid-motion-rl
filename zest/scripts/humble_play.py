"""Play a HuMBLE command policy without a motion dataset; use mjpython on macOS."""

import argparse
import time
from pathlib import Path

import mujoco
import torch

from zest.scripts.humble_evaluate import evaluation_env
from zest.utils.humble_runner import load_policy, read_checkpoint
from zest.utils.runner import resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--motion", type=Path, help="Required reference for teacher playback; optional for fidelity playback"
    )
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    saved = read_checkpoint(args.checkpoint)
    is_teacher = saved["humble"]["env_config"]["stage"] == "teacher"
    commands = not is_teacher and args.motion is None
    env = evaluation_env(
        saved, num_envs=1, threads=1, commands=commands, motions=[args.motion] if args.motion else None
    )
    viewer = None
    try:
        device = resolve_device(args.device)
        torch.set_num_threads(2)
        policy = load_policy(args.checkpoint, env, device)
        if commands:
            env.set_command(args.command)
            env.stand()
        data = mujoco.MjData(env.sim.model)
        if not args.headless:
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(env.sim.model, data)
        failures = 0
        for step in range(args.steps):
            start = time.monotonic()
            if viewer is not None and not viewer.is_running():
                break
            with torch.inference_mode():
                _, _, done, info = env.step(policy(env.obs.to(device)))
            failures += int(info["failed"].sum())
            if commands and done.any():
                env.stand()
            data.qpos[:], data.qvel[:] = env.sim.qpos[0], env.sim.qvel[0]
            mujoco.mj_forward(env.sim.model, data)
            if viewer is not None:
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                time.sleep(max(0, env.dt - (time.monotonic() - start)))
        print(f"Played {step + 1} steps; failures={failures}; assistance=0")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()

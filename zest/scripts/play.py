"""Replay a policy or inspect its kinematic reference (use mjpython on macOS)."""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from zest.envs.g1.config import G1Config
from zest.envs.g1.tracking import G1Tracking
from zest.utils.runner import ZestRunner, checkpoint_settings, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--motion", type=Path)
    p.add_argument("--reference", action="store_true", help="Kinematic reference only; no policy or physics")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if not args.reference and not args.checkpoint:
        p.error("Provide --checkpoint, or --reference --motion")
    if args.steps < 1:
        p.error("--steps must be positive")
    saved = checkpoint_settings(args.checkpoint) if args.checkpoint else None
    motion = args.motion or (saved["contract"]["motions"][0]["path"] if saved else None)
    if not motion:
        p.error("--motion is required")
    cfg = G1Config(**saved["env_config"]) if saved else G1Config()
    cfg.num_envs = cfg.num_threads = 1
    cfg.randomize = cfg.observation_noise = cfg.assistance = False
    env = G1Tracking([motion], cfg, evaluation=True)
    viewer = None
    try:
        device = resolve_device(args.device)
        policy = None
        if not args.reference:
            runner = ZestRunner(env, saved["train_config"], device=device)
            runner.load(args.checkpoint, restore_training=False)
            policy = runner.get_inference_policy()
        data = mujoco.MjData(env.sim.model)
        if not args.headless:
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(env.sim.model, data)
        failures = 0
        for step in range(args.steps):
            start = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                break
            if args.reference:
                t = (step * env.dt) % env.library.durations[0]
                ref = env.library.sample(np.array([0]), np.array([t]))
                data.qpos[:7] = ref["qpos"][0, :7]
                data.qpos[env.sim.qpos_ids] = ref["qpos"][0, 7:]
                data.qvel[:6] = ref["qvel"][0, :6]
                data.qvel[env.sim.qvel_ids] = ref["qvel"][0, 6:]
            else:
                with torch.inference_mode():
                    action = policy(env.get_observations().to(device))
                    _, _, _, info = env.step(action)
                failures += int(info["failed"].sum())
                data.qpos[:] = env.sim.qpos[0]
                data.qvel[:] = env.sim.qvel[0]
            mujoco.mj_forward(env.sim.model, data)
            if viewer is not None:
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                time.sleep(max(0, env.dt - (time.perf_counter() - start)))
        print(f"Played {step + 1} steps; policy failures: {failures}; assistance: 0")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()

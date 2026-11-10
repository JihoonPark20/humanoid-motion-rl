"""Train ZEST on prepared G1 motion references."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch

from zest.envs.g1.config import G1Config
from zest.envs.g1.tracking import G1Tracking
from zest.utils.runner import ZestRunner, checkpoint_settings, resolve_device, training_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motions", nargs="+", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-threads", type=int)
    p.add_argument("--iterations", type=int, default=1000, help="Additional PPO iterations")
    p.add_argument("--seed", type=int)
    p.add_argument("--config", type=Path, help="JSON overrides for G1Config")
    p.add_argument("--resume", type=Path)
    p.add_argument("--log-dir", type=Path)
    p.add_argument("--no-assistance", action="store_true")
    p.add_argument("--uniform-sampling", action="store_true")
    p.add_argument("--no-randomization", action="store_true")
    args = p.parse_args()
    if args.iterations < 1:
        p.error("--iterations must be positive")
    saved = checkpoint_settings(args.resume) if args.resume else None
    cfg = G1Config(**saved["env_config"]) if saved else G1Config()
    train_cfg = saved["train_config"] if saved else training_config()
    if args.config:
        values = json.loads(args.config.read_text())
        cfg = G1Config(**{**asdict(cfg), **values})
    for field in ("num_envs", "num_threads", "seed"):
        if getattr(args, field) is not None:
            setattr(cfg, field, getattr(args, field))
    if args.no_assistance:
        cfg.assistance = False
    if args.uniform_sampling:
        cfg.adaptive_sampling = False
    if args.no_randomization:
        cfg.randomize = cfg.observation_noise = False
    motions = args.motions or ([m["path"] for m in saved["contract"]["motions"]] if saved else None)
    if not motions:
        p.error("Provide --motions (use zest-data to prepare the dataset first)")
    device = resolve_device(args.device)
    torch.set_num_threads(min(4, cfg.num_threads))
    torch.manual_seed(cfg.seed)
    directory = args.log_dir or Path("logs/g1") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    env = G1Tracking(motions, cfg)
    try:
        runner = ZestRunner(env, train_cfg, str(directory), device)
        if args.resume:
            runner.load(args.resume)
        snapshot = dict(
            env=asdict(cfg),
            train=train_cfg,
            contract=env.contract(),
            device=device,
            versions={name: version(name) for name in ("torch", "mujoco", "rsl-rl-lib", "numpy")},
        )
        (directory / "run_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"MuJoCo CPU: {cfg.num_envs} environments / {cfg.num_threads} threads; PPO: {device}")
        print(f"Run: {directory}")
        runner.learn(args.iterations, init_at_random_ep_len=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()

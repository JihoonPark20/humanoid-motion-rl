"""Train or resume one HuMBLE stage for a specified number of iterations."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch

from zest.envs.g1.humble import G1Humble
from zest.envs.g1.humble_config import HumbleConfig
from zest.utils.humble_runner import HumbleRunner, config_for, read_checkpoint
from zest.utils.runner import resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["teacher", "distill", "warmup", "finetune"])
    p.add_argument("--motions", nargs="+", type=Path)
    sources = p.add_mutually_exclusive_group()
    sources.add_argument("--init", type=Path, help="Previous stage checkpoint")
    sources.add_argument("--resume", type=Path, help="Continue the same stage")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-threads", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument(
        "--eval-interval", type=int, default=1000, help="Unassisted reference evaluation interval; 0 disables"
    )
    p.add_argument("--log-dir", type=Path)
    p.add_argument("--config", type=Path, help="JSON with env and optional train overrides")
    p.add_argument("--tiny", action="store_true", help="Small networks for execution tests, not paper settings")
    args = p.parse_args()
    if args.iterations < 1 or args.save_interval < 1:
        p.error("Iterations and save interval must be positive")
    if args.eval_interval < 0:
        p.error("--eval-interval cannot be negative")
    saved = read_checkpoint(args.resume or args.init) if args.resume or args.init else None
    values = saved["humble"]["env_config"].copy() if saved else {}
    stage = args.stage or values.get("stage", "teacher")
    values["stage"] = stage
    overrides = json.loads(args.config.read_text()) if args.config else {}
    values.update(overrides.get("env", {}))
    for key in ("num_envs", "num_threads", "seed"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    cfg = HumbleConfig(**values)
    cfg.validate()
    if cfg.stage != "teacher" and saved is None:
        p.error("This stage requires --init or --resume")
    train_cfg = saved["humble"]["train_config"] if args.resume else config_for(cfg.stage, args.tiny)
    if args.init and not args.tiny:
        # Preserve critic architecture across stages, including custom experimental configurations.
        train_cfg["critic"] = saved["humble"]["train_config"]["critic"]
        if cfg.stage in ("warmup", "finetune"):
            train_cfg["actor"] = saved["humble"]["train_config"]["actor"]
    for key, value in overrides.get("train", {}).items():
        if isinstance(value, dict):
            train_cfg[key].update(value)
        else:
            train_cfg[key] = value
    motions = args.motions or ([m["path"] for m in saved["humble"]["contract"]["motions"]] if saved else None)
    if not motions:
        p.error("Provide --motions prepared with humble-data")
    directory = args.log_dir or Path("logs/humble") / f"{cfg.stage}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    torch.set_num_threads(min(4, cfg.num_threads))
    torch.manual_seed(cfg.seed)
    env = G1Humble(motions, cfg)
    try:
        runner = HumbleRunner(env, train_cfg, device)
        if args.resume:
            runner.resume(args.resume)
        elif args.init:
            runner.initialize(args.init)
        (directory / "run_config.json").write_text(
            json.dumps(
                dict(
                    env=asdict(cfg),
                    train=train_cfg,
                    contract=env.contract(),
                    device=device,
                    started_at=datetime.now().astimezone().isoformat(),
                    versions={name: version(name) for name in ("torch", "mujoco", "rsl-rl-lib", "numpy", "h5py")},
                    source_hashes={
                        str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in Path(__file__).resolve().parents[1].rglob("*.py")
                    },
                ),
                indent=2,
            )
        )
        (directory / "status.json").write_text(json.dumps(dict(status="running", pid=os.getpid()), indent=2))
        print(
            f"HuMBLE / {cfg.stage} | {device.upper()} learner | {cfg.num_envs} CPU environments\n"
            f"Run: {directory}\n"
            f"Training: {args.iterations:,} additional iterations | Ctrl+C to save and stop",
            flush=True,
        )
        runner.learn(directory, args.iterations, save_interval=args.save_interval, eval_interval=args.eval_interval)
    except Exception as error:
        (directory / "status.json").write_text(json.dumps(dict(status="failed", error=str(error)), indent=2))
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()

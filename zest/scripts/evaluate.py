"""Evaluate complete clips from frame zero, always without assistive wrench."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zest.envs.g1.config import G1Config
from zest.envs.g1.tracking import G1Tracking
from zest.utils.runner import ZestRunner, checkpoint_settings, resolve_device


def evaluate(env, policy, device):
    rows = []
    for clip_id, clip in enumerate(env.library.clips):
        env.reset(clip_ids=np.full(env.num_envs, clip_id))
        active = np.ones(env.num_envs, dtype=bool)
        counts = np.zeros(env.num_envs, dtype=int)
        sums = {}
        success = np.zeros(env.num_envs, dtype=bool)
        max_steps = int(np.ceil((clip.duration + env.cfg.dwell_seconds) / env.dt))
        for _ in range(max_steps):
            with torch.inference_mode():
                action = policy(env.get_observations().to(device))
                _, _, done, info = env.step(action)
            counts[active] += 1
            for name, value in info["metrics"].items():
                if name not in sums:
                    sums[name] = np.zeros(env.num_envs)
                sums[name][active] += value[active]
            ended = done.numpy() & active
            success[ended] = ~info["failed"][ended]
            active[ended] = False
            if not active.any():
                break
        for i in range(env.num_envs):
            rows.append(
                dict(
                    motion=clip.path.name,
                    motion_sha256=clip.digest,
                    seed=env.cfg.seed,
                    rollout=i,
                    success=bool(success[i]),
                    steps=int(counts[i]),
                    completion=float(min(counts[i] / max_steps, 1)),
                    **{key: float(value[i] / max(counts[i], 1)) for key, value in sums.items()},
                )
            )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--motions", nargs="+", type=Path, help="Optional held-out clips in the same G1 format")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--randomized", action="store_true", help="Enable training noise, dynamics and pushes")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    saved = checkpoint_settings(args.checkpoint)
    motions = args.motions or [m["path"] for m in saved["contract"]["motions"]]
    device = resolve_device(args.device)
    torch.set_num_threads(min(4, args.num_threads))
    rows = []
    for seed in args.seeds:
        cfg = G1Config(**saved["env_config"])
        cfg.num_envs, cfg.num_threads, cfg.seed = args.num_envs, args.num_threads, seed
        cfg.assistance = False
        cfg.randomize = cfg.observation_noise = args.randomized
        env = G1Tracking(motions, cfg, evaluation=True)
        try:
            runner = ZestRunner(env, saved["train_config"], device=device)
            runner.load(args.checkpoint, restore_training=False)
            rows.extend(evaluate(env, runner.get_inference_policy(), device))
        finally:
            env.close()
    rates = [
        np.mean([r["success"] for r in rows if r["motion_sha256"] == digest])
        for digest in dict.fromkeys(r["motion_sha256"] for r in rows)
    ]
    summary = dict(
        success_rate=float(np.mean([r["success"] for r in rows])),
        mean_completion=float(np.mean([r["completion"] for r in rows])),
        per_motion_success_mean=float(np.mean(rates)),
        per_motion_success_p10=float(np.percentile(rates, 10)),
        per_motion_success_min=float(np.min(rates)),
        assistance=0.0,
        randomized=args.randomized,
        checkpoint=str(args.checkpoint.resolve()),
        env_config=asdict(cfg),
        seeds=args.seeds,
        evaluated_motions=[str(Path(m).resolve()) for m in motions],
        rollouts=rows,
    )
    output = args.output or args.checkpoint.parent / "evaluations" / (
        "randomized.json" if args.randomized else "nominal.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Success: {summary['success_rate']:.1%}; completion: {summary['mean_completion']:.1%}; {output}")


if __name__ == "__main__":
    main()

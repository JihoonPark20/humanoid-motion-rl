"""Assistance-free reference fidelity and standing-start command evaluation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from zest.envs.g1.humble import G1Humble
from zest.envs.g1.humble_config import HumbleConfig
from zest.scripts.evaluate import evaluate
from zest.utils.humble_runner import load_policy, read_checkpoint
from zest.utils.runner import resolve_device

COMMANDS = [
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (-0.4, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, -0.3, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (0.5, 0.2, 0.5),
]


def command_evaluation(env, policy, device, seconds=8.0):
    """One-second stand, command for half the remaining horizon, then stop.

    Early failures remain in the report. Steady-state metrics are absent if a
    rollout does not survive the full command plateau; no success-only aggregate.
    """
    if seconds <= 2:
        raise ValueError("Command evaluation needs more than two seconds")
    rows = []
    command_end = 1 + (seconds - 1) / 2
    steps = int(np.ceil(seconds / env.dt))
    env.possible_steps[:] = steps + 1
    for command in COMMANDS:
        env.set_command([0, 0, 0])
        env.stand()
        env.possible_steps[:] = steps + 1
        active = np.ones(env.num_envs, dtype=bool)
        history = [[] for _ in active]
        failed = np.zeros(env.num_envs, dtype=bool)
        for step in range(steps):
            t = step * env.dt
            requested = command if 1 <= t < command_end else (0, 0, 0)
            env.set_command(requested)
            with torch.inference_mode():
                _, _, done, info = env.step(policy(env.obs.to(device)))
            for i in np.flatnonzero(active):
                history[i].append(
                    (t, info["metrics"]["command_linear_error"][i], info["metrics"]["command_yaw_error"][i])
                )
            failed |= active & info["failed"]
            active &= ~done.numpy()
            if not active.any():
                break
        for i, trace in enumerate(history):
            values = np.asarray(trace)
            plateau = (values[:, 0] >= 1) & (values[:, 0] < command_end)
            steady = (values[:, 0] >= command_end - 1) & (values[:, 0] < command_end)
            stop = values[:, 0] >= command_end
            reached_end = len(values) * env.dt >= command_end
            linear_tol = max(0.05, 0.1 * np.linalg.norm(command[:2]))
            yaw_tol = max(0.1, 0.1 * abs(command[2]))
            inside = plateau & (values[:, 1] <= linear_tol) & (values[:, 2] <= yaw_tol)
            # Require 0.2 s continuously inside the tolerance, rather than one lucky sample.
            dwell = max(1, int(round(0.2 / env.dt)))
            hits = np.convolve(inside.astype(int), np.ones(dwell, dtype=int), mode="valid")
            first = np.flatnonzero(hits == dwell) if len(inside) >= dwell else []
            rows.append(
                dict(
                    command=list(command),
                    seed=env.cfg.seed,
                    rollout=i,
                    success=bool(not failed[i] and len(trace) == steps),
                    failed=bool(failed[i]),
                    completion=len(trace) / steps,
                    linear_mae=float(values[:, 1].mean()),
                    yaw_mae=float(values[:, 2].mean()),
                    command_linear_mae=float(values[plateau, 1].mean()) if plateau.any() else None,
                    steady_linear_error=float(values[steady, 1].mean()) if reached_end and steady.any() else None,
                    steady_yaw_error=float(values[steady, 2].mean()) if reached_end and steady.any() else None,
                    stop_linear_error=float(values[stop, 1].mean()) if stop.any() else None,
                    rise_time=float(values[first[0], 0] - 1) if len(first) else None,
                )
            )
    return rows


def evaluation_env(saved, *, seed=0, num_envs=4, threads=2, motions=None, commands=False, randomized=False):
    cfg = HumbleConfig(**saved["humble"]["env_config"])
    cfg.num_envs, cfg.num_threads, cfg.seed = num_envs, threads, seed
    cfg.randomize = cfg.observation_noise = randomized
    cfg.assistance = False
    if commands:
        if cfg.stage == "teacher":
            raise ValueError("A reference-conditioned teacher is not a standalone command policy")
        cfg.stage, cfg.goal_fraction = "finetune", 1.0
    else:
        cfg.goal_fraction = 0.0
        motions = motions or [m["path"] for m in saved["humble"]["contract"]["motions"]]
    return G1Humble(motions, cfg, evaluation=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--mode", choices=["reference", "commands"], default="commands")
    p.add_argument("--motions", type=Path, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--randomized", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    saved = read_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    torch.set_num_threads(min(4, args.num_threads))
    rows = []
    for seed in args.seeds:
        env = evaluation_env(
            saved,
            seed=seed,
            num_envs=args.num_envs,
            threads=args.num_threads,
            motions=args.motions,
            commands=args.mode == "commands",
            randomized=args.randomized,
        )
        try:
            policy = load_policy(args.checkpoint, env, device)
            rows.extend(
                command_evaluation(env, policy, device, args.seconds)
                if args.mode == "commands"
                else evaluate(env, policy, device)
            )
        finally:
            env.close()
    report = dict(
        mode=args.mode,
        assistance=0,
        checkpoint=str(args.checkpoint.resolve()),
        randomized=args.randomized,
        success_rate=float(np.mean([r["success"] for r in rows])),
        mean_completion=float(np.mean([r["completion"] for r in rows])),
        rollouts=rows,
        notes="Nominal repeated seeds are deterministic duplicates. Reference errors are unaligned and over realized steps.",
    )
    output = args.output or args.checkpoint.parent / "evaluations" / f"{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Success={report['success_rate']:.1%}, completion={report['mean_completion']:.1%}: {output}")


if __name__ == "__main__":
    main()

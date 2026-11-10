"""Portable HuMBLE stages with atomic checkpoints and bounded long-running jobs."""

import copy
import json
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from torch.utils.tensorboard import SummaryWriter

from zest.utils.console import TrainingConsole
from zest.utils.runner import TrackingPPO, training_config


def normalize_by_task(advantages, tasks):
    result = advantages.clone()
    for task in (0, 1):
        mask = tasks.squeeze(-1) == task
        values = advantages[mask]
        if values.numel() > 1:
            result[mask] = (values - values.mean()) / (values.std(unbiased=False) + 1e-8)
    return result


class HumblePPO(TrackingPPO):
    warmup = False

    def compute_returns(self, obs):
        # Ask upstream to leave raw advantages untouched, then normalize by task.
        self.normalize_advantage_per_mini_batch = True
        super().compute_returns(obs)
        self.normalize_advantage_per_mini_batch = False
        self.storage.advantages = normalize_by_task(self.storage.advantages, self.storage.observations["task"])

    def update(self):
        if not self.warmup:
            return super().update()
        total, count = 0.0, 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            value = self.critic(batch.observations)
            loss = (value - batch.returns).square().mean() * self.value_loss_coef
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total += loss.item()
            count += 1
        self.critic.update_normalization(self.storage.observations.flatten(0, 1))
        self.storage.clear()
        return {"value": total / count}


def config_for(stage, tiny=False):
    cfg = training_config()
    cfg["multi_gpu"] = None
    cfg["obs_groups"]["actor"] = ["teacher" if stage == "teacher" else "student"]
    cfg["actor"]["hidden_dims"] = [1024, 512, 256] if stage == "teacher" else [1024, 1024, 1024]
    cfg["actor"]["distribution_cfg"]["init_std"] = 1.0 if stage == "teacher" else 0.16
    cfg["algorithm"].update(
        class_name="zest.utils.humble_runner:HumblePPO",
        entropy_coef=0.0015 if stage == "teacher" else 0.001,
        learning_rate=0.001 if stage == "teacher" else 1e-6,
        schedule="adaptive" if stage == "teacher" else "fixed",
        lam=0.95 if stage == "teacher" else 1.0,
    )
    cfg["distill_batches"] = 32
    if tiny:
        cfg["actor"]["hidden_dims"] = [64, 32]
        cfg["critic"]["hidden_dims"] = [64, 32]
        cfg["algorithm"].update(num_learning_epochs=2, num_mini_batches=2)
        cfg["num_steps_per_env"] = 8
        cfg["distill_batches"] = 2
    return cfg


def read_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("humble", {}).get("version") != 1:
        raise ValueError("Expected a HuMBLE checkpoint; ZEST residual policies are incompatible")
    return saved


def make_actor(obs, cfg, device):
    actor_cfg = {k: v for k, v in cfg["actor"].items() if k != "class_name"}
    return MLPModel(obs, cfg["obs_groups"], "actor", 29, **actor_cfg).to(device)


def compatible(saved, env, *, data=True):
    old, new = saved["humble"]["contract"].copy(), env.contract().copy()
    for contract in (old, new):
        contract.pop("policy_obs")  # Teacher and student intentionally differ.
        contract["motions"] = [m["sha256"] for m in contract["motions"]]
        if not data:
            contract.pop("motions")
    if old != new:
        raise ValueError("HuMBLE asset/action/observation/data contract mismatch")


class HumbleRunner:
    def __init__(self, env, cfg, device="cpu"):
        self.env, self.cfg, self.device = env, copy.deepcopy(cfg), device
        self.alg = HumblePPO.construct_algorithm(env.obs, env, self.cfg, device)
        self.alg.warmup = env.cfg.stage == "warmup"
        self.iteration = 0
        self.total_seconds = 0.0
        self.best_completion = -1.0
        self.expert = None
        self.expert_cfg = None
        self.parent_checkpoint = None
        if self.alg.warmup:
            self.alg.actor.requires_grad_(False)
            self.alg.actor.eval()

    def initialize(self, path):
        saved = read_checkpoint(path)
        compatible(saved, self.env)
        previous = saved["humble"]["env_config"]["stage"]
        expected = dict(distill="teacher", warmup="distill", finetune="warmup")
        if expected.get(self.env.cfg.stage) != previous:
            raise ValueError(
                f"{self.env.cfg.stage} requires an initialized {expected.get(self.env.cfg.stage)} checkpoint"
            )
        self.alg.critic.load_state_dict(saved["critic_state_dict"])
        if self.env.cfg.stage == "distill":
            self.expert_cfg = saved["humble"]["train_config"]
            self.expert = make_actor(self.env.obs, self.expert_cfg, self.device)
            self.expert.load_state_dict(saved["actor_state_dict"])
            self.expert.eval().requires_grad_(False)
        else:
            self.alg.actor.load_state_dict(saved["actor_state_dict"])
        self.parent_checkpoint = str(Path(path).resolve())

    def resume(self, path):
        saved = read_checkpoint(path)
        compatible(saved, self.env)
        meta = saved["humble"]
        if meta["env_config"]["stage"] != self.env.cfg.stage:
            raise ValueError("Use --init for a stage transition, --resume for the same stage")
        # Reject silent changes to the MDP; thread/environment count may change.
        old, new = meta["env_config"].copy(), asdict(self.env.cfg)
        for c in (old, new):
            for key in ("num_envs", "num_threads"):
                c.pop(key)
        if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True):
            raise ValueError("Resume configuration differs from saved task")
        self.alg.load(saved, None, True)
        self.iteration, self.total_seconds = meta["completed_iterations"], meta["elapsed_seconds"]
        self.best_completion = meta.get("best_completion", -1.0)
        self.env.sampler.load_state_dict(meta["sampler"])
        self.env.assistance_levels[:] = meta["assistance_levels"]
        self.env.rng.bit_generator.state = meta["env_rng"]
        self.env.reset()  # New RSI episodes; not an exact simulator-state continuation.
        torch.set_rng_state(meta["torch_rng"].cpu())
        if self.device == "mps" and meta.get("mps_rng") is not None:
            torch.mps.set_rng_state(meta["mps_rng"].cpu())
        if self.device.startswith("cuda") and meta.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(meta["cuda_rng"].cpu(), self.device)
        self.parent_checkpoint = meta.get("parent_checkpoint")
        if self.env.cfg.stage == "distill":
            self.expert_cfg = saved["expert_cfg"]
            self.expert = make_actor(self.env.obs, self.expert_cfg, self.device)
            self.expert.load_state_dict(saved["expert_state_dict"])
            self.expert.eval().requires_grad_(False)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = self.alg.save()
        saved["humble"] = dict(
            version=1,
            env_config=asdict(self.env.cfg),
            train_config=self.cfg,
            contract=self.env.contract(),
            completed_iterations=self.iteration,
            elapsed_seconds=self.total_seconds,
            sampler=self.env.sampler.state_dict(),
            assistance_levels=self.env.assistance_levels.copy(),
            env_rng=self.env.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state() if self.device == "mps" else None,
            cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.startswith("cuda") else None,
            parent_checkpoint=self.parent_checkpoint,
            best_completion=self.best_completion,
        )
        if self.expert is not None:
            saved["expert_cfg"] = self.expert_cfg
            saved["expert_state_dict"] = self.expert.state_dict()
        temporary = path.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(path)

    def collect_update(self):
        obs = self.env.get_observations().to(self.device)
        distill = self.env.cfg.stage == "distill"
        if distill and self.expert is None:
            raise RuntimeError("Distillation requires a teacher checkpoint")
        observations, labels, logs = [], [], []
        with torch.no_grad():
            for _ in range(self.cfg["num_steps_per_env"]):
                if distill:
                    observations.append(obs.clone())
                    labels.append(self.expert(obs))
                    actions = self.alg.actor(obs)  # Student's own states; deterministic DAgger rollout.
                else:
                    actions = self.alg.act(obs)
                obs, rewards, dones, info = self.env.step(actions)
                obs = obs.to(self.device)
                if not torch.isfinite(rewards).all():
                    raise FloatingPointError("Nonfinite rewards")
                if not distill:
                    self.alg.process_env_step(obs, rewards.to(self.device), dones.to(self.device), info)
                if "log" in info:
                    logs.append(info["log"])
            if not distill:
                self.alg.compute_returns(obs)
        if distill:
            data = torch.stack(observations).flatten(0, 1)
            targets = torch.stack(labels).flatten(0, 1)
            losses = []
            for indices in torch.randperm(len(targets), device=self.device).chunk(self.cfg["distill_batches"]):
                prediction = self.alg.actor(data[indices])
                loss = (prediction - targets[indices]).square().mean()
                self.alg.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.alg.actor.parameters(), 1.0)
                self.alg.optimizer.step()
                losses.append(loss.item())
            self.alg.actor.update_normalization(data)
            losses = {"distillation_mse": float(np.mean(losses))}
        else:
            losses = self.alg.update()
        if not all(np.isfinite(v) for v in losses.values()):
            raise FloatingPointError("Nonfinite optimization loss")
        if logs:
            losses.update({k: float(np.mean([log[k] for log in logs])) for k in logs[0]})
        return losses

    def evaluate_references(self, directory):
        from zest.envs.g1.humble import G1Humble
        from zest.envs.g1.humble_config import HumbleConfig
        from zest.scripts.evaluate import evaluate

        cfg = HumbleConfig(**asdict(self.env.cfg))
        cfg.num_envs, cfg.num_threads = 1, 1
        cfg.assistance = cfg.randomize = cfg.observation_noise = False
        cfg.goal_fraction = 0.0
        env = G1Humble([c.path for c in self.env.library.clips], cfg, evaluation=True)
        was_training = self.alg.actor.training
        try:
            self.alg.actor.eval()
            rows = evaluate(env, self.alg.actor, self.device)
        finally:
            self.alg.actor.train(was_training)
            env.close()
        metrics = {
            "Unassisted/completion": float(np.mean([r["completion"] for r in rows])),
            "Unassisted/success": float(np.mean([r["success"] for r in rows])),
        }
        report = dict(
            iteration=self.iteration,
            scope="training references; not held-out",
            assistance=0,
            metrics=metrics,
            rollouts=rows,
        )
        (Path(directory) / f"reference_evaluation_{self.iteration}.json").write_text(json.dumps(report, indent=2))
        if metrics["Unassisted/completion"] > self.best_completion:
            self.best_completion = metrics["Unassisted/completion"]
            self.save(Path(directory) / "best_reference.pt")
        return metrics

    def learn(self, directory, iterations, save_interval=100, eval_interval=1000):
        directory = Path(directory)
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        writer = SummaryWriter(str(directory))
        console = TrainingConsole(self.iteration, iterations)
        reason = "iteration_limit"
        try:
            with (directory / "metrics.jsonl").open("a") as stream:
                for _ in range(iterations):
                    if stopped or (directory / "STOP").exists():
                        reason = "signal" if stopped else "stop_file"
                        break
                    start = time.monotonic()
                    metrics = self.collect_update()
                    elapsed = time.monotonic() - start
                    self.total_seconds += elapsed
                    self.iteration += 1
                    if eval_interval and self.iteration % eval_interval == 0:
                        console.note(f"Evaluating unassisted tracking at iteration {self.iteration}...")
                        evaluation_start = time.monotonic()
                        metrics.update(self.evaluate_references(directory))
                        self.total_seconds += time.monotonic() - evaluation_start
                        console.note(
                            f"Reference evaluation | completion {metrics['Unassisted/completion']:.1%} "
                            f"| success {metrics['Unassisted/success']:.1%}"
                        )
                    metrics.update(
                        iteration=self.iteration,
                        elapsed_seconds=self.total_seconds,
                        transitions_per_second=self.env.num_envs * self.cfg["num_steps_per_env"] / elapsed,
                        learning_rate=self.alg.learning_rate,
                    )
                    stream.write(json.dumps(metrics) + "\n")
                    stream.flush()
                    for name, value in metrics.items():
                        writer.add_scalar(name, value, self.iteration)
                    if self.iteration % save_interval == 0 or self.iteration == 1:
                        self.save(directory / "latest.pt")
                        (directory / "status.json").write_text(
                            json.dumps(
                                dict(
                                    status="running",
                                    pid=os.getpid(),
                                    iterations=self.iteration,
                                    elapsed_seconds=self.total_seconds,
                                ),
                                indent=2,
                            )
                        )
                    if self.iteration % (10 * save_interval) == 0:
                        self.save(directory / f"model_{self.iteration}.pt")
                    console.update(metrics, report="Unassisted/completion" in metrics)
            self.save(directory / "latest.pt")
            (directory / "status.json").write_text(
                json.dumps(
                    dict(
                        status="stopped", reason=reason, iterations=self.iteration, elapsed_seconds=self.total_seconds
                    ),
                    indent=2,
                )
            )
            console.finish(reason, self.iteration, directory / "latest.pt")
        finally:
            console.close()
            writer.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def load_policy(path, env, device):
    saved = read_checkpoint(path)
    compatible(saved, env, data=False)
    actor = make_actor(env.obs, saved["humble"]["train_config"], device)
    actor.load_state_dict(saved["actor_state_dict"])
    return actor.eval()

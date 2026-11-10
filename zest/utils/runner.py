"""Thin rsl_rl PPO adapter: terminal-state bootstrapping and curriculum persistence."""

from dataclasses import asdict

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device unavailable: {requested}")
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
    elif device.type != "cpu":
        raise ValueError("Use auto, cpu, mps, or cuda[:N]")
    return str(device)


def training_config():
    return dict(
        num_steps_per_env=24,
        save_interval=100,
        logger="tensorboard",
        check_for_nan=True,
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        actor=dict(
            class_name="rsl_rl.models:MLPModel",
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=dict(class_name="rsl_rl.modules:GaussianDistribution", std_type="scalar", init_std=1.0),
        ),
        critic=dict(
            class_name="rsl_rl.models:MLPModel", hidden_dims=[512, 512, 256], activation="elu", obs_normalization=True
        ),
        algorithm=dict(
            class_name="zest.utils.runner:TrackingPPO",
            clip_param=0.2,
            desired_kl=0.01,
            entropy_coef=0.001,
            gamma=0.99,
            lam=0.95,
            learning_rate=0.001,
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            schedule="adaptive",
            use_clipped_value_loss=True,
            value_loss_coef=0.5,
            rnd_cfg=None,
            symmetry_cfg=None,
        ),
    )


class TrackingPPO(PPO):
    def process_env_step(self, obs, rewards, dones, extras):
        # Upstream 5.5.2 bootstraps timeouts with V(s_t); use V(s_{t+1}) before auto-reset instead.
        timeouts = extras.get("time_outs")
        if timeouts is not None and timeouts.any():
            with torch.no_grad():
                values = self.critic(extras["terminal_observations"].to(self.device)).squeeze(-1)
                rewards = rewards + self.gamma * values * timeouts.to(self.device)
        super().process_env_step(obs, rewards, dones, {k: v for k, v in extras.items() if k != "time_outs"})


class ZestRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        metadata = dict(infos or {})
        metadata["zest"] = dict(
            contract=self.env.contract(),
            env_config=asdict(self.env.cfg),
            train_config=self.cfg,
            sampler=self.env.sampler.state_dict(),
            rng=self.env.rng.bit_generator.state,
            completed_iterations=self.current_learning_iteration + 1,
            learning_rate=self.alg.learning_rate,
        )
        super().save(str(path), metadata)

    def load(self, path, *, restore_training=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved = checkpoint["infos"]["zest"]
        expected, actual = saved["contract"].copy(), self.env.contract().copy()
        # Paths can change across machines, but training must use the same ordered motion content.
        for contract in (expected, actual):
            contract["motions"] = [m["sha256"] for m in contract["motions"]]
        if not restore_training:
            expected.pop("motions")
            actual.pop("motions")
        if expected != actual:
            raise ValueError("Checkpoint policy/motion contract does not match this environment")
        self.alg.load(checkpoint, None, True)
        self.current_learning_iteration = saved["completed_iterations"]
        if restore_training:
            self.alg.learning_rate = saved["learning_rate"]
            for group in self.alg.optimizer.param_groups:
                group["lr"] = saved["learning_rate"]
            self.env.sampler.load_state_dict(saved["sampler"])
            self.env.rng.bit_generator.state = saved["rng"]
            self.env.reset()
        return saved


def checkpoint_settings(path):
    return torch.load(path, map_location="cpu", weights_only=False)["infos"]["zest"]

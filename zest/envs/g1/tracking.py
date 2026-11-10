"""G1 reference tracking with adaptive RSI and model-based assistive wrench."""

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from zest.envs.g1.config import JOINT_NAMES, KEY_INDICES, G1Config
from zest.motion.library import MotionLibrary
from zest.motion.sampler import AdaptiveSampler
from zest.sim.mujoco_batch import MujocoBatch
from zest.utils.math import conjugate, difference, inverse_rotate, multiply, rotate

GRAVITY = np.array([0.0, 0.0, -1.0])


def assistive_wrench(reference, position, quaternion, velocity, omega, mass, inertia, com, cfg):
    """Equation S6, world-frame force/moment about the controlled link origin."""
    force = mass * (
        reference["root_acc"]
        + cfg.force_kp * (reference["qpos"][:, :3] - position)
        + cfg.force_kd * (reference["root_vel"] - velocity)
        - 9.81 * GRAVITY
    )

    def inertia_world(v):
        return rotate(quaternion, inverse_rotate(quaternion, v) @ inertia.T)

    moment = inertia_world(reference["root_alpha"])
    moment += cfg.torque_kp * inertia_world(difference(reference["qpos"][:, 3:7], quaternion))
    moment += cfg.torque_kd * inertia_world(reference["root_omega"] - omega)
    moment += np.cross(omega, inertia_world(omega))
    moment -= np.cross(rotate(quaternion, np.broadcast_to(com, position.shape)), mass * 9.81 * GRAVITY)
    return np.concatenate((force, moment), axis=-1)


class G1Tracking(VecEnv):
    device = "cpu"
    num_actions = 29

    def __init__(self, motions, cfg=None, *, evaluation=False):
        self.cfg = cfg or G1Config()
        self.cfg.validate()
        self.num_envs = self.cfg.num_envs
        self.dt = self.cfg.dt
        self.max_episode_length = int(np.ceil(self.cfg.episode_seconds / self.dt))
        self.rng = np.random.default_rng(self.cfg.seed)
        self.evaluation = evaluation
        self.sim = MujocoBatch(self.cfg, self.rng)
        self.library = self.make_library(motions, self.sim.model)
        self.sampler = AdaptiveSampler(self.library.durations, self.cfg)
        _, self.kp, self.kd, self.effort, self.scale = self.cfg.motor_parameters()
        n = self.num_envs
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.clip_ids = np.zeros(n, dtype=int)
        self.times = np.zeros(n)
        self.bin_ids = np.zeros(n, dtype=int)
        self.beta = np.zeros(n)
        self.possible_steps = np.ones(n, dtype=int)
        self.score_sum = np.zeros(n)
        self.episode_rewards = np.zeros(n)
        self.actions = np.zeros((n, 29))
        self.previous_actions = self.actions.copy()
        self.wrench = np.zeros((n, 6))
        self.tracking_rewards = np.zeros((n, 7))
        self.next_push = np.zeros(n)
        self.closed = False
        self.configure_task()
        self.reset_idx(np.arange(n))
        self.obs = self.compute_observations()

    def make_library(self, motions, model):
        return MotionLibrary(motions, model)

    def configure_task(self):
        pass

    def joint_targets(self, target):
        return target["qpos"][:, 7:] + self.scale * self.actions

    def update_curriculum(self, ids):
        self.sampler.update(self.bin_ids[ids], self.score_sum[ids], self.possible_steps[ids])

    def state(self):
        ids = self.sim.body_ids
        return dict(
            position=self.sim.xpos[:, ids[0]],
            quaternion=self.sim.xquat[:, ids[0]],
            body_pos=self.sim.xpos[:, ids],
            body_quat=self.sim.xquat[:, ids],
            velocity=self.sim.linear_velocity[:, 0],
            omega=self.sim.angular_velocity[:, 0],
            joints=self.sim.qpos[:, self.sim.qpos_ids],
            joint_vel=self.sim.qvel[:, self.sim.qvel_ids],
        )

    def reset_idx(self, ids, clip_ids=None, start_times=None):
        if len(ids) == 0:
            return
        if clip_ids is None and not self.evaluation:
            clips, times, bins, beta = self.sampler.sample(len(ids), self.rng)
        else:
            clips = (
                self.rng.integers(len(self.library.clips), size=len(ids)) if clip_ids is None else np.asarray(clip_ids)
            )
            times = np.zeros(len(ids)) if start_times is None else np.asarray(start_times)
            bins, beta = np.zeros(len(ids), dtype=int), np.zeros(len(ids))
        ref = self.library.sample(clips, times)
        self.sim.reset(ids)
        self.sim.qpos[ids, :7] = ref["qpos"][:, :7]
        self.sim.qpos[np.ix_(ids, self.sim.qpos_ids)] = ref["qpos"][:, 7:]
        self.sim.qvel[ids, :6] = ref["qvel"][:, :6]
        self.sim.qvel[np.ix_(ids, self.sim.qvel_ids)] = ref["qvel"][:, 6:]
        if self.cfg.randomize:
            self.sim.qpos[ids, :3] += self.rng.uniform(
                -self.cfg.reset_position_noise, self.cfg.reset_position_noise, (len(ids), 3)
            )
            self.sim.qpos[np.ix_(ids, self.sim.qpos_ids)] += self.rng.normal(
                0, self.cfg.reset_joint_noise, (len(ids), 29)
            )
            self.sim.qvel[ids] += self.rng.normal(0, self.cfg.reset_velocity_noise, (len(ids), self.sim.model.nv))
        self.sim.ctrl[ids] = ref["qpos"][:, 7:]
        self.sim.xfrc_applied[ids] = 0
        self.sim.forward(ids)
        self.clip_ids[ids], self.times[ids], self.bin_ids[ids], self.beta[ids] = clips, times, bins, beta
        remaining = self.library.durations[clips] + self.cfg.dwell_seconds - times
        # Remaining feasible horizon, rather than the full clip length, for late RSI starts.
        self.possible_steps[ids] = np.minimum(self.max_episode_length, np.ceil(remaining / self.dt).astype(int))
        if self.evaluation:
            self.possible_steps[ids] = np.ceil(remaining / self.dt).astype(int)
        self.episode_length_buf[ids] = 0
        self.score_sum[ids] = self.episode_rewards[ids] = 0
        self.actions[ids] = self.previous_actions[ids] = 0
        self.wrench[ids] = self.tracking_rewards[ids] = 0
        self.next_push[ids] = self.rng.uniform(0, 10, len(ids))

    def reset(self, clip_ids=None, start_times=None):
        self.reset_idx(np.arange(self.num_envs), clip_ids, start_times)
        self.obs = self.compute_observations()
        return self.obs

    def compute_observations(self):
        s = self.state()
        ref = self.library.sample(self.clip_ids, self.times + self.dt)
        torso_q = s["body_quat"][:, 1]
        prop = np.concatenate(
            (
                inverse_rotate(torso_q, self.sim.angular_velocity[:, 1]),
                inverse_rotate(torso_q, GRAVITY),
                s["joints"],
                s["joint_vel"],
                self.actions,
            ),
            axis=-1,
        )
        if self.cfg.observation_noise:
            std = np.r_[np.full(3, 0.1), np.full(3, 0.015), np.full(29, 0.005), np.full(29, 0.25), np.zeros(29)]
            prop += self.rng.normal(size=prop.shape) * std
        rq = ref["qpos"][:, 3:7]
        target = np.concatenate(
            (
                ref["qpos"][:, 2:3],
                inverse_rotate(rq, ref["root_vel"]),
                inverse_rotate(rq, ref["root_omega"]),
                inverse_rotate(rq, GRAVITY),
                ref["qpos"][:, 7:],
            ),
            axis=-1,
        )
        actor = np.concatenate((prop, target), axis=-1)
        q = s["quaternion"]
        keypos = inverse_rotate(q[:, None], s["body_pos"][:, KEY_INDICES] - s["position"][:, None])
        keyvel = inverse_rotate(q[:, None], self.sim.linear_velocity[:, KEY_INDICES])
        contacts = inverse_rotate(q[:, None], self.sim.contacts)
        phase = np.minimum(self.times / self.library.durations[self.clip_ids], 1)
        critic = np.concatenate(
            (
                actor,
                inverse_rotate(q, s["velocity"]),
                s["position"][:, 2:3],
                contacts.reshape(self.num_envs, -1),
                keypos.reshape(self.num_envs, -1),
                keyvel.reshape(self.num_envs, -1),
                self.wrench,
                self.beta[:, None],
                self.tracking_rewards,
                phase[:, None],
            ),
            axis=-1,
        )
        if not np.isfinite(critic).all():
            raise FloatingPointError("Nonfinite observation")
        return TensorDict(
            {
                "policy": torch.from_numpy(actor.astype(np.float32)),
                "critic": torch.from_numpy(critic.astype(np.float32)),
            },
            batch_size=[self.num_envs],
        )

    def get_observations(self):
        return self.obs

    def reward_and_failure(self, ref, requested_torque):
        s = self.state()
        q, rq = s["quaternion"], ref["qpos"][:, 3:7]
        kp = inverse_rotate(q[:, None], s["body_pos"][:, KEY_INDICES] - s["position"][:, None])
        rkp = inverse_rotate(rq[:, None], ref["body_pos"][:, KEY_INDICES] - ref["qpos"][:, None, :3])
        kq = multiply(conjugate(q[:, None]), s["body_quat"][:, KEY_INDICES])
        rkq = multiply(conjugate(rq[:, None]), ref["body_quat"][:, KEY_INDICES])
        root_error = ref["qpos"][:, :3] - s["position"]
        orientation_error = difference(rq, q)
        joint_error = ref["qpos"][:, 7:] - s["joints"]
        errors = (
            root_error,
            orientation_error,
            ref["root_omega"] - s["omega"],
            ref["root_vel"] - s["velocity"],
            joint_error,
            rkp - kp,
            difference(rkq, kq),
        )
        sigmas = (
            0.4,
            0.5,
            1.5,
            0.6,
            0.3 * np.sqrt(29),
            0.2 * np.sqrt(len(KEY_INDICES)),
            0.4 * np.sqrt(len(KEY_INDICES)),
        )
        self.tracking_rewards = np.stack(
            [
                np.exp(-0.25 * np.square(e.reshape(self.num_envs, -1)).sum(-1) / sig**2)
                for e, sig in zip(errors, sigmas)
            ],
            axis=-1,
        )
        penalty = 0.15 * np.linalg.norm(self.actions - self.previous_actions, axis=-1)
        penalty += 1e-5 * np.linalg.norm(self.sim.qacc[:, self.sim.qvel_ids], axis=-1)
        penalty += (
            np.maximum(self.sim.limits[:, 0] - s["joints"], 0) + np.maximum(s["joints"] - self.sim.limits[:, 1], 0)
        ).sum(-1)
        penalty += 0.1 * np.maximum(np.abs(requested_torque) - self.effort, 0).sum(-1)
        reward = (self.tracking_rewards.sum(-1) + 1 - penalty) * self.dt
        root_norm = np.linalg.norm(root_error, axis=-1)
        angle_norm = np.linalg.norm(orientation_error, axis=-1)
        failure = (
            (root_norm > self.cfg.max_root_error)
            | (angle_norm > self.cfg.max_orientation_error)
            | (np.linalg.norm(self.sim.contacts, axis=-1).max(-1) > self.cfg.max_contact_force)
        )
        metrics = dict(
            joint_mae=np.abs(joint_error).mean(-1),
            root_position_error=root_norm,
            root_orientation_error=angle_norm,
            root_angular_velocity_error=np.linalg.norm(errors[2], axis=-1),
            similarity=self.tracking_rewards[:, 4].copy(),
        )
        return reward, failure, metrics

    def step(self, actions):
        if self.closed:
            raise RuntimeError("Environment is closed")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        actions = np.asarray(actions)
        if actions.shape != self.actions.shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions of shape {self.actions.shape}")
        self.previous_actions[:] = self.actions
        self.actions[:] = actions
        target = self.library.sample(self.clip_ids, self.times + self.dt)
        joint_targets = self.joint_targets(target)
        torque_sum = np.zeros_like(self.actions)
        for k in range(self.cfg.decimation):
            s = self.state()
            ref = self.library.sample(self.clip_ids, self.times + (k + 1) * self.cfg.sim_dt)
            torso_ref = dict(
                qpos=np.concatenate((ref["body_pos"][:, 1], ref["body_quat"][:, 1]), axis=-1),
                root_vel=ref["body_vel"][:, 1],
                root_omega=ref["body_omega"][:, 1],
                root_acc=ref["body_acc"][:, 1],
                root_alpha=ref["body_alpha"][:, 1],
            )
            self.wrench = (
                assistive_wrench(
                    torso_ref,
                    s["body_pos"][:, 1],
                    s["body_quat"][:, 1],
                    self.sim.linear_velocity[:, 1],
                    self.sim.angular_velocity[:, 1],
                    self.sim.mass,
                    self.sim.inertia,
                    self.sim.com,
                    self.cfg,
                )
                * self.beta[:, None]
            )
            torso = self.sim.body_ids[1]
            applied = self.wrench.copy()
            # xfrc_applied acts at torso COM: preserve the requested moment about the torso origin.
            applied[:, 3:] -= np.cross(self.sim.xipos[:, torso] - s["body_pos"][:, 1], applied[:, :3])
            self.sim.xfrc_applied[:] = 0
            self.sim.xfrc_applied[:, torso] = applied
            torque_sum += np.abs(self.kp * (joint_targets - s["joints"]) - self.kd * s["joint_vel"])
            self.sim.step(joint_targets)
        self.times += self.dt
        self.episode_length_buf += 1
        # Refresh acceleration-stage contacts at the final state before reward/critic evaluation.
        self.sim.forward()
        ref = self.library.sample(self.clip_ids, self.times)
        reward, failure, metrics = self.reward_and_failure(ref, torque_sum / self.cfg.decimation)
        self.score_sum += self.tracking_rewards[:, 4]
        self.episode_rewards += reward
        timeout = self.episode_length_buf.numpy() >= self.possible_steps
        dones = failure | timeout
        extras = {
            "time_outs": torch.from_numpy(timeout & ~failure),
            "metrics": metrics,
            "failed": failure.copy(),
            "clip_ids": self.clip_ids.copy(),
            "reference_times": self.times.copy(),
        }
        if np.any(timeout & ~failure):
            extras["terminal_observations"] = self.compute_observations()
        ids = np.flatnonzero(dones)
        if len(ids):
            extras["log"] = {
                "Episode/reward": float(self.episode_rewards[ids].mean()),
                "Episode/similarity": float(np.mean(self.score_sum[ids] / self.possible_steps[ids])),
                "Episode/failure": float(failure[ids].mean()),
                "Curriculum/beta": float(self.beta[ids].mean()),
            }
            if not self.evaluation:
                self.update_curriculum(ids)
            self.reset_idx(ids)
        if self.cfg.randomize:
            self.next_push -= self.dt
            push_ids = np.flatnonzero(self.next_push <= 0)
            if len(push_ids):
                self.sim.qvel[push_ids, :2] += self.rng.uniform(-0.5, 0.5, (len(push_ids), 2))
                self.next_push[push_ids] = self.rng.uniform(self.dt, 10, len(push_ids))
                self.sim.forward(push_ids)
        self.obs = self.compute_observations()
        return self.obs, torch.from_numpy(reward.astype(np.float32)), torch.from_numpy(dones.copy()), extras

    def contract(self):
        return {
            "version": 2,
            "asset_sha256": self.sim.asset_digest,
            "joint_names": JOINT_NAMES,
            "motions": self.library.manifest,
            "dt": self.dt,
            "sim_dt": self.cfg.sim_dt,
            "kp": self.kp.tolist(),
            "kd": self.kd.tolist(),
            "action_scale": self.scale.tolist(),
            "torso_inertia": self.sim.inertia.tolist(),
            "torso_com_offset": self.sim.com.tolist(),
            "contact_convention": "world_force_on_robot_then_root_frame",
            "policy_obs": self.obs["policy"].shape[-1],
            "critic_obs": self.obs["critic"].shape[-1],
        }

    def close(self):
        if not self.closed:
            self.sim.close()
            self.closed = True

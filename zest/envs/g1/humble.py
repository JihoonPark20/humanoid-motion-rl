"""HuMBLE flat-ground teacher, prior and mixed reference/command tasks."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from zest.envs.g1.config import BODY_NAMES, JOINT_NAMES
from zest.envs.g1.humble_config import HumbleConfig, joint_parameters
from zest.envs.g1.tracking import GRAVITY, G1Tracking
from zest.motion.commands import CommandLibrary
from zest.sim.mujoco_batch import composite_inertia
from zest.utils.math import difference, inverse_rotate


class G1Humble(G1Tracking):
    def __init__(self, motions=None, cfg=None, *, evaluation=False):
        self._temporary = None
        super().__init__(motions, cfg or HumbleConfig(), evaluation=evaluation)

    def make_library(self, motions, model):
        if not motions:
            if not self.evaluation:
                raise ValueError("Training requires prepared HOLD motions")
            self._temporary = tempfile.TemporaryDirectory(prefix="humble-deploy-")
            path = Path(self._temporary.name) / "nominal.npz"
            q = np.repeat(model.qpos0[None], 51, axis=0)
            q[:, 7:] = joint_parameters()[2]
            np.savez(
                path,
                version=1,
                qpos=q,
                fps=50.0,
                joint_names=JOINT_NAMES,
                quaternion_order="wxyz",
                commands=np.zeros((51, 3)),
            )
            motions = [path]
        return CommandLibrary(motions, model, self.cfg.command_bin)

    def configure_task(self):
        self.nominal = joint_parameters()[2]
        self.sim.mass, self.sim.inertia, self.sim.com = composite_inertia(self.sim.model, self.nominal)
        n = self.num_envs
        self.goal = np.zeros(n, dtype=bool)
        if self.cfg.stage == "finetune":
            self.goal[: int(round(n * self.cfg.goal_fraction))] = True
        self.commands = np.zeros((n, 3))
        self.command_timer = np.zeros(n)
        self.foot_velocity = np.zeros((n, 2, 3))
        self.foot_acceleration = np.zeros((n, 2, 3))
        self.assistance_levels = np.full(len(self.sampler.failure), self.cfg.beta_max)
        self.norms = []
        for clip in self.library.clips:
            v = clip.values
            q = v["body_quat"][:, 1]
            parts = (
                inverse_rotate(q, v["body_vel"][:, 1]),
                inverse_rotate(q, v["body_omega"][:, 1]),
                inverse_rotate(q, GRAVITY),
                v["qpos"][:, 7:],
            )
            self.norms.append([np.maximum(x.std(0), 0.2) for x in parts])
        self.max_episode_length = int(np.ceil(max(self.cfg.episode_seconds, self.cfg.goal_seconds) / self.dt))
        self.fixed_command = None

    def draw_commands(self, count):
        commands = self.rng.uniform(self.cfg.command_low, self.cfg.command_high, (count, 3))
        commands[self.rng.random(count) < self.cfg.stop_probability] = 0
        return commands

    def reset_idx(self, ids, clip_ids=None, start_times=None):
        if len(ids) == 0:
            return
        goal_ids = ids[self.goal[ids]]
        if len(goal_ids):
            self.commands[goal_ids] = (
                self.fixed_command if self.fixed_command is not None else self.draw_commands(len(goal_ids))
            )
        super().reset_idx(ids, clip_ids, start_times)
        if len(goal_ids):
            clips, times = self.library.initial_states(self.commands[goal_ids], self.rng, self.cfg.reset_window)
            super().reset_idx(goal_ids, clips, times)
            self.possible_steps[goal_ids] = int(np.ceil(self.cfg.goal_seconds / self.dt))
        ref_ids = ids[~self.goal[ids]]
        if not self.evaluation and self.cfg.stage != "teacher":
            self.possible_steps[ref_ids] = np.minimum(
                self.possible_steps[ref_ids], int(np.ceil(self.cfg.reference_seconds / self.dt))
            )
        self.beta[ids] = 0
        if self.cfg.stage == "teacher" and self.cfg.assistance and not self.evaluation:
            self.beta[ids] = self.assistance_levels[self.bin_ids[ids]]
        self.command_timer[ids] = self.cfg.command_interval
        self.foot_velocity[ids] = self.sim.linear_velocity[ids, 2:4]
        self.foot_acceleration[ids] = 0
        self.refresh_reference_commands()

    def refresh_reference_commands(self):
        ids = np.flatnonzero(~self.goal)
        if len(ids):
            self.commands[ids] = self.library.sample(self.clip_ids[ids], self.times[ids] + self.dt)["commands"]

    def joint_targets(self, target):
        return self.nominal + self.scale * self.actions

    def sensor(self, name):
        sensor = self.sim.model.sensor(name)
        return self.sim.sensordata[:, sensor.adr[0] : sensor.adr[0] + sensor.dim[0]]

    def compute_observations(self):
        self.refresh_reference_commands()
        s = self.state()
        ref = self.library.sample(self.clip_ids, self.times + self.dt)
        q, rq = s["body_quat"][:, 1], ref["body_quat"][:, 1]
        clean = np.c_[
            inverse_rotate(q, self.sim.angular_velocity[:, 1]),
            inverse_rotate(q, GRAVITY),
            s["joints"],
            s["joint_vel"],
            self.actions,
        ]
        prop = clean.copy()
        if self.cfg.observation_noise:
            std = np.r_[np.full(3, 0.1), np.full(3, 0.015), np.full(29, 0.005), np.full(29, 0.25), np.zeros(29)]
            prop += self.rng.normal(size=prop.shape) * std
        target = np.c_[
            ref["body_pos"][:, 1, 2],
            inverse_rotate(rq, ref["body_vel"][:, 1]),
            inverse_rotate(rq, ref["body_omega"][:, 1]),
            inverse_rotate(rq, GRAVITY),
            ref["qpos"][:, 7:],
        ]
        target[self.goal] = 0  # Defined padding convention for reference-free goal critic inputs.
        student = np.c_[prop, self.commands]
        teacher = np.c_[student, target]
        ee_pos = inverse_rotate(q[:, None], s["body_pos"][:, 2:6] - s["body_pos"][:, 1:2])
        ee_vel = inverse_rotate(q[:, None], self.sim.linear_velocity[:, 2:6] - self.sim.linear_velocity[:, 1:2])
        wrenches = np.concatenate([-self.sensor(f"wrench_{b}") for b in BODY_NAMES[2:6]], axis=-1)
        critic = np.c_[
            clean,
            self.commands,
            target,
            inverse_rotate(q, self.sim.linear_velocity[:, 1]),
            s["body_pos"][:, 1, 2],
            wrenches,
            ee_pos.reshape(self.num_envs, -1),
            ee_vel.reshape(self.num_envs, -1),
            np.linalg.norm(self.wrench[:, :3], axis=-1),
            np.linalg.norm(self.wrench[:, 3:], axis=-1),
            self.beta,
            self.goal.astype(float),
        ]
        arrays = dict(
            policy=teacher if self.cfg.stage == "teacher" else student,
            teacher=teacher,
            student=student,
            critic=critic,
            task=self.goal[:, None].astype(float),
        )
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("Nonfinite HuMBLE observation")
        return TensorDict(
            {key: torch.from_numpy(value.astype(np.float32)) for key, value in arrays.items()},
            batch_size=[self.num_envs],
        )

    def reward_and_failure(self, ref, requested_torque):
        s = self.state()
        q, rq = s["body_quat"][:, 1], ref["body_quat"][:, 1]
        velocity = inverse_rotate(q, self.sim.linear_velocity[:, 1])
        omega = inverse_rotate(q, self.sim.angular_velocity[:, 1])
        gravity = inverse_rotate(q, GRAVITY)
        reference_gravity = inverse_rotate(rq, GRAVITY)
        errors = [
            inverse_rotate(rq, ref["body_vel"][:, 1]) - velocity,
            inverse_rotate(rq, ref["body_omega"][:, 1]) - omega,
            reference_gravity - gravity,
            ref["qpos"][:, 7:] - s["joints"],
        ]
        norms = np.stack(
            [
                np.linalg.norm(e / np.stack([self.norms[k][i] for k in self.clip_ids]), axis=-1)
                for i, e in enumerate(errors)
            ],
            axis=-1,
        )
        track = ((5 * np.exp(-norms / 10) - norms**2) * [1, 1, 1, 0.5]).sum(-1)
        self.tracking_rewards[:] = 0
        self.tracking_rewards[:, 4] = np.exp(-norms[:, 3] / 10)
        foot_v = self.sim.linear_velocity[:, 2:4]
        acceleration = (foot_v - self.foot_velocity) / self.dt
        jerk = (acceleration - self.foot_acceleration) / self.dt
        # A reset has no preceding acceleration history; exclude its first two samples.
        jerk[self.episode_length_buf.numpy() <= 2] = 0
        self.foot_velocity[:], self.foot_acceleration[:] = foot_v, acceleration
        contact = np.linalg.norm(self.sim.contacts[:, 2:4], axis=-1) > 1.0
        reg = -0.5 * np.linalg.norm(self.actions - self.previous_actions, axis=-1)
        reg -= 0.0005 * np.linalg.norm(self.sim.actuator_force, axis=-1)
        reg -= 0.25 * (np.linalg.norm(foot_v, axis=-1) * contact).sum(-1)
        reg -= 0.001 * np.linalg.norm(jerk, axis=-1).sum(-1)
        clearance = np.maximum(0, 0.06 - s["body_pos"][:, 2:4, 2])
        clearance_reward = 3 * np.exp(
            -np.tanh(2 * np.linalg.norm(ref["body_vel"][:, 2:4, :2], axis=-1)) * clearance**2 / 0.0009
        ).sum(-1)
        command_error = np.c_[velocity[:, :2], omega[:, 2]] - self.commands
        cmd_reward = 5 * (
            np.exp(-np.square(command_error[:, :2]).sum(-1) / 0.25) + np.exp(-(command_error[:, 2] ** 2) / 0.25)
        )
        reward = (30 + reg + np.where(self.goal, cmd_reward, track + clearance_reward)) * self.dt
        height_error = np.abs(ref["body_pos"][:, 1, 2] - s["body_pos"][:, 1, 2])
        angle = np.arccos(np.clip((gravity * reference_gravity).sum(-1), -1, 1))
        failure = self.sensor("ground_depth")[:, 0] < -self.cfg.max_penetration
        failure |= np.linalg.norm(self.sensor("ground_peak"), axis=-1) > self.cfg.max_contact_force
        failure |= self.sensor("feet_collision")[:, 0] > 0
        failure |= np.where(self.goal, s["body_pos"][:, 1, 2] < 0.5, (height_error > 0.3) | (angle > 0.6))
        metrics = dict(
            joint_mae=np.abs(errors[3]).mean(-1),
            torso_velocity_error=np.linalg.norm(errors[0], axis=-1),
            torso_angular_velocity_error=np.linalg.norm(errors[1], axis=-1),
            root_orientation_error=np.linalg.norm(difference(rq, q), axis=-1),
            command_linear_error=np.linalg.norm(command_error[:, :2], axis=-1),
            command_yaw_error=np.abs(command_error[:, 2]),
            similarity=self.tracking_rewards[:, 4].copy(),
        )
        return reward, failure, metrics

    def update_curriculum(self, ids):
        ids = ids[~self.goal[ids]]
        if not len(ids):
            return
        self.sampler.update(self.bin_ids[ids], self.score_sum[ids], self.possible_steps[ids])
        if self.cfg.stage != "teacher":
            return
        for bin_id in np.unique(self.bin_ids[ids]):
            selected = ids[self.bin_ids[ids] == bin_id]
            similarity = np.clip(np.mean(self.score_sum[selected] / self.possible_steps[selected]), 0, 1)
            success = self.rng.random() < similarity
            if self.cfg.curriculum == "paper_literal":
                delta = 0.05 if success else -0.1
            else:
                delta = -0.1 if success else 0.05
            self.assistance_levels[bin_id] = np.clip(self.assistance_levels[bin_id] + delta, 0, self.cfg.beta_max)

    def step(self, actions):
        _, reward, done, info = super().step(actions)
        self.refresh_reference_commands()
        if self.cfg.stage == "finetune":
            self.command_timer -= self.dt
            ids = np.flatnonzero(self.goal & (self.command_timer <= 0))
            if len(ids):
                self.commands[ids] = (
                    self.fixed_command if self.fixed_command is not None else self.draw_commands(len(ids))
                )
                self.command_timer[ids] = self.cfg.command_interval
        self.obs = self.compute_observations()
        return self.obs, reward, done, info

    def set_command(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Command must be three finite velocities")
        self.fixed_command = command
        self.commands[self.goal] = command
        self.obs = self.compute_observations()

    def stand(self):
        """Evaluation reset independent of reference files and command-conditioned RSI."""
        self.reset()
        self.sim.qpos[:] = self.sim.model.qpos0
        self.sim.qpos[:, self.sim.qpos_ids] = self.nominal
        self.sim.qvel[:] = 0
        self.sim.ctrl[:] = self.nominal
        self.sim.forward()
        self.foot_velocity[:] = 0
        self.foot_acceleration[:] = 0
        self.obs = self.compute_observations()
        return self.obs

    def contract(self):
        result = super().contract()
        result.update(
            family="humble",
            version=1,
            action="nominal_plus_scaled_action",
            nominal=self.nominal.tolist(),
            critic_padding="goal_reference_zero",
            student_obs=96,
            teacher_obs=135,
            terrain="flat",
            curriculum=self.cfg.curriculum,
        )
        return result

    def close(self):
        super().close()
        if self._temporary:
            self._temporary.cleanup()

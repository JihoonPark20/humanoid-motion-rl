"""Flat-ground HuMBLE G1 configuration; paper Table S7 actuator values."""

from dataclasses import dataclass

import numpy as np

from zest.envs.g1.config import JOINT_NAMES, G1Config, motor_parameters


def joint_parameters():
    table = {
        "waist_roll": (150, 3, 0, 0.1),
        "waist_pitch": (150, 3, 0, 0.1),
        "waist_yaw": (100, 2, 0, 0.1),
        "hip_roll": (40, 3, 0, 0.2),
        "hip_pitch": (40, 3, -0.1, 0.2),
        "hip_yaw": (40, 3, 0, 0.1),
        "knee": (80, 2.5, 0.2, 0.2),
        "ankle_roll": (40, 2, 0, 0.05),
        "ankle_pitch": (40, 2, -0.1, 0.2),
        "shoulder_roll": (20, 1, 0.25, 0.1),
        "shoulder_pitch": (20, 1, 0.1, 0.1),
        "shoulder_yaw": (20, 1, 0, 0.1),
        "elbow": (30, 1, 1, 0.1),
        "wrist_roll": (20, 0.5, 0, 0.01),
        "wrist_pitch": (20, 0.5, 0, 0.01),
        "wrist_yaw": (20, 0.5, 0, 0.01),
    }
    values = []
    for name in JOINT_NAMES:
        key = name.removesuffix("_joint").removeprefix("left_").removeprefix("right_")
        p, d, nominal, scale = table[key]
        if name == "right_shoulder_roll_joint":
            nominal = -nominal
        values.append((p, d, nominal, scale))
    return np.array(values).T


@dataclass
class HumbleConfig(G1Config):
    stage: str = "teacher"
    beta_max: float = 0.75
    # Explicit interpretation of the inconsistent prose in Section S5.
    curriculum: str = "success_decay"
    goal_fraction: float = 0.5
    reference_seconds: float = 3.0
    goal_seconds: float = 8.0
    command_low: tuple = (-0.8, -0.6, -1.5)
    command_high: tuple = (1.2, 0.6, 1.5)
    command_interval: float = 3.0
    stop_probability: float = 0.15
    command_bin: float = 0.2
    reset_window: float = 0.2
    max_contact_force: float = 6000.0
    max_penetration: float = 0.03

    def motor_parameters(self):
        arm, _, _, effort, _ = motor_parameters()
        kp, kd, _, scale = joint_parameters()
        return arm, kp, kd, effort, scale

    def validate(self):
        super().validate()
        if self.stage not in ("teacher", "distill", "warmup", "finetune"):
            raise ValueError("Invalid HuMBLE stage")
        if self.curriculum not in ("success_decay", "paper_literal"):
            raise ValueError("Invalid assistance interpretation")
        if not 0 <= self.goal_fraction <= 1 or not 0 <= self.stop_probability <= 1:
            raise ValueError("Invalid task/stop fraction")
        if min(self.reference_seconds, self.goal_seconds, self.command_interval, self.command_bin) <= 0:
            raise ValueError("Durations and command bin must be positive")
        if self.reset_window < 0 or np.any(np.asarray(self.command_low) >= self.command_high):
            raise ValueError("Invalid reset window or command bounds")

"""G1 constants adapted from legged-gym-mujoco / Unitree RL Mjlab.

Motor parameters: unitree_rl_mjlab @ 1425b15f (Apache-2.0).
See licenses/ and resources/robots/g1/unitree_mjlab.json.
"""

from dataclasses import dataclass

import numpy as np

JOINT_NAMES = (
    [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]
)
BODY_NAMES = [
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "right_knee_link",
]
KEY_INDICES = np.arange(1, len(BODY_NAMES))


def reflected(rotors, gears):
    return rotors[0] * (gears[1] * gears[2]) ** 2 + rotors[1] * gears[2] ** 2 + rotors[2]


def motor_parameters():
    a5020 = reflected((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
    a14 = reflected((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
    a22 = reflected((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
    a4010 = reflected((0.068e-4, 0, 0), (1, 5, 5))
    armature, effort, scale = [], [], []
    for name in JOINT_NAMES:
        if "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
            a, e = 2 * a5020, 50
        elif "hip_roll" in name or "knee" in name:
            a, e = a22, 139
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            a, e = a14, 88
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            a, e = a4010, 5
        else:
            a, e = a5020, 25
        armature.append(a)
        effort.append(e)
        scale.append(0.05 if "wrist" in name else 0.2 if "hip" in name or "knee" in name else 0.1)
    armature = np.array(armature)
    omega = 2 * np.pi * 10  # Engineering choice; not a measured ZEST G1 actuator calibration.
    return armature, armature * omega**2, 2 * armature * omega, np.array(effort), np.array(scale)


@dataclass
class G1Config:
    num_envs: int = 64
    num_threads: int = 10
    seed: int = 42
    sim_dt: float = 0.004
    decimation: int = 5
    episode_seconds: float = 10.0
    dwell_seconds: float = 0.2
    adaptive_sampling: bool = True
    assistance: bool = True
    randomize: bool = True
    observation_noise: bool = True
    bin_seconds: float = 4.0
    ema_alpha: float = 0.005
    temperature_base: float = 1.0
    uniform_floor: float = 0.15
    beta_max: float = 0.6
    similarity_target: float = 0.8
    force_kp: float = 0.0
    force_kd: float = 10.0
    torque_kp: float = 200.0
    torque_kd: float = 10.0
    max_root_error: float = 1.0
    max_orientation_error: float = 1.5
    max_contact_force: float = 3000.0
    reset_position_noise: float = 0.02
    reset_joint_noise: float = 0.02
    reset_velocity_noise: float = 0.1

    def motor_parameters(self):
        return motor_parameters()

    @property
    def dt(self):
        return self.sim_dt * self.decimation

    def validate(self):
        if self.num_envs < 1 or self.num_threads < 1 or self.decimation < 1:
            raise ValueError("Environment count, thread count and decimation must be positive")
        if self.sim_dt <= 0 or self.episode_seconds < self.dt or self.dwell_seconds < 0:
            raise ValueError("Invalid simulation or episode duration")
        if not 0 < self.ema_alpha <= 1 or not 0 <= self.uniform_floor <= 1:
            raise ValueError("Invalid sampler coefficients")
        if not 0 <= self.beta_max < 1 or self.similarity_target <= 0 or self.temperature_base <= 0:
            raise ValueError("Invalid curriculum coefficients")

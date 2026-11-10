"""Validated robot-space references, FK, and continuous-time interpolation."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from zest.envs.g1.config import BODY_NAMES, JOINT_NAMES
from zest.utils.math import difference, inverse_rotate, slerp


def velocities(qpos, fps):
    v = np.gradient(qpos[:, :3], 1 / fps, axis=0)
    delta = difference(qpos[1:, 3:7], qpos[:-1, 3:7]) * fps
    omega = np.concatenate((delta[:1], (delta[:-1] + delta[1:]) / 2, delta[-1:]))
    joints = np.gradient(qpos[:, 7:], 1 / fps, axis=0)
    return np.concatenate((v, inverse_rotate(qpos[:, 3:7], omega), joints), axis=-1)


def validate_motion(qpos, fps, joint_names):
    if list(joint_names) != JOINT_NAMES:
        raise ValueError("Motion joint_names must exactly match G1's canonical 29-joint order")
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 3 or not np.isfinite(qpos).all():
        raise ValueError("Expected finite qpos with shape (frames >= 3, 36)")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1, atol=1e-4):
        raise ValueError("Root quaternions must be normalized wxyz")


class MotionClip:
    def __init__(self, path, model):
        self.path = Path(path).resolve()
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as data:
            if int(data["version"]) != 1 or str(data["quaternion_order"]) != "wxyz":
                raise ValueError("Unsupported motion format")
            qpos = data["qpos"].astype(np.float64)
            self.fps = float(data["fps"])
            validate_motion(qpos, self.fps, data["joint_names"].tolist())
        self.duration = (len(qpos) - 1) / self.fps
        joint_ids = [model.joint(n).id for n in JOINT_NAMES]
        pos_ids = model.jnt_qposadr[joint_ids]
        body_ids = [model.body(n).id for n in BODY_NAMES]
        positions = np.empty((len(qpos), len(body_ids), 3))
        rotations = np.empty((len(qpos), len(body_ids), 4))
        data = mujoco.MjData(model)
        for i, row in enumerate(qpos):
            data.qpos[:7] = row[:7]
            data.qpos[pos_ids] = row[7:]
            mujoco.mj_kinematics(model, data)
            positions[i] = data.xpos[body_ids]
            rotations[i] = data.xquat[body_ids]
        root_omega_delta = difference(qpos[1:, 3:7], qpos[:-1, 3:7]) * self.fps
        root_omega = np.concatenate(
            (root_omega_delta[:1], (root_omega_delta[:-1] + root_omega_delta[1:]) / 2, root_omega_delta[-1:])
        )
        root_v = np.gradient(qpos[:, :3], 1 / self.fps, axis=0)
        body_delta = difference(rotations[1:], rotations[:-1]) * self.fps
        body_omega = np.concatenate((body_delta[:1], (body_delta[:-1] + body_delta[1:]) / 2, body_delta[-1:]))
        body_vel = np.gradient(positions, 1 / self.fps, axis=0)
        self.values = dict(
            qpos=qpos,
            qvel=velocities(qpos, self.fps),
            body_pos=positions,
            body_quat=rotations,
            body_vel=body_vel,
            body_omega=body_omega,
            body_acc=np.gradient(body_vel, 1 / self.fps, axis=0),
            body_alpha=np.gradient(body_omega, 1 / self.fps, axis=0),
            root_vel=root_v,
            root_omega=root_omega,
            root_acc=np.gradient(root_v, 1 / self.fps, axis=0),
            root_alpha=np.gradient(root_omega, 1 / self.fps, axis=0),
        )

    def sample(self, times):
        index = np.clip(np.asarray(times) * self.fps, 0, len(self.values["qpos"]) - 1)
        lo = np.floor(index).astype(int)
        hi = np.minimum(lo + 1, len(self.values["qpos"]) - 1)
        f = index - lo
        out = {}
        for name, value in self.values.items():
            weight = f.reshape((-1,) + (1,) * (value.ndim - 1))
            out[name] = (1 - weight) * value[lo] + weight * value[hi]
        out["qpos"][:, 3:7] = slerp(self.values["qpos"][lo, 3:7], self.values["qpos"][hi, 3:7], f[:, None])
        out["body_quat"] = slerp(self.values["body_quat"][lo], self.values["body_quat"][hi], f[:, None, None])
        # Hold the final pose during dwell; do not extrapolate terminal velocities.
        ended = np.asarray(times) >= self.duration
        for name in (
            "qvel",
            "body_vel",
            "body_omega",
            "body_acc",
            "body_alpha",
            "root_vel",
            "root_omega",
            "root_acc",
            "root_alpha",
        ):
            out[name][ended] = 0
        return out


class MotionLibrary:
    def __init__(self, paths, model):
        if not paths:
            raise ValueError("At least one prepared motion is required")
        self.clips = [MotionClip(path, model) for path in paths]
        self.durations = np.array([c.duration for c in self.clips])
        self.manifest = [{"path": str(c.path), "sha256": c.digest} for c in self.clips]

    def sample(self, clip_ids, times):
        out = {}
        for clip_id in np.unique(clip_ids):
            mask = clip_ids == clip_id
            values = self.clips[clip_id].sample(times[mask])
            for key, value in values.items():
                if key not in out:
                    out[key] = np.empty((len(times),) + value.shape[1:])
                out[key][mask] = value
        return out

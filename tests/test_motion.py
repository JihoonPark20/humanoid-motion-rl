import numpy as np
import pytest

from zest.envs.g1.config import JOINT_NAMES
from zest.motion.library import MotionClip, validate_motion, velocities
from zest.utils.math import difference, inverse_rotate, rotate, slerp


def test_antipodal_quaternions_and_pi_rotation():
    a = np.array([[1.0, 0, 0, 0]])
    np.testing.assert_allclose(slerp(a, -a, np.array([[0.5]])), a)
    q = np.array([[0.0, 0, 0, 1]])
    np.testing.assert_allclose(np.linalg.norm(difference(q, a), axis=-1), np.pi)
    v = np.array([[1.0, 2, 3]])
    np.testing.assert_allclose(inverse_rotate(q, rotate(q, v)), v)


def test_motion_sampling_joint_order_and_terminal_dwell(tmp_path, model):
    qpos = np.repeat(model.qpos0[None], 51, axis=0)
    qpos[:, 0] = np.arange(51) / 50
    qpos[:, 3:7] = [np.cos(0.25), 0, 0, np.sin(0.25)]
    qpos[:, 7:] = np.arange(29) * 0.01
    path = tmp_path / "walk.npz"
    np.savez(path, version=1, qpos=qpos, fps=50.0, joint_names=JOINT_NAMES, quaternion_order="wxyz")
    clip = MotionClip(path, model)
    sample = clip.sample(np.array([0.5]))
    assert clip.duration == 1
    np.testing.assert_allclose(sample["qpos"][0, 3:7], [np.cos(0.25), 0, 0, np.sin(0.25)])
    np.testing.assert_allclose(sample["qpos"][0, 7:], np.arange(29) * 0.01)
    np.testing.assert_allclose(sample["root_vel"], [[1, 0, 0]], atol=1e-10)
    np.testing.assert_allclose(clip.sample(np.array([1.1]))["qvel"], 0)


def test_motion_validation_rejects_wrong_order_and_quaternions(model):
    q = np.repeat(model.qpos0[None], 3, axis=0)
    with pytest.raises(ValueError, match="joint_names"):
        validate_motion(q, 50, JOINT_NAMES[::-1])
    q[:, 3:7] = 0
    with pytest.raises(ValueError, match="normalized"):
        validate_motion(q, 50, JOINT_NAMES)


def test_fk_and_world_vs_local_angular_velocity(motion, model):
    clip = MotionClip(motion, model)
    np.testing.assert_allclose(clip.values["body_pos"][:, 0], clip.values["qpos"][:, :3])
    q = clip.values["qpos"].copy()
    angle = np.arange(len(q)) * 0.02
    q[:, 3:7] = np.stack((np.cos(angle / 2), np.zeros(len(q)), np.zeros(len(q)), np.sin(angle / 2)), axis=-1)
    np.testing.assert_allclose(velocities(q, 50)[:, 3:6], np.tile([0, 0, 1], (len(q), 1)), atol=1e-10)

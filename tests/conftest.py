import numpy as np
import pytest

from zest.envs.g1.config import JOINT_NAMES, G1Config
from zest.sim.mujoco_batch import load_model


@pytest.fixture(scope="session")
def model():
    return load_model(G1Config(randomize=False))


@pytest.fixture
def motion(tmp_path, model):
    qpos = np.repeat(model.qpos0[None], 11, axis=0)
    path = tmp_path / "stand.npz"
    np.savez(path, version=1, qpos=qpos, fps=50.0, joint_names=JOINT_NAMES, quaternion_order="wxyz")
    return path

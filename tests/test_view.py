import copy

import numpy as np
import pytest
import torch

from zest.envs.g1.config import G1Config
from zest.envs.g1.humble import G1Humble
from zest.envs.g1.humble_config import HumbleConfig
from zest.envs.g1.tracking import G1Tracking
from zest.scripts.view import Playback
from zest.utils.humble_runner import HumbleRunner, config_for
from zest.utils.runner import ZestRunner, training_config


def test_reference_scrubbing_does_not_step_physics(motion):
    with np.load(motion) as archive:
        data = dict(archive)
    data["qpos"][:, 0] = np.arange(len(data["qpos"])) * 0.01
    np.savez(motion, **data)
    playback = Playback(motion=motion)
    try:
        simulator = playback.env.sim.qpos.copy()
        playback.advance()
        assert playback.data.qpos[0] == pytest.approx(0.01)
        playback.time = 0.1
        playback.refresh()
        assert playback.data.qpos[0] == pytest.approx(0.05)
        np.testing.assert_array_equal(simulator, playback.env.sim.qpos)
        playback.reset()
        assert playback.time == 0 and playback.data.qpos[0] == 0
    finally:
        playback.close()


@pytest.mark.parametrize("family", ["zest", "humble"])
def test_policy_viewer_preserves_checkpoint_parameters(motion, tmp_path, family):
    torch.set_num_threads(2)
    if family == "humble":
        with np.load(motion) as archive:
            data = dict(archive)
        data["commands"] = np.zeros((len(data["qpos"]), 3))
        np.savez(motion, **data)
        env = G1Humble([motion], HumbleConfig(num_envs=1, num_threads=1))
        runner = HumbleRunner(env, config_for("teacher", tiny=True))
    else:
        env = G1Tracking([motion], G1Config(num_envs=1, num_threads=1))
        cfg = training_config()
        cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [16]
        runner = ZestRunner(env, cfg)
    checkpoint = tmp_path / f"{family}.pt"
    runner.save(checkpoint)  # Fixture only: no optimization or training call.
    env.close()
    playback = Playback(checkpoint=checkpoint)
    try:
        before = copy.deepcopy(playback.policy.state_dict())
        for _ in range(3):
            playback.advance()
        assert all(torch.equal(value, playback.policy.state_dict()[key]) for key, value in before.items())
        assert np.isfinite(playback.data.qpos).all()
        assert not playback.env.beta.any()
    finally:
        playback.close()

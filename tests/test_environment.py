import numpy as np
import pytest
import torch

from zest.envs.g1.config import G1Config
from zest.envs.g1.tracking import G1Tracking
from zest.sim.mujoco_batch import MujocoBatch


def config(**kwargs):
    return G1Config(num_envs=2, num_threads=1, randomize=False, observation_noise=False, **kwargs)


def test_ground_reaction_points_upwards():
    sim = MujocoBatch(config(), np.random.default_rng(0))
    try:
        for _ in range(50):
            sim.step(np.zeros((2, 29)))
        sim.forward()
        force = sim.contacts[:, 2:4].sum(axis=1)
        assert np.all(force[:, 2] > 0)
        np.testing.assert_allclose(force[:, 2], sim.mass * 9.81, rtol=0.1)
    finally:
        sim.close()


def test_actor_has_no_measured_root_position_or_velocity(motion):
    env = G1Tracking([motion], config(), evaluation=True)
    try:
        before = env.get_observations().clone()
        env.sim.qpos[:, :2] += 10
        env.sim.qvel[:, :3] += 2
        env.sim.forward()
        after = env.compute_observations()
        torch.testing.assert_close(before["policy"], after["policy"])
        assert not torch.allclose(before["critic"], after["critic"])
        assert before["policy"].shape == (2, 132)
    finally:
        env.close()


def test_residual_target_and_observation_snapshot(motion):
    env = G1Tracking([motion], config(), evaluation=True)
    try:
        obs = env.get_observations()
        original = obs.clone()
        target = env.library.sample(env.clip_ids, env.times + env.dt)["qpos"][:, 7:]
        env.step(torch.full((2, 29), 0.1))
        np.testing.assert_allclose(env.sim.ctrl, target + 0.1 * env.scale, atol=1e-8)
        torch.testing.assert_close(obs, original)
        np.testing.assert_allclose(env.sim.xfrc_applied, 0)
    finally:
        env.close()


def test_timeout_terminal_observation_and_failure_precedence(motion):
    env = G1Tracking([motion], config(episode_seconds=0.02))
    try:
        _, _, done, extras = env.step(torch.zeros(2, 29))
        assert done.all() and extras["time_outs"].all()
        assert "terminal_observations" in extras
        assert (env.episode_length_buf == 0).all()
        env.sim.qpos[:, 0] += 5
        env.sim.forward()
        _, _, done, extras = env.step(torch.zeros(2, 29))
        assert done.all() and not extras["time_outs"].any()
    finally:
        env.close()


def test_evaluation_starts_at_zero_and_holds_zero_assistance(motion):
    env = G1Tracking([motion], config(episode_seconds=0.02), evaluation=True)
    try:
        np.testing.assert_allclose(env.times, 0)
        assert (env.possible_steps > env.max_episode_length).all()
        for _ in range(3):
            env.step(torch.zeros(2, 29))
            np.testing.assert_allclose(env.wrench, 0)
            np.testing.assert_allclose(env.beta, 0)
        with pytest.raises(ValueError):
            env.step(torch.zeros(1, 29))
    finally:
        env.close()

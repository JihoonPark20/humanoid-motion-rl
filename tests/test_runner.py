import numpy as np
import pytest
import torch

from zest.envs.g1.config import G1Config
from zest.envs.g1.tracking import G1Tracking
from zest.utils.runner import ZestRunner, training_config


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_ppo_bootstrap_update_checkpoint_and_resume(motion, tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    torch.manual_seed(12)
    torch.set_num_threads(2)
    env = G1Tracking(
        [motion], G1Config(num_envs=2, num_threads=1, episode_seconds=0.02, randomize=False, observation_noise=False)
    )
    cfg = training_config()
    cfg["num_steps_per_env"] = 2
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [32, 16]
    try:
        runner = ZestRunner(env, cfg, device=device)
        old = next(runner.alg.actor.parameters()).detach().clone()
        with torch.inference_mode():
            obs = env.get_observations().to(device)
            action = runner.alg.act(obs)
            obs, rewards, dones, extras = env.step(action)
            terminal_values = runner.alg.critic(extras["terminal_observations"].to(device)).squeeze(-1)
            expected = rewards.to(device) + runner.alg.gamma * terminal_values * extras["time_outs"].to(device)
            runner.alg.process_env_step(obs.to(device), rewards.to(device), dones.to(device), extras)
            torch.testing.assert_close(runner.alg.storage.rewards[0].squeeze(-1), expected)
            runner.alg.storage.clear()
        runner.learn(2)
        assert not torch.equal(old, next(runner.alg.actor.parameters()))
        assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
        checkpoint = tmp_path / f"{device}.pt"
        failure = env.sampler.failure.copy()
        runner.save(checkpoint)
        env.sampler.failure[:] = 0
        runner.load(checkpoint)
        np.testing.assert_allclose(env.sampler.failure, failure)
        assert runner.current_learning_iteration == 2
        runner.learn(1)
        assert runner.current_learning_iteration == 2
    finally:
        env.close()

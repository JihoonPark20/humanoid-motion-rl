import numpy as np
import torch

from zest.envs.g1.config import G1Config
from zest.envs.g1.tracking import G1Tracking
from zest.scripts.evaluate import evaluate


def test_evaluation_excludes_autoreset_episodes(motion):
    cfg = G1Config(num_envs=2, num_threads=1, randomize=False, observation_noise=False)
    env = G1Tracking([motion], cfg, evaluation=True)
    original_step = env.step
    first = True

    def step(actions):
        nonlocal first
        if first:
            env.sim.qpos[0, 0] += 5
            env.sim.forward()
            first = False
        return original_step(actions)

    env.step = step
    try:
        rows = evaluate(env, lambda obs: torch.zeros(2, 29), "cpu")
        assert rows[0]["steps"] == 1 and not rows[0]["success"]
        assert rows[1]["steps"] > 1 and rows[1]["success"]
        assert rows[1]["completion"] == 1
        assert all(np.isfinite(row["joint_mae"]) for row in rows)
    finally:
        env.close()

import numpy as np

from zest.envs.g1.config import G1Config
from zest.envs.g1.tracking import assistive_wrench
from zest.motion.sampler import AdaptiveSampler


def test_sampler_coverage_short_clips_and_difficulty():
    cfg = G1Config()
    sampler = AdaptiveSampler([1.0, 5.0], cfg)
    sampler.failure[:] = 0
    sampler.failure[-1] = 1
    probs = sampler.probabilities()
    assert probs[-1] > probs[0]
    assert np.all(probs >= cfg.uniform_floor / len(probs))
    np.testing.assert_allclose(probs.sum(), 1)
    clips, times, bins, beta = sampler.sample(1000, np.random.default_rng(3))
    assert np.all(times < sampler.durations[clips])
    assert np.all((beta >= 0) & (beta <= 0.6))
    assert sampler.visits.sum() == 1000
    np.testing.assert_allclose(beta[sampler.failure[bins] == 0], 0)


def test_early_failure_counts_missing_steps_and_duplicate_bins():
    s = AdaptiveSampler([4.0], G1Config(ema_alpha=0.5))
    s.update(np.array([0, 0]), np.array([10.0, 0.0]), np.array([10, 10]))
    np.testing.assert_allclose(s.failure, 0.625)
    restored = AdaptiveSampler([4.0], G1Config())
    restored.load_state_dict(s.state_dict())
    np.testing.assert_allclose(restored.failure, s.failure)


def test_wrench_gravity_compensation_and_rotation_equivariance():
    cfg = G1Config()
    q = np.array([[1.0, 0, 0, 0]])
    p, v = np.zeros((1, 3)), np.zeros((1, 3))
    ref = dict(qpos=np.c_[p, q], root_acc=v.copy(), root_alpha=v.copy(), root_vel=v.copy(), root_omega=v.copy())
    wrench = assistive_wrench(ref, p, q, v, v, 10.0, np.eye(3), np.array([0.1, 0, 0]), cfg)
    np.testing.assert_allclose(wrench, [[0, 0, 98.1, 0, -9.81, 0]])
    ref["root_alpha"][:] = [1, 2, 3]
    np.testing.assert_allclose(assistive_wrench(ref, p, q, v, v, 10.0, np.eye(3), np.zeros(3), cfg)[:, 3:], [[1, 2, 3]])

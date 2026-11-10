"""ZEST S5-S6: difficulty-weighted RSI and episode-constant assistance."""

import numpy as np


class AdaptiveSampler:
    def __init__(self, durations, cfg):
        self.cfg = cfg
        self.durations = np.asarray(durations)
        self.width = min(cfg.bin_seconds, self.durations.min())
        if self.width <= 0:
            raise ValueError("Bin width and clip durations must be positive")
        self.clip_ids = np.repeat(np.arange(len(durations)), np.ceil(self.durations / self.width).astype(int))
        self.starts = np.concatenate([np.arange(int(np.ceil(d / self.width))) * self.width for d in durations])
        self.ends = np.minimum(self.starts + self.width, self.durations[self.clip_ids])
        self.failure = np.ones(len(self.starts))
        self.visits = np.zeros(len(self.starts), dtype=np.int64)

    def probabilities(self):
        if not self.cfg.adaptive_sampling:
            return np.full(len(self.failure), 1 / len(self.failure))
        temperature = self.cfg.temperature_base / np.log1p(len(self.failure))
        logits = (self.failure - self.failure.max()) / temperature
        p = np.exp(logits)
        p /= p.sum()
        return (1 - self.cfg.uniform_floor) * p + self.cfg.uniform_floor / len(p)

    def sample(self, count, rng):
        bins = rng.choice(len(self.failure), count, p=self.probabilities())
        times = rng.uniform(self.starts[bins], self.ends[bins])
        np.add.at(self.visits, bins, 1)
        beta = np.clip(1 - (1 - self.failure[bins]) / self.cfg.similarity_target, 0, self.cfg.beta_max)
        if not self.cfg.assistance:
            beta[:] = 0
        return self.clip_ids[bins], times, bins, beta

    def update(self, bins, score_sum, possible_steps):
        # Normalize by the possible episode length, including steps lost to early failure.
        scores = np.clip(score_sum / np.maximum(possible_steps, 1), 0, 1)
        for b in np.unique(bins):
            values = 1 - scores[bins == b]
            alpha = 1 - (1 - self.cfg.ema_alpha) ** len(values)
            self.failure[b] = (1 - alpha) * self.failure[b] + alpha * values.mean()

    def state_dict(self):
        return {"failure": self.failure.copy(), "visits": self.visits.copy()}

    def load_state_dict(self, state):
        for name in ("failure", "visits"):
            value = np.asarray(state[name])
            if value.shape != self.failure.shape or not np.isfinite(value).all():
                raise ValueError("Sampler state does not match the motion library")
            getattr(self, name)[:] = value

"""Versioned command annotation and command-conditioned reference initialization."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from zest.motion.library import MotionLibrary
from zest.utils.math import inverse_rotate


def annotate(values, fps, cutoff=3.0, anticipation=0.15):
    """Engineering reconstruction: low-pass velocity + bounded acceleration-dependent lookahead.

    The paper does not publish filter/shift coefficients. anticipation is a maximum
    shift in seconds, not an assertion of the original authors' implementation.
    """
    if not 0 < cutoff < fps / 2 or anticipation < 0:
        raise ValueError("Invalid annotation filter/anticipation")
    q = values["body_quat"][:, 1]
    v = inverse_rotate(q, values["body_vel"][:, 1])
    w = inverse_rotate(q, values["body_omega"][:, 1])
    raw = np.c_[v[:, :2], w[:, 2]]
    filtered = sosfiltfilt(butter(2, cutoff, fs=fps, output="sos"), raw, axis=0, padlen=min(12, len(raw) - 1))
    acceleration = np.linalg.norm(np.gradient(filtered, 1 / fps, axis=0), axis=-1)
    times = np.arange(len(raw)) / fps
    shifted = times + anticipation * np.tanh(acceleration)
    return np.stack([np.interp(shifted, times, filtered[:, i]) for i in range(3)], axis=-1).astype(np.float32)


class CommandLibrary(MotionLibrary):
    def __init__(self, paths, model, bin_width=0.2):
        super().__init__(paths, model)
        self.bin_width = bin_width
        self.bins = {}
        for k, clip in enumerate(self.clips):
            with np.load(clip.path, allow_pickle=False) as source:
                if "commands" not in source:
                    raise ValueError(f"{clip.path}: use humble-data to prepare command annotations")
                commands = source["commands"]
            if commands.shape != (len(clip.values["qpos"]), 3) or not np.isfinite(commands).all():
                raise ValueError("Invalid motion commands")
            clip.values["commands"] = commands
            # Bound the resident memory of the full library; FK/differentiation was float64.
            clip.values = {name: value.astype(np.float32) for name, value in clip.values.items()}
            for frame in range(0, len(commands), max(1, int(clip.fps / 10))):
                key = tuple(np.floor(commands[frame] / bin_width).astype(int))
                self.bins.setdefault(key, []).append((k, frame / clip.fps))
        self.keys = np.array(list(self.bins))

    def initial_states(self, commands, rng, window=0.2):
        result = []
        for command in commands:
            key = tuple(np.floor(command / self.bin_width).astype(int))
            if key not in self.bins:
                key = tuple(self.keys[np.argmin(np.sum((self.keys - key) ** 2, axis=-1))])
            candidates = self.bins[key]
            clip, time = candidates[rng.integers(len(candidates))]
            time = np.clip(time + rng.uniform(-window, window), 0, max(0, self.durations[clip] - 0.1))
            result.append((clip, time))
        result = np.asarray(result)
        return result[:, 0].astype(int), result[:, 1]

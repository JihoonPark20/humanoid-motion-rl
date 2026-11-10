"""Compact training progress without changing machine-readable metrics."""

import os
import shutil
import sys
import time


def duration(seconds):
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class TrainingConsole:
    def __init__(self, start, iterations, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.live = self.stream.isatty() and os.environ.get("TERM") != "dumb"
        self.start, self.iterations = start, iterations
        self.begin = time.monotonic()
        self.visible = False

    def update(self, metrics, report=False):
        iteration = int(metrics["iteration"])
        done = iteration - self.start
        if not self.live and not (report or done == 1 or done == self.iterations or done % 100 == 0):
            return
        elapsed = time.monotonic() - self.begin
        eta = elapsed / max(done, 1) * max(self.iterations - done, 0)
        progress = min(max(done / self.iterations, 0), 1)
        filled = int(progress * 10)
        fields = [
            f"[{'=' * filled}{'-' * (10 - filled)}] {progress:5.1%}",
            f"iter {iteration}/{self.start + self.iterations}",
            f"ETA {duration(eta)}",
        ]
        for key, label, spec in (
            ("Episode/reward", "reward", ".1f"),
            ("Episode/similarity", "sim", ".3f"),
            ("Curriculum/beta", "assist", ".2f"),
            ("distillation_mse", "MSE", ".4g"),
            ("transitions_per_second", "steps/s", ".0f"),
        ):
            if key in metrics:
                fields.append(f"{label} {metrics[key]:{spec}}")
        if self.live:
            width = max(1, shutil.get_terminal_size().columns - 1)
            line = fields.pop(0)
            for field in fields:
                if len(line) + len(field) + 3 > width:
                    break
                line += f" | {field}"
            self.stream.write("\r\x1b[2K" + line[:width])
            self.visible = True
        else:
            self.stream.write(" | ".join(fields) + "\n")
        self.stream.flush()

    def note(self, message):
        if self.visible:
            self.stream.write("\r\x1b[2K")
            self.visible = False
        self.stream.write(message + "\n")
        self.stream.flush()

    def finish(self, reason, iteration, checkpoint):
        self.note(
            f"{'Completed' if reason == 'iteration_limit' else 'Stopped'} at iteration {iteration} "
            f"({duration(time.monotonic() - self.begin)} this session).\nCheckpoint: {checkpoint}"
        )

    def close(self):
        if self.visible:
            self.stream.write("\n")
            self.stream.flush()
            self.visible = False

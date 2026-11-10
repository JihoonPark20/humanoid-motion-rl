import io
import os
import unittest
from unittest.mock import patch

from zest.utils.console import TrainingConsole


class Terminal(io.StringIO):
    def isatty(self):
        return True


class TestTrainingConsole(unittest.TestCase):
    def test_redirected_output_is_sparse_plain_text_and_keeps_evaluation(self):
        stream = io.StringIO()
        console = TrainingConsole(0, 200, stream)
        for iteration in (1, 2, 99, 100, 150, 200):
            console.update({"iteration": iteration}, report=iteration == 150)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(f"iter {n}/200" in line for n, line in zip((1, 100, 150, 200), lines)))
        self.assertNotIn("\x1b", stream.getvalue())
        self.assertNotIn("\r", stream.getvalue())

    def test_live_output_fits_width_and_leaves_clean_final_message(self):
        stream = Terminal()
        with (
            patch.dict(os.environ, {"TERM": "xterm"}),
            patch("zest.utils.console.shutil.get_terminal_size", return_value=os.terminal_size((60, 24))),
        ):
            console = TrainingConsole(0, 10, stream)
            for iteration in (1, 2):
                console.update({"iteration": iteration, "Episode/reward": 123.4, "Episode/similarity": 0.8})
            frames = stream.getvalue().split("\r\x1b[2K")[1:]
            self.assertEqual(len(frames), 2)
            self.assertTrue(all(len(frame) < 60 and "\n" not in frame for frame in frames))
            console.finish("signal", 2, "logs/run/latest.pt")
            console.close()
        self.assertIn("Stopped at iteration 2", stream.getvalue())
        self.assertTrue(stream.getvalue().endswith("Checkpoint: logs/run/latest.pt\n"))

    def test_resume_eta_uses_this_session_and_missing_episode_metrics_are_omitted(self):
        stream = io.StringIO()
        with patch("zest.utils.console.time.monotonic", side_effect=[100.0, 120.0]):
            console = TrainingConsole(1000, 100, stream)
            console.update({"iteration": 1050, "distillation_mse": 0.0123}, report=True)
        line = stream.getvalue()
        self.assertIn("50.0%", line)
        self.assertIn("iter 1050/1100", line)
        self.assertIn("ETA 00:20", line)
        self.assertIn("MSE 0.0123", line)
        self.assertNotIn("reward", line)


if __name__ == "__main__":
    unittest.main()

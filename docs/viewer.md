# Viser playback

`zest-view` displays the existing MuJoCo G1 model through [mjviser](https://github.com/mujocolab/mjviser) and [Viser](https://viser.studio/). It supports both ZEST and HuMBLE checkpoints. This is a playback process: it never updates a policy or launches a training job.

Requires Python 3.11–3.13 for the optional viewer dependencies. Core training remains compatible with the project's existing Python range.

```bash
uv sync --extra viewer

# Reference motion only, without a checkpoint.
uv run --extra viewer zest-view --motion data/hold/S0_normal_walk_1.npz

# Teacher checkpoint: open paused, with Reference/Policy selection.
uv run --extra viewer zest-view \
  --checkpoint PATH_TO_TEACHER.pt --reference --paused

# A trained student: velocity sliders; reference files are unnecessary.
uv run --extra viewer zest-view --checkpoint PATH_TO_STUDENT.pt
```

Open **http://127.0.0.1:8080**. Unlike the native macOS viewer, this uses regular Python and does not require `mjpython`. Use `--port 8081` if another viewer already occupies 8080. `--host` defaults to the local machine only. Ctrl+C stops the viewer.

The Playback panel provides:

- **Mode:** kinematic reference or policy execution, when a checkpoint and reference are available.
- **Motion:** select a checkpoint's reference clip, or override it with `--motion FILE`.
- **Playing / Reset / Speed:** pause, restart and change playback speed without changing the simulator timestep.
- **Time:** scrub references while paused; disabled during policy execution.
- **Follow robot:** keep the camera centered on the moving robot.
- **Velocity command:** forward/lateral/yaw commands for a HuMBLE student loaded without a reference override.

The scene controls expose MuJoCo geometry, contact and frame visualization. Policy playback always disables assistive forces. Untrained checkpoints can fall and reset; the failure counter makes those resets visible. Teacher policies remain reference-conditioned; they do not receive free velocity controls.

Checkpoints are loaded once when the viewer starts. Restart it to inspect a newer checkpoint after training. Training metrics are available separately through `uv run tensorboard --logdir logs/humble`.

GUI requests are queued and applied by the simulation thread. Physics and policy inference do not run in GUI callback threads. Reference scrubbing recomputes forward kinematics without stepping physics. Tests cover reference timing and unchanged model/normalization parameters during ZEST and HuMBLE playback.

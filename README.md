# ZEST · MuJoCo

**English** | [한국어](README.ko.md)

Unofficial implementation of [ZEST](https://arxiv.org/abs/2602.00401) for **Unitree G1 (29 DoF)**, adapted from `legged-gym-mujoco`. MuJoCo physics runs on CPU; PPO runs on **MPS, CUDA, or CPU**.

| Paper | Idea | Workspace status |
| --- | --- | --- |
| [ZEST](https://arxiv.org/abs/2602.00401) | Learn reference-motion tracking with adaptive sampling and assistive forces. | Implemented with documented deviations; convergence unverified. |
| [HuMBLE](https://arxiv.org/abs/2610.10489) | Distill a motion-tracking teacher into a velocity-command policy, then refine it with multi-task RL. | Flat-ground stages implemented; convergence unverified. |

[HOLD](https://github.com/rai-opensource/hold)'s G1 locomotion data is supported for HuMBLE. The quick start below uses LAFAN1 for ZEST; see the [HuMBLE guide](docs/humble.md) for data, all four training stages, playback and resuming training.

## Quick start

Python 3.10–3.13. On Linux/NVIDIA, install a CUDA-enabled [PyTorch](https://pytorch.org/get-started/locally/) build first.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Download and prepare a 10-second Unitree LAFAN1 G1 reference.
zest-data --seconds 10
zest-train --motions data/lafan1/walk1_subject1.npz --device mps
```

Use `--device cuda` on NVIDIA or `--device auto` for automatic selection. Defaults: 64 environments, 10 CPU threads, 1,000 PPO iterations. Adjust with `--num-envs`, `--num-threads`, and `--iterations`.

## Replay and evaluate

Replace `PATH` with a checkpoint under `logs/g1/`.

```bash
# Browser viewer (Python 3.11+): http://127.0.0.1:8080
python -m pip install -e '.[viewer]'
zest-view --checkpoint PATH
zest-evaluate --checkpoint PATH
zest-train --resume PATH --iterations 1000
```

Evaluation starts at frame zero and disables assistance. JSON/CSV results are saved beside the checkpoint. Add `--randomized` for noise, pushes, and dynamics variation; `--motions` selects held-out clips. Preview references with `zest-view --motion FILE`; terminal-only playback uses `zest-play --checkpoint PATH --headless`.

## Scope

ZEST tracking and HuMBLE teacher, distillation, critic adaptation and multi-task PPO, with replay/evaluation. **CPU/MPS smoke-tested; CUDA unverified.** Short validation runs do not establish convergence or paper-level performance. Rough terrain, hardware deployment and human-motion retargeting are outside this version.

[Data](docs/data.md) · [Implementation](docs/implementation.md) · [Viser](docs/viewer.md) · [HuMBLE review](docs/humble-review.md) · [Validation](docs/validation.md) · [Credits](docs/credits.md)

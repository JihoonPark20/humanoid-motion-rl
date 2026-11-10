# ZEST · HuMBLE · MuJoCo

**English** | [한국어](README.ko.md)

Unofficial implementations for **Unitree G1 (29 DoF)**, adapted from `legged-gym-mujoco`. MuJoCo physics runs on CPU; learning runs on **MPS, CUDA, or CPU**.

| Paper | Idea | Status |
| --- | --- | --- |
| [ZEST](https://arxiv.org/abs/2602.00401) | Reference-motion tracking with adaptive sampling and assistive forces. | Implemented with documented deviations; convergence unverified. |
| [HuMBLE](https://arxiv.org/abs/2610.10489) | Distill a tracking teacher into a velocity-command policy, then refine it with multi-task RL. | Flat-ground stages implemented; convergence unverified. |

Both use **[HOLD](https://github.com/rai-opensource/hold) G1 locomotion** by default. HOLD is the dataset released by RAI Institute and Boston Dynamics and used in HuMBLE. Using it for ZEST is this project's choice.

## Setup and data

Python 3.10–3.13; Viser requires 3.11+. On Linux/NVIDIA, install a CUDA-enabled [PyTorch](https://pytorch.org/get-started/locally/) build first.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Prepare the first 30 seconds of HOLD's S0_normal_walk_1.
zest-data --seconds 30 --height-offset 0.005
```

The explicit 5 mm offset addresses this walk's floor penetration in the local G1 model; it is recorded in metadata. This single clip is a starting example. See [data preparation](docs/data.md) for larger subsets. Existing converted files are never overwritten.

## Train

Choose ZEST tracking or the HuMBLE pipeline:

```bash
# ZEST
zest-train --motions data/hold/*.npz --device mps --iterations 10000

# HuMBLE: teacher → distillation → critic adaptation → multi-task PPO
humble-train --stage teacher --motions data/hold/*.npz --config configs/humble-flat.json --device mps --iterations 10000
humble-evaluate --checkpoint TEACHER.pt --mode reference
humble-train --stage distill --init TEACHER.pt --device mps --iterations 20000
humble-train --stage warmup --init PRIOR.pt --device mps --iterations 5000
humble-train --stage finetune --init WARMUP.pt --device mps --iterations 20000
```

Replace `TEACHER.pt`, `PRIOR.pt` and `WARMUP.pt` with the previous stage's `latest.pt` under `logs/humble/`. Check unassisted teacher tracking before distillation. Iteration counts are examples, not convergence guarantees. Use `--device cuda` on NVIDIA; adjust parallelism with `--num-envs` and `--num-threads`. See the [HuMBLE guide](docs/humble.md) for stage details and the eight-clip data preparation example.

## View, evaluate and resume

Replace `PATH` with a checkpoint from the corresponding learner (`logs/g1/` or `logs/humble/`).

```bash
python -m pip install -e '.[viewer]'
zest-view --checkpoint PATH  # http://127.0.0.1:8080; supports both learners

# ZEST
zest-evaluate --checkpoint PATH
zest-train --resume PATH --device mps --iterations 10000

# HuMBLE student (use --mode reference for a teacher)
humble-evaluate --checkpoint PATH --mode commands
humble-train --resume PATH --device mps --iterations 10000
```

Evaluation disables assistance. Preview a reference with `zest-view --motion data/hold/S0_normal_walk_1.npz`. HuMBLE saves on Ctrl+C; resumed iteration counts are additional updates.

**CPU/MPS smoke-tested; CUDA unverified.** Paper-level performance, rough terrain and hardware deployment remain outside the validated scope.

[Data](docs/data.md) · [Implementation](docs/implementation.md) · [Viser](docs/viewer.md) · [HuMBLE review](docs/humble-review.md) · [Validation](docs/validation.md) · [Credits](docs/credits.md)

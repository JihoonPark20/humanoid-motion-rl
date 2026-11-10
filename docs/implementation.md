# Implementation notes

This is an independent MuJoCo implementation of [ZEST, arXiv:2602.00401v1](https://arxiv.org/html/2602.00401v1). The original uses Isaac Lab. Equations and tables guide this implementation; unpublished author code, original reference motions, and calibrated actuator parameters are not available in this workspace.

## Structure

```text
zest/
  envs/g1/       # robot configuration and motion-tracking VecEnv
  sim/          # CPU MuJoCo batch model construction
  motion/       # reference interpolation, FK, adaptive sampler
  scripts/      # data preparation, train, play, evaluate
  utils/        # rotation math and rsl_rl integration
  resources/    # G1 XML, meshes, asset provenance
tests/          # math, data, MDP, PPO and device checks
```

The `legged-gym-mujoco` separation between environments, simulation, scripts and utilities is retained. Its G1 asset, actuator specifications and CPU batch design are reused. Go2, locomotion rewards, SAC/FlashSAC and their collectors are not required by this implementation.

## Paper correspondence

| Component | Implementation |
| --- | --- |
| Actor, Table S3 | Torso angular velocity/gravity, joint angles/velocities, previous action; next-step reference height, local linear/angular velocity, gravity, joint angles. 132 values for G1. No measured global root state or observation history. |
| Critic, Table S3 | Actor input plus root velocity/height, root/keybody contact forces, local keybody positions/velocities, assistance wrench/scale, seven tracking terms and scalar phase. 217 values. |
| Residual action | Next reference joint angles + per-joint residual scale. Implicit MuJoCo position actuators enforce torque bounds. |
| Tracking, Table S4 | Seven exponential tracking terms; action-change, joint-acceleration, joint-limit and requested-torque-limit penalties; survival reward. All weighted by control dt. |
| Adaptive RSI, S5 | Valid temporal bins, joint-tracking similarity, failure EMA, temperature-scaled probabilities with a uniform floor. |
| Assistance, S6 | Reference acceleration feedforward, gravity/gyroscopic terms and pose/velocity feedback; bin-dependent scale fixed throughout each episode. |
| PPO, Table S7 | rsl_rl 5.5.2, normalized actor 512/256/128 and critic 512/512/256 MLPs, ELU, 24-step rollouts, 5 epochs, 4 minibatches. |
| Timing, Table S6 | 0.004 s physics, decimation 5 (50 Hz control), 10 s training horizon. |

Full rotational error uses a shortest-path quaternion logarithm. Body pose rewards are relative to the pelvis. Runtime math and MuJoCo are NumPy/CPU; observations cross to the learner as float32. CPU/MPS/CUDA use the same MDP and PPO, with no CUDA-only kernel or mixed-precision path.

## Explicit implementation choices and deviations

- **Simulation/contact:** CPU MuJoCo rather than Isaac Lab. All robot collision geoms permit frictional ground contact and self-contact. MuJoCo sliding friction is randomized in [0.6,1.0]; the paper's independent static/dynamic friction and restitution are not directly reproduced. Link mass and inertia scale together in [0.9,1.1]. Flat floor only; box scenes are not implemented.
- **Actuators:** Unitree RL Mjlab reflected motor inertias are reused, including simplified doubled ankle/waist values. Gains use 10 Hz and critical damping. This is not the paper's calibrated closed-chain actuator model. Residual scales are 0.20 for hip/knee, 0.05 for wrists, and 0.10 otherwise; the per-joint allocation is an engineering choice.
- **Frames and assistance:** Pelvis is the motion root; proprioception is torso-based. The assistive controller tracks the torso reference and applies a wrench to the torso. It uses whole-body mass, the nominal torso-link rotational inertia, and the whole-body COM offset at the default pose. Moments are shifted from torso origin to the COM where MuJoCo applies Cartesian wrenches. This explicit G1 mapping should be revisited against any released author implementation. A locked whole-body inertia combined with pelvis feedback and torso actuation was unstable and is not used.
- **Keybodies:** Torso, ankle-roll links, wrist-yaw links and knees. Contact sensors report net world force on each selected body against the floor, then convert to the root frame for the critic. No reference contact labels are needed.
- **Terminations:** Root-position error >1 m, orientation error >1.5 rad, or selected-body net contact force >3000 N. These thresholds, reset perturbations and 0.2 s dwell are explicit engineering defaults. Contact itself and non-upright poses are not failures. MuJoCo numerical instability raises an error rather than silently resetting a corrupted simulation.
- **RSI normalization:** The possible horizon is the remaining clip plus dwell, capped at the training horizon. This resolves the late-start ambiguity in S5's full-clip-length notation. Simultaneous completions in one bin use their mean failure and effective EMA coefficient `1-(1-alpha)**count`, making the batched update independent of environment order. The initial failure level is one.
- **Torque penalty:** Uses mean absolute requested PD torque over physics substeps, before actuator clipping; otherwise a penalty on torque-limit violations would always vanish.
- **Timeout bootstrapping:** A small PPO adapter bootstraps from the terminal observation before auto-reset. The upstream 5.5.2 shortcut uses the previous value estimate. Genuine failures override coincident timeouts.
- **Training size:** Local default 64 environments, versus 4096 in the paper. This changes batch size and optimization behaviour; it is a development default, not an equivalent reproduction budget. CPU physics remains CPU physics even with a CUDA learner.

## Checkpoints, evaluation and ablations

Checkpoints store policy/critic and normalization state, optimizer, learning rate, completed iteration count, motion/asset hashes, environment/training configuration, sampler failure levels/visits, and the environment RNG. Resume starts new RSI episodes; it is not a bitwise continuation of the simulator or device RNG. Moving a checkpoint between MPS and CPU is tested. A motion hash mismatch is rejected for training; evaluation can intentionally use different clips of the same format. Only load checkpoints from trusted sources (PyTorch checkpoint loading uses pickle).

Evaluation starts at reference time zero, runs the full clip plus dwell (without the training horizon cap), and sets assistance to zero. Reported success means completing without an engineering termination; also inspect tracking errors and completion. Nominal runs are deterministic and repeated seeds are not independent robustness evidence. `--randomized` enables perturbations. Metrics are averaged only over the initial rollout, excluding subsequent auto-resets; short failed rollouts should be interpreted together with completion and success.

```bash
zest-train --motions FILE.npz --no-assistance
zest-train --motions FILE.npz --uniform-sampling
zest-train --motions FILE.npz --no-randomization
zest-evaluate --checkpoint PATH --motions HELD_OUT.npz --randomized
```

Use `--config FILE.json` for G1Config overrides. Changes affecting the environment must be reported with any comparison. Settings and dependency versions are saved to each run's `run_config.json`. TensorBoard reads `logs/`.

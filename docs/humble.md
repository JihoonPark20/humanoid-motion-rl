# HuMBLE on G1

The flat-ground pipeline implements teacher PPO, online student-rollout distillation, critic-only adaptation, and mixed-task PPO. CPU/MPS execution is tested; CUDA and convergence are unverified. This is an independent MuJoCo adaptation, with the differences below. See the [pre-implementation review](humble-review.md) for rationale and [validation](validation.md) for measured results.

## Data

```bash
uv sync --extra dev
uv run zest-data --seconds 30 --height-offset 0.005
```

The shared `zest-data` importer (`humble-data` is an alias) downloads a pinned, checksum-verified HOLD G1 HDF5, reads its timestamps and joint names, resamples to 50 Hz, generates torso-frame commands and audits every converted frame for joint limits and floor penetration. Raw files are immutable. Converted files include source hashes, crop bounds, coordinate assumptions and annotation parameters; existing outputs are not overwritten. License attribution is retained with downloads. The command above uses the same explicit walking-clip height offset as the README.

Use `--clips walk/S0_normal_walk_1.h5 ...` for additional release paths, or `--source FILE.h5 ...` for local data. `--height-offset METRES` applies an explicit recorded vertical translation, never an automatic correction. The default is zero. `--start`, `--seconds`, `--cutoff` and `--anticipation` control conversion. Use a separate output directory for changed preprocessing.

The eight-clip example uses `data/hold_flat/` with the first 30 s of four recordings and their provided mirrors: `S0_normal_walk_1`, `normal_sidestep`, `shuffle_turn_in_place`, and `shuffle_start`. Walking clips have a recorded +0.005 m height offset because the unmodified local geometry slightly exceeded the 0.03 m penetration termination; other clips are unchanged. All 12,008 converted frames were checked, with no joint-limit violation above 1e-5 rad. Maximum remaining penetration is approximately 0.02503 m. This is a small training subset, not a full HOLD benchmark or a certified retargeting result. Backward and sprint coverage remain limited.

Prepare this subset once in a fresh checkout (downloaded data is not distributed with the repository):

```bash
uv run zest-data --seconds 30 --height-offset 0.005 --output data/hold_flat \
  --clips walk/S0_normal_walk_1.h5 walk/S0_normal_walk_1_mirrored.h5
uv run zest-data --seconds 30 --output data/hold_flat \
  --clips sidestep/normal_sidestep.h5 sidestep/normal_sidestep_mirrored.h5 \
  shuffle/shuffle_turn_in_place.h5 shuffle/shuffle_turn_in_place_mirrored.h5 \
  shuffle/shuffle_start.h5 shuffle/shuffle_start_mirrored.h5
```

Keep each original/mirror pair and all crops from a recording in the same partition. Do not use the mirrored training clip as held-out evidence. Evaluation accepts separate `--motions` explicitly; the initial eight-clip experiment has no independent held-out benchmark yet. `metadata.recording_group` records the original recording name.

## Stages

The commands below illustrate the progression. Evaluate the teacher before distillation; a successful process exit does not mean a competent policy. Example iteration counts are experiment budgets, not convergence guarantees. Replace paths with the checkpoints produced by each command. Each run requires a fresh log directory.

```bash
# Teacher: the same portable learner on MPS/CUDA/CPU.
uv run humble-train --stage teacher --motions data/hold_flat/*.npz \
  --config configs/humble-flat.json --device mps \
  --iterations 1000000 --log-dir logs/humble/teacher

uv run humble-evaluate --checkpoint logs/humble/teacher/latest.pt --mode reference

# DAgger-style distillation: student states are labeled by the frozen teacher.
uv run humble-train --stage distill --init logs/humble/teacher/latest.pt \
  --device mps --iterations 20000 --log-dir logs/humble/prior

# Actor weights AND normalization statistics stay fixed here.
uv run humble-train --stage warmup --init logs/humble/prior/latest.pt \
  --device mps --iterations 5000 --log-dir logs/humble/warmup

# 50:50 reference/goal environments; independent advantage normalization.
uv run humble-train --stage finetune --init logs/humble/warmup/latest.pt \
  --device mps --iterations 20000 --log-dir logs/humble/finetune
```

`--iterations` specifies additional optimization iterations for each invocation, including resumed runs. Do not automatically distill an unsuccessful teacher. `--tiny` selects small test networks, while normal runs use the paper-sized MLPs. For a tiny pipeline, use `--tiny` consistently. `--config` can override `env` fields and `train` fields; defaults for non-teacher stages use a fixed 1e-6 learning rate. Reduced environment counts change the optimization regime relative to the paper's 4,096 environments.

## Monitor, stop and resume

Runs write `metrics.jsonl`, TensorBoard events, `run_config.json`, `status.json` and atomic `latest.pt`. A numbered checkpoint is retained every ten save intervals. `--save-interval` defaults to 100 iterations. `status.json` includes a process ID while running. An exception records a failed status; the last successfully written checkpoint remains available.

Interactive terminals show a single updating progress line with iteration count, estimated remaining time and the metrics that fit the terminal width. Evaluation results and the final checkpoint path appear separately. Redirected output uses plain summary lines every 100 iterations, at the first/final update and after evaluations. Full metrics still go to JSONL and TensorBoard every iteration. Progress and ETA cover the additional iterations requested for this invocation; the iteration counter includes previous training. Training similarity may use assistance and is not an unassisted success rate. Console changes apply when starting or resuming a process.

Every 1,000 iterations, a separate assistance-free environment evaluates the training references without altering the training simulator or policy. Results are saved as `reference_evaluation_ITERATION.json`; `best_reference.pt` retains the best mean completion. This is training-set monitoring, not held-out validation. Set `--eval-interval` to change the interval or zero to disable it. Training ends after the requested iterations or a manual stop.

```bash
uv run tensorboard --logdir logs/humble

# Stop after the current optimization iteration, saving the latest checkpoint.
touch logs/humble/teacher/STOP

# Resume in a new directory for additional iterations.
uv run humble-train --resume logs/humble/teacher/latest.pt \
  --device mps --iterations 1000000 --log-dir logs/humble/teacher_resume
```

SIGINT/SIGTERM also request a checkpointed stop at an iteration boundary. Normal system sleep suspends progress; an idle-sleep inhibitor may be used for an intentionally unattended Mac run. Resume restores optimizer, normalization, curriculum and RNG state, but starts new RSI episodes; it is not a bitwise simulator continuation. CPU↔MPS loading is tested. Asset/action/observation/data contract mismatches are rejected. Only load trusted checkpoints.

## Replay and evaluation

The [Viser browser viewer](viewer.md) supports references, teacher playback and student velocity controls without `mjpython`:

```bash
uv run --extra viewer zest-view --checkpoint logs/humble/finetune/latest.pt
```

```bash
# macOS GUI; use python on Linux. Student playback needs no motion files.
mjpython -m zest.scripts.humble_play \
  --checkpoint logs/humble/finetune/latest.pt --command 0.5 0 0

uv run humble-play --checkpoint logs/humble/finetune/latest.pt --headless --steps 500
uv run humble-evaluate --checkpoint logs/humble/finetune/latest.pt --mode commands
uv run humble-evaluate --checkpoint logs/humble/finetune/latest.pt \
  --mode reference --motions data/hold_test/*.npz --randomized
```

Teacher playback uses its stored reference paths, or `--motion FILE`. Existing `zest-play --reference --motion FILE` also previews converted HOLD references. All evaluation disables assistance. JSON and CSV retain failures and completion, rather than reporting only survivors. Reference mode measures full clips from frame zero, with unaligned errors over realized steps. Command mode starts standing, holds zero for 1 s, applies a command, then stops; it measures command error, response time and stopping error. A response must remain within tolerance for 0.2 s. Steady-state results are null if the rollout fails before the command plateau ends. Repeated nominal seeds are deterministic duplicates.

The first evaluator is a reproducible local diagnostic, not the complete paper benchmark: DTW alignment, calibrated force-sweep robustness and all published command grids remain future additions. `--randomized` includes the currently supported dynamics/noise/impulsive-push variation.

## Implemented definitions and deviations

- **Actions/observations:** Table S7 nominal targets, gains and scales. Student: 96 values without measured linear velocity or references; teacher: 135 values; clean privileged critic: 191 values. Teacher and student use one action convention. The critic receives task identity, true torso velocity and end-effector floor wrenches; the actor does not. Goal-task reference channels are zero-padded. The inherited motor armatures and torque limits are approximate asset specifications.
- **Learning:** Teacher MLP 1024/512/256; student 1024/1024/1024; critic 512/512/256. Distillation minimizes action MSE on student-generated states with a frozen teacher; this instantiates the paper's norm objective with a squared mean. Teacher critic parameters carry through distillation and are adapted during warm-start. GAE bootstraps truncations from pre-reset terminal observations. Both reference and goal advantages are normalized separately.
- **Rewards:** Per-trajectory standard deviations with 0.2 floors; HuMBLE G1 velocity/gravity/joint tracking, action smoothing, torque norm, feet sliding/jerk, reference-only clearance, survival and goal command rewards. Rewards are scaled by control dt. Torque uses the final substep actuator force; feet use ankle-roll link frames. Jerk uses control-rate differences and excludes the first two reset samples. These discretizations/contact-frame choices may differ from the authors'.
- **Assistance:** Nominal whole-body composite inertia about the torso origin, world-frame control and conversion to the torso COM application point. Composite rotational energy is tested against the constituent bodies. Teacher assistance is constant per episode and updated per temporal bin; simultaneous completions in a bin use mean similarity. `success_decay` decreases beta on successful tracking. `paper_literal` exposes the contradictory printed direction. This interpretation, bin aggregation, inherited adaptive sampler and remaining-horizon normalization are explicit adaptations, not a claim of exact reproduction. All subsequent stages use zero assistance.
- **Commands/initialization:** Versioned second-order 3 Hz low-pass filtering with acceleration-dependent lookahead capped at 0.15 s. Published filter/shift coefficients are unavailable. Commands use stored G1 timing with no extra scale warp. GC-RSI bins commands at 0.2 per component, uses the nearest populated bin when empty, and samples a ±0.2 s temporal window. Goal commands resample every 3 s, including a 15% stop probability. These numeric choices are local defaults.
- **Physics:** 50 Hz control over CPU MuJoCo, nominal flat floor, reference-specific height/gravity terminations, goal minimum torso height, ground penetration, peak contact force and foot collision checks. Contact sensors expose floor wrenches, not every possible articulated interaction wrench. Randomization currently reuses the ZEST implementation's friction/mass/inertia, reset/noise and push ranges; it does not reproduce every entry of Table S5. No terrain height scan or roughness is included in this agreed first phase.
- **Data/resources:** Full FK/differentiation uses CPU float64, then resident motion arrays use float32. Loading is still eager; benchmark memory before using the entire release. Physics remains CPU even with a CUDA learner. No AMP, CUDA kernel or alternate MDP is required for the portable path.

Training loss or assisted similarity alone is insufficient evidence of success. The next scientific gate is unassisted teacher completion across the motion library, followed by held-out and command-policy comparisons. Hardware deployment, box climbing and hierarchical parkour remain outside this version.

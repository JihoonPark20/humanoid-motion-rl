# Validation record

Validated locally on **2026-11-10**, Apple M1 Max (32 GB), macOS 15.2, Python 3.12.4. Installed versions: MuJoCo 3.17.0, PyTorch 2.15.0, rsl_rl 5.5.2, NumPy 2.5.4. `uv.lock` records the resolved dependency graph.

## Checks

- Rotation conventions, antipodal quaternions and angular velocity differentiation.
- HOLD HDF5 timestamp/named-joint conversion, motion sampling, FK, terminal dwell and invalid-input rejection.
- Adaptive probabilities, valid clip bins, early-failure scoring, duplicate-bin EMA and curriculum restore.
- Wrench gravity/feedforward terms; ground reaction force direction and approximate weight balance.
- Actor isolation from measured root translation/velocity, immutable observation snapshots, residual targets.
- Timeout terminal observations, failure precedence and zero assistance in full-clip evaluation.
- PPO parameter updates and terminal-value bootstrapping on CPU and MPS; checkpoint/resume on both.
- Evaluation metrics exclude auto-reset episodes after the initial rollout ends.
- CUDA tests are skipped when hardware is absent. Linux CPU CI is configured but has not run locally.
- Wheel build includes G1 XML/meshes/licenses; an extracted wheel loads the model outside the source directory.
- Native macOS `mjpython` viewer tested for reference and learned-policy playback (30 steps each).

```bash
uv sync --extra dev
uv run ruff check zest tests
uv run pytest -q
```

## HuMBLE flat-ground implementation

Also validated on 2026-11-10, with h5py 3.16.0 added. **22 tests passed, 2 CUDA tests skipped** across the combined suite. Lint and wheel build passed. New checks cover timestamp-based HDF5 resampling and named-joint reordering, composite-inertia rotational energy, student observation isolation, clean critic inputs, nominal-target actions, contact penetration detection, task-wise advantage normalization, and command-conditioned initialization.

All four learning stages were updated on CPU and MPS. Tests verify that the distillation teacher remains frozen, critic warm-start changes the critic while preserving the entire actor state (including normalization), checkpoints transfer between stages, and CPU resume preserves saved policy state. Periodic evaluation leaves the training simulator and actor state unchanged. A STOP-file execution test saved a valid checkpoint without performing another iteration. The final student's command playback ran without source motion files, including 30 steps in the native macOS viewer. These checks do not establish policy quality.

Real HOLD execution artifacts (ignored locally):

- `logs/humble_check_{teacher,distill,warmup,finetune}/latest.pt`: two tiny-network iterations per stage on MPS. The final command evaluation completed 0/16 rollouts, with about 20.2% mean completion; this is a deliberately tiny execution check.
- `logs/humble_teacher_pilot/latest.pt`: 100 iterations × 24 steps × 32 environments = 76,800 transitions, using full-size teacher/critic networks on MPS. Updates took approximately 47.2 seconds. Unassisted reference evaluation completed 0/8 rollouts, with 6.27% mean completion on the 10-second training clip.
- `logs/humble_flat_pilot/latest.pt`: 25 iterations × 24 × 64 environments = 38,400 transitions on the eight-clip, 30-second subset. Updates took approximately 20.75 seconds without numerical failures.
- `logs/humble_flat_resume_check/latest.pt`: one more MPS iteration from that checkpoint, followed by automatic unassisted evaluation of all eight training references. Completion was 1.81%, with 0/8 completed rollouts. This validates resume and monitoring; the policy is not yet a useful teacher.

The small dataset contains four original recordings and four mirrors, not eight independent skills. Full-dataset coverage, held-out generalization, convergence, DTW-based paper comparisons, rough terrain, Linux runtime and CUDA remain unverified. The [HuMBLE guide](humble.md) records the actuator, curriculum, annotation, contact and randomization differences.

A bounded **48-hour teacher run** was launched from the 26-iteration flat-ground checkpoint at `logs/humble/teacher_48h`, using 64 environments, six simulation threads and MPS. `run_config.json` records the start time, package versions and source hashes; `status.json`, `metrics.jsonl` and `logs/humble/teacher_48h.console.log` contain live state. The launch uses macOS `caffeinate -i` for idle-sleep inhibition. It saves every 100 iterations and evaluates the training references without assistance every 1,000. This record confirms launch, not completion or convergence; inspect the live status for the current result. No automatic transition to distillation is scheduled.

The user subsequently requested manual control of training. The job was **stopped at iteration 185**, its final checkpoint was verified against `status.json`, and process exit was confirmed. The old STOP file is retained; resume commands create a new run directory.

## Viser playback

Viser 1.1.1 and mjviser 0.0.16 are optional viewer dependencies. The viewer served HTTP successfully and completed a bounded startup/shutdown check. Three new playback tests passed: kinematic reference scrubbing without simulation stepping, and immutable policy/normalization parameters during ZEST and HuMBLE playback. No optimization steps were run for these tests. Native Chrome automation conflicted with concurrent browser use, so browser rendering and mouse-control verification were not completed in this session.

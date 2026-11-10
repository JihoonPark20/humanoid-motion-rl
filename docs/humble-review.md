# HuMBLE feasibility review

Reviewed 2026-11-10 for Unitree G1 (29 DoF), MuJoCo CPU simulation, and a shared PyTorch learner on macOS/MPS and Linux/CUDA. **This is the pre-implementation assessment.** The user subsequently selected flat-ground implementation first and a multi-day experiment budget. Current implementation and remaining limitations are in the [HuMBLE guide](humble.md); this historical review does not establish policy performance.

## Assessment

An independent implementation of the core learning pipeline, simulation training, playback and evaluation is technically feasible. The existing G1 assets, batched simulator, motion interpolation, PPO integration and checkpoint tooling provide a useful foundation. However, this is a substantial new training pipeline, not a configuration change or direct distillation of the current checkpoint. The current ZEST validation policy completed **0/8 unassisted nominal rollouts**; see [validation](validation.md). A competent HOLD teacher must be trained before useful distillation is possible.

Use HOLD's retargeted G1 locomotion subset as the primary HuMBLE dataset. Mac development can validate the complete algorithm and small experiments; convergence across the full locomotion repertoire and paper-level results remain empirical goals with an uncertain compute budget. CUDA compatibility is a design target, not a tested claim. The paper trains in Isaac Lab and also evaluates in MuJoCo; its MuJoCo evaluations do not establish the cost or convergence of training this CPU implementation. [HuMBLE](https://arxiv.org/html/2610.10489v1)

## Relationship to ZEST

[ZEST](https://arxiv.org/abs/2602.00401) learns reference-conditioned motion tracking using adaptive sampling and assistive dynamics. HuMBLE uses a related tracking setup to create a teacher, then learns a policy controlled by longitudinal velocity, lateral velocity and turn rate. The motion prior ultimately resides in the learned policy parameters; the final policy does not need a reference clip at inference.

The required stages are:

| Stage | Required behavior | Main implementation work |
| --- | --- | --- |
| Data | Retargeted motions plus frame-aligned steering commands | HOLD importer, kinematic audit, command annotation, reproducible splits |
| Teacher | One reference-conditioned policy covers the locomotion library | HuMBLE observations, action convention, rewards, terminations and assistance |
| Prior | Student acts; teacher labels the same visited states | Online DAgger-style rollouts, frozen teacher and independent normalizers |
| Critic adaptation | Adapt the teacher critic to the student policy | Critic-only updates; actor parameters and actor normalization remain fixed |
| Fine-tuning | Shared policy learns reference-guided and goal-conditioned tasks | Mixed environments, task-aware critic, separately normalized advantages, command-conditioned initialization |
| Evaluation | Measure motion fidelity, command response and robustness | Command schedules, failure-aware metrics and controlled ablations |

The nominal fine-tuning allocation is 50:50 reference/goal environments, with 3 s and 8 s horizons respectively. The critic receives a task indicator; the actor does not. The paper uses 5,000 critic-adaptation iterations. Style preservation comes from the reference-guided task during RL, not from adding an AMP discriminator or assuming action distillation alone solves command generalization. These details are documented in the method and Table S9. [HuMBLE](https://arxiv.org/html/2610.10489v1)

## HOLD: verified availability and remaining risks

Inspected release: [`f80b04d384d4ec8922c88deada878c88597a3cf9`](https://github.com/rai-opensource/hold/tree/f80b04d384d4ec8922c88deada878c88597a3cf9). The dataset repository contains data and documentation; no training implementation or usable teacher checkpoint was found there. This is not a claim that no other release exists.

Local enumeration of Git LFS pointers gave:

| Subset | Files | Payload bytes |
| --- | ---: | ---: |
| G1 locomotion | 218 HDF5 | 1,052,719,920 |
| G1 teleoperation | 33 HDF5 | 273,989,696 |
| All G1 | 251 HDF5 | 1,326,709,616 |

The counts include mirrored files. The human BVH release has two scale variants; it should not be confused with the G1 subset or the paper's exact training split. Start with `data/g1/locomotion`, not the entire loco-manipulation release. Keep an original and its mirrored counterpart in the same train/validation/test partition; also group segments from the same recording. Existing mirrors must not be duplicated as new data. The release uses [CC BY 4.0](https://github.com/rai-opensource/hold/blob/f80b04d384d4ec8922c88deada878c88597a3cf9/LICENSE); preserve attribution and conversion provenance.

Two actual LFS payloads were downloaded, and their SHA-256 hashes matched the published pointers:

| Sample under `data/g1/locomotion` | Frames | Duration | SHA-256 |
| --- | ---: | ---: | --- |
| `walk/S0_normal_walk_1.h5` | 10,439 | 77.794414 s | `ff18b6316a3c89a17c79920d388d59114b0d6fd863a628128d5bf78b5f9a8ccd` |
| `sprint/sprint_full_room_002.h5` | 199 | 1.475694 s | `fd5a7b1028083ce615018d8d187e7aebc2220ae6085ebf9b5e7972e894672ef1` |

Observed schema in both samples:

```text
robot_data/meta_data/quat_order                  # w, x, y, z
robot_data/trajectory_data/time                 # [T]
robot_data/trajectory_data/bodies/labels         # ["base"]
robot_data/trajectory_data/bodies/{pos_w,quat_w,lin_vel_w,ang_vel_w}
robot_data/trajectory_data/joints/labels         # 29 named joints
robot_data/trajectory_data/joints/{pos,vel}      # [T,29]
```

The joint labels match this workspace's 29 names and order. Positions were finite and within the current model's joint limits. Quaternions were approximately unit length. **Timestamps were spaced by 0.007453 s, approximately 134.174 Hz, not 120 Hz or 50 Hz.** The release documents 80% spatial scaling and corresponding temporal scaling. Use the stored timestamps, normalize quaternions and resample once to the 50 Hz controller; do not reapply the human-to-G1 scaling. [G1 format](https://github.com/rai-opensource/hold/blob/f80b04d384d4ec8922c88deada878c88597a3cf9/docs/g1.md), [processing](https://github.com/rai-opensource/hold/blob/f80b04d384d4ec8922c88deada878c88597a3cf9/docs/process.md)

Neither inspected sample contains steering command annotations. The only stored body is `base`, whereas commands and tracking use the torso. Recover torso motion with forward kinematics and differentiated rotations on the local model; do not treat pelvis velocity as torso velocity. The mapping from the exported `base` frame to the local free joint must be established rather than inferred solely from matching joint names.

Two concrete quality checks need follow-up:

- Assuming `base` maps directly to the current pelvis free joint, forward kinematics at 100 evenly spaced frames per clip produced maximum robot/floor collision penetration of **25.43 mm for walking and 40.51 mm for sprinting**. This is a diagnostic of this data/model pairing, not proof of a defective dataset. Investigate base transforms, floor offsets and collision geometry before changing trajectories. An automatic height shift could distort contact timing and should not be applied silently.
- In both samples, the three left wrist angles closely follow the previous frame's three waist angles: maximum absolute differences were about **0.000101 rad** and **0.000371 rad**, respectively. This unusual coupling warrants inspection of more clips and the export/retargeting convention. It is not sufficient evidence to relabel joints or declare the release corrupted.

These samples verify accessibility and a plausible import route, not quality of all 218 clips or dynamic trackability. Download only the needed LFS subset initially. Keep raw files immutable, preserve checksums, record any corrections and exclude bad clips through an explicit manifest. Cache processed arrays on disk or load clips lazily: the present eager float64 FK/derivative cache will amplify memory beyond the raw payload size.

## Compatibility changes required

| Area | Current workspace | HuMBLE extension |
| --- | --- | --- |
| Actions | `q_target = q_reference + scale * action` | Nominal-pose targets with Table S7 gains/scales |
| Actor | 132-dimensional reference-conditioned input | Separate teacher/student observation contracts; student uses commands |
| Critic | 217 values, partly based on noisy actor observations | Clean privileged observations, consistent shape across stages, task bit and optional G1 terrain samples |
| Rewards | ZEST-style exponential tracking and regularization | Trajectory-normalized tracking, locomotion regularizers and task-specific command reward |
| Assistance | Bin-dependent EMA; torso-link rotational inertia | Re-derived nominal composite inertia and explicit resolution of the paper's schedule ambiguity |
| Termination | Current tracking thresholds | Task-specific G1 conditions from Table S4 |
| PPO | Global advantage normalization | Separate normalization for each task group |
| Data/reset | Reference clips and adaptive bins | Steering annotations, command-bin index and goal-conditioned RSI |
| Replay/evaluation | Reference-conditioned rollouts | Standalone command-driven rollouts plus teacher/prior/final comparisons |

**Actions cannot be copied unchanged.** For an old residual teacher, an action label would have to be converted to the same physical target:

```text
a_student = (q_reference + scale_teacher * a_teacher - q_nominal) / scale_student
```

This is only a coordinate conversion, not a policy-quality fix. It requires matching joint semantics, compatible gains and consistent action history. HuMBLE's published G1 gains and scaling also differ from the current defaults. Prefer training a new HuMBLE teacher and student with one common nominal-target convention; retain the existing ZEST convention as its own configuration. Record the action contract in checkpoints and reject incompatible loads.

**Avoid privileged-input leakage.** Table S3 marks measured linear velocity as unavailable for G1; the deployment section also discusses a velocity-free variant. Recommend that variant as the default here. With one IMU, 29 joints and a previous 29-value action, angular velocity + gravity + joint position + joint velocity + previous action + three commands gives **96 inputs**. This dimension is our explicit implementation choice, not a universal dimension stated by the paper. Reference observations, true simulator linear velocity, task indicator and terrain scans must not leak into this student. Teacher and critic inputs should be declared independently. [HuMBLE, Table S3 and deployment section](https://arxiv.org/html/2610.10489v1)

**Reuse the installed learner carefully.** Inspection of local `rsl_rl` 5.5.2 found a `Distillation` algorithm that already executes student actions and collects teacher labels on the same observations. It is a useful foundation after defining separate observation groups and action semantics. Its MSE/Huber choices should be documented against the paper's norm notation. Its PPO normalizes advantages globally or per minibatch, not by task; merely concatenating environments will not implement HuMBLE. Preserve the existing terminal-observation bootstrap correction when adding task masks.

**Critic adaptation must really freeze the actor.** Freeze actor normalization statistics as well as weights, exclude actor updates, and verify identical actions on a fixed observation batch before/after adaptation. Keep the critic input schema stable from teacher training onward so critic transfer is meaningful. The goal task's unused reference channels require a documented convention; the paper does not fully specify every padding detail.

## Reproducibility ambiguities

1. **Assistive gain update.** Section S5 says assistance is multiplied by beta, but also says beta increases with probability equal to tracking similarity `s` and decreases with probability `1-s`, while describing assistance as fading with proficiency. With Table S8's increments, the unclipped expectation is `E[delta beta] = 0.05*s - 0.1*(1-s) = 0.15*s - 0.1`. Above `s=2/3`, this increases assistance. The PDF text confirms the same wording. Treat this as an unresolved inconsistency, not permission to claim an arbitrary reversal reproduces the authors. Expose and document the chosen schedule, distinguish literal and corrected interpretations, and test assistance-off performance. Author clarification would improve fidelity but need not block the rest of the engineering. [HuMBLE, Section S5 and Table S8](https://arxiv.org/pdf/2610.10489)
2. **Assistive inertia.** HuMBLE explicitly specifies nominal composite rigid-body inertia. The current torso-link approximation is not equivalent. Derive the composite inertia and COM moment shift with consistent torso/world frames and application point; validate rotational response. The earlier instability in this workspace combined multiple frame/control choices and does not prove composite inertia itself is inappropriate.
3. **Command annotation.** The paper describes local torso velocities, low-pass filtering and an acceleration-dependent temporal shift. The inspected paper/data documentation does not provide enough numeric detail to uniquely reconstruct the filter and shift. Choose versioned parameters, plot the resulting command/velocity alignment, and report sensitivity instead of presenting the reconstruction as exact.
4. **Experimental identity.** Exact training/benchmark clip manifests, some command sampling/binning choices and complete stage stopping budgets are not established by the inspected release. Define reproducible local choices and keep them distinct from paper claims. The full public HOLD release must not be assumed to be the exact experimental split.
5. **Simulator equivalence.** MuJoCo contact geometry, friction and actuator behavior differ from Isaac Lab. Match the specified G1 gains and available randomizations, document approximations, and tune only against a fixed validation protocol. Adding a CUDA learner does not remove these differences.

## Device strategy and scale

Retain one CPU MuJoCo simulation path and one float32 PyTorch training path. Device selection should affect tensor placement, not rewards, actions or task definitions. Keep preprocessing and FK on CPU; disable CUDA-only optimizations and mixed precision in the portable baseline. MPS-compatible masked reductions suffice for task-wise advantages. Save checkpoints in a device-independent form, including both normalizers, stage, task configuration, data manifest and optimizer state.

The Table S9 prior MLP has three 1,024-unit hidden layers. With 96 inputs and 29 outputs, it is approximately 2.23 million parameters: model size itself is practical on this 32 GB Mac. Rollout buffers, activations and eager motion caches can be much larger; profile those before increasing environments.

The paper's 4,096 environments × 24 rollout steps give 98,304 transitions per iteration. Applying that batch size to 5,000 critic warm-start iterations implies **491,520,000 transitions for that phase alone**. This is a sample-budget calculation, not a measured runtime or a minimum required for every independent implementation. The existing 76,800-transition ZEST smoke run cannot predict HuMBLE convergence or runtime with larger networks and the full dataset.

Use explicit small development configurations and separately recorded reproduction configurations. Reducing environment count changes optimization statistics; repeating the same number of iterations is not an equivalent training budget. CUDA can accelerate learning, but CPU simulation may remain dominant. A future GPU physics backend is a separate optimization project with its own parity validation, not a prerequisite for the first implementation.

## Recommended implementation sequence and acceptance gates

Proposed extensions can remain under `zest/`: `motion/hold.py` and `motion/commands.py` for data, `envs/g1/humble.py` for task variants, `utils/distillation.py` and `utils/multitask_ppo.py` for learning. These are proposed modules, not existing commands. Keep the common simulator/assets and the already archived upstream provenance; there is no need to restore the full external repository.

| Milestone | Deliverable | Evidence required before proceeding |
| --- | --- | --- |
| 1. Data contract | Pinned HOLD importer and command annotations | Checksums; joint/frame/time tests; contact/limit audit; reference playback; grouped splits |
| 2. Teacher | HuMBLE G1 reference tracker | Assistance-off full-clip completion and errors across forward/backward/side/turn/stop validation clips; multiple seeds |
| 3. Prior | Online teacher-student training | Teacher stays frozen; no privileged student inputs; held-out action error and successful closed-loop rollouts |
| 4. Fine-tuning | Critic adaptation and mixed-task PPO | Frozen actor during warm-start; task-wise normalization; compatible resets/timeouts; command gains without unacceptable fidelity loss |
| 5. Reproduction evaluation | Repeatable comparison suite | Teacher/prior/final, goal-only and reference-only ablations under matched budgets |
| 6. Terrain extension | Heightfields and privileged terrain observations | Flat/rough comparisons; no terrain observation in actor; updated contact tests |

For the first milestone sequence, prioritize flat-ground walking, backward/sideways motion, turning and stopping; then expand the accepted motion/command envelope to jogging and sprinting. This is an engineering recommendation, not a reduced implementation presented as the paper's full result. If rough-terrain fidelity is required from the outset, include the terrain path in the teacher phase: the paper introduces roughness during teacher training, not merely during final evaluation. Box parkour, separate climbing experts and hierarchical navigation are additional systems, outside the agreed core locomotion scope.

Evaluation should report full-trajectory completion and falls alongside joint/torso errors. For the command policy, include step response/rise time, steady-state and whole-rollout velocity error, stop behavior, simultaneous translation/turn commands, command coverage and push recovery. Report both raw and DTW-aligned fidelity, including the time-warp magnitude: alignment can conceal response delay. Failed runs must remain visible rather than disappearing from averages. Separate reference-initialized evaluation from starts at a standing pose. Keep held-out motion tests distinct from deliberately out-of-distribution command tests.

The decisions still useful from the user are the initial inclusion of rough terrain and the allowable continuous training budget per experiment. Robot, dataset preference, simulation scope and the MPS/CUDA design direction are already established. These decisions affect prioritization and attainable validation depth, not the feasibility of writing the algorithm.

## Work performed for this review

Read the HuMBLE paper and supplementary tables, ZEST abstract, HOLD README and G1/processing documentation; inspected the current environment, simulator and installed PPO/distillation code; enumerated pinned LFS metadata; downloaded and hash-verified two HDF5 samples; checked schema, timing, joint limits and provisional FK/contact compatibility. Research downloads remain under ignored `.cache/research/`. No full HOLD training, HuMBLE implementation, CUDA run or policy-quality benchmark was performed. README language links and local documentation links were checked after editing.

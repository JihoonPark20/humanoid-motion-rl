# Motion data

## Recommendation for HuMBLE: HOLD

Use [HOLD's G1 locomotion trajectories](https://github.com/rai-opensource/hold/tree/f80b04d384d4ec8922c88deada878c88597a3cf9/data/g1/locomotion) for HuMBLE. The release includes robot-space motion and uses CC BY 4.0. `humble-data` implements pinned download, conversion, command annotation and per-clip kinematic audits; a full-dataset quality audit remains outstanding. See the [implementation guide](humble.md) for commands and the [feasibility review](humble-review.md) for observed timing, geometry and data-quality issues. This recommendation concerns suitability for this task, not a benchmark claiming universal dataset quality.

## Currently supported dataset: Unitree-retargeted LAFAN1

Use [Unitree's G1 LAFAN1 trajectories](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) for the first experiments. They provide 30 Hz robot-space motion with a documented joint order, avoiding a separate human-to-robot retargeting pipeline. Retargeting considers kinematics; it does not guarantee dynamically feasible trajectories. This is a practical starting recommendation, not a claim that this dataset is universally highest quality.

The downloader pins revision `ce1572906efe6157840e8474d5a0d7aa87481e74`, retains the upstream README/license, and records each source SHA-256 and crop interval. Motion content is governed by LAFAN1's CC BY-NC-ND 4.0 terms; code/model licenses are separate. Downloaded and converted data stay under git-ignored `data/`; they are not included in the package.

```bash
# Short first experiment (walk1_subject1, first 10 seconds).
zest-data --seconds 10

# Separate output avoids overwriting a short reference used by an existing checkpoint.
zest-data --all --output data/lafan1_full

# Another clip, or your own already-retargeted Unitree-format CSV.
zest-data --clips dance1_subject1 --seconds 10 --output data/lafan1_dance
zest-data --source /path/to/g1.csv --output data/custom
```

The CSV importer expects headerless rows: root XYZ in metres, root quaternion **XYZW**, then 29 joint angles in radians in the dataset's declared order. It converts to **WXYZ**, resamples at 50 Hz using linear translation/joint interpolation and quaternion SLERP, and removes the initial horizontal translation. It does not silently clamp joints, filter motion, or fix ground penetration. Inspect new motions using `zest-play --reference --motion FILE.npz` before long training.

Internal NPZ version 1 stores `qpos[frames,36]`, scalar `fps`, `joint_names[29]`, `quaternion_order="wxyz"`, and provenance metadata; loading never enables pickle. MuJoCo forward kinematics provides body references. World linear/angular velocities and accelerations are finite-differenced; free-joint angular velocity is converted into MuJoCo's local convention. The final pose is held with zero reference velocity during dwell.

## Expansion candidates

| Dataset | Fit for this project | Additional work |
| --- | --- | --- |
| [AMASS](https://amass.is.tue.mpg.de/) | Broad optical motion-capture coverage; useful for a larger skill library | Registration, applicable dataset/model terms, and G1 retargeting; [GMR](https://github.com/YanjieZe/GMR) is one possible tool |
| [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | Large motion library with G1 output and annotations | User acceptance of access conditions; a separate importer for its documented units/layout |
| Unitree LAFAN1 | Small enough for local iteration, directly usable G1 references | Inspect dynamic feasibility and select clips |

Neither alternative has been downloaded or benchmarked here. Use held-out subjects/clips for generalization claims. Evaluating the training reference measures tracking of that reference, not unseen-skill generalization. Start with one short walk, then add dance/jump and multi-contact clips; do not interpret a single walk experiment as validation of athletic control.

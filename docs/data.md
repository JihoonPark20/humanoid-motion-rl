# Motion data

## Default for ZEST and HuMBLE: HOLD

Use [HOLD's G1 locomotion trajectories](https://github.com/rai-opensource/hold/tree/f80b04d384d4ec8922c88deada878c88597a3cf9/data/g1/locomotion) for both learners. RAI Institute and Boston Dynamics released HOLD and used it in HuMBLE; the release provides G1-retargeted motion under CC BY 4.0. Choosing HOLD for ZEST is a local dataset adaptation, not a claim about ZEST's original training data. `zest-data` implements pinned download, conversion, command annotation and per-clip kinematic audits. `humble-data` remains an alias. Both learners accept the resulting NPZ files; ZEST uses the motion references, while HuMBLE also uses the command annotations.

```bash
# Default clip: walk/S0_normal_walk_1.h5, output: data/hold/.
zest-data --seconds 30 --height-offset 0.005
zest-view --motion data/hold/S0_normal_walk_1.npz

# Additional clips: choose release paths and a separate output directory.
zest-data --clips walk/S0_normal_walk_1.h5 --seconds 30 --output data/hold_unmodified
```

The quick start explicitly applies a +0.005 m vertical offset to this walking clip because of floor penetration in the local G1 geometry. The importer itself defaults to zero offset, records the choice and never overwrites prepared files. This correction is not a dataset-wide default. A single walk does not cover the commands needed for a general locomotion policy. See the [HuMBLE guide](humble.md) for the prepared eight-clip subset, additional imports and recording-group splits. A full-dataset quality audit remains outstanding; the [feasibility review](humble-review.md) documents observed timing, geometry and data-quality issues.

## Conversion and format

The importer reads HOLD HDF5 timestamps, named joints and WXYZ root quaternions. It resamples to 50 Hz with linear translation/joint interpolation and quaternion SLERP, and removes initial horizontal translation. Height offsets must be explicitly requested; joint angles are not silently clamped. Each conversion records the pinned revision, source hash, crop bounds, command annotation settings and geometry audit. Downloaded and converted data stay under git-ignored `data/` and are not packaged; source license attribution is retained with downloads.

Internal NPZ version 1 stores `qpos[frames,36]`, scalar `fps`, `joint_names[29]`, `quaternion_order="wxyz"`, provenance metadata and HuMBLE command annotations. Loading never enables pickle. MuJoCo forward kinematics provides body references. World linear/angular velocities and accelerations are finite-differenced; free-joint angular velocity is converted into MuJoCo's local convention. The final pose is held with zero reference velocity during dwell.

## Evaluation splits

Keep each original recording, its mirrored counterpart and all crops in the same partition. Evaluate held-out recordings separately; training-reference tracking does not measure generalization. Expand locomotion coverage within HOLD after verifying a short walking reference. A single walk does not validate broad locomotion or athletic control.

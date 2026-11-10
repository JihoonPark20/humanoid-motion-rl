"""Pinned HOLD G1 download, timestamp-aware conversion, annotation and geometry audit."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path, PurePosixPath

import h5py
import mujoco
import numpy as np

from zest.envs.g1.config import JOINT_NAMES
from zest.envs.g1.humble_config import HumbleConfig
from zest.motion.commands import annotate
from zest.motion.library import MotionClip, validate_motion
from zest.sim.mujoco_batch import load_model
from zest.utils.math import slerp

REVISION = "f80b04d384d4ec8922c88deada878c88597a3cf9"
RAW = f"https://raw.githubusercontent.com/rai-opensource/hold/{REVISION}"
MEDIA = f"https://media.githubusercontent.com/media/rai-opensource/hold/{REVISION}"


def fetch(clip, directory):
    relative = PurePosixPath(clip)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".h5":
        raise ValueError("Clip must be a relative locomotion HDF5 path")
    path = f"data/g1/locomotion/{relative}"
    with urllib.request.urlopen(f"{RAW}/{path}", timeout=60) as response:
        pointer = response.read().decode()
    digest = pointer.split("oid sha256:")[1].splitlines()[0]
    size = int(pointer.split("size ")[1].strip())
    target = Path(directory) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(f"{MEDIA}/{path}", timeout=60) as response, tmp.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        tmp.replace(target)
    if target.stat().st_size != size or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError(f"HOLD checksum mismatch: {target}")
    license_path = Path(directory) / "LICENSE"
    if not license_path.exists():
        with urllib.request.urlopen(f"{RAW}/LICENSE", timeout=60) as response:
            license_path.write_bytes(response.read())
    return target


def convert(source, destination, *, start=0.0, seconds=None, height_offset=0.0, cutoff=3.0, anticipation=0.15):
    source, destination = Path(source), Path(destination)
    with h5py.File(source) as f:
        group = f["robot_data/trajectory_data"]
        times = group["time"][:]
        names = [x.decode() for x in group["joints/labels"][:]]
        order = [x.decode() for x in f["robot_data/meta_data/quat_order"][:]]
        bodies = [x.decode() for x in group["bodies/labels"][:]]
        if set(names) != set(JOINT_NAMES) or len(names) != 29 or order != list("wxyz") or bodies != ["base"]:
            raise ValueError("Unsupported HOLD joints, quaternion order or body schema")
        q = np.c_[
            group["bodies/pos_w"][:, 0],
            group["bodies/quat_w"][:, 0],
            group["joints/pos"][:][:, [names.index(n) for n in JOINT_NAMES]],
        ]
    if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("HOLD timestamps must be finite and strictly increasing")
    times -= times[0]
    lengths = np.linalg.norm(q[:, 3:7], axis=-1)
    if not np.isfinite(q).all() or np.any(np.abs(lengths - 1) > 0.01):
        raise ValueError("Nonfinite motion or invalid quaternion")
    if start < 0 or (seconds is not None and seconds <= 0) or not np.isfinite(height_offset):
        raise ValueError("Invalid crop or height offset")
    q[:, 3:7] /= lengths[:, None]
    end = times[-1] if seconds is None else min(times[-1], start + seconds)
    target_times = start + np.arange(max(0, int(np.floor((end - start) * 50)) + 1)) / 50
    if len(target_times) < 3:
        raise ValueError("Crop must contain at least three 50 Hz frames")
    lo = np.clip(np.searchsorted(times, target_times, side="right") - 1, 0, len(times) - 2)
    alpha = ((target_times - times[lo]) / (times[lo + 1] - times[lo]))[:, None]
    result = q[lo] * (1 - alpha) + q[lo + 1] * alpha
    result[:, 3:7] = slerp(q[lo, 3:7], q[lo + 1, 3:7], alpha)
    result[:, :2] -= result[0, :2]
    result[:, 2] += height_offset
    validate_motion(result, 50.0, JOINT_NAMES)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared motion: {destination}")
    model = load_model(HumbleConfig(randomize=False))
    base = dict(version=1, qpos=result, fps=50.0, joint_names=JOINT_NAMES, quaternion_order="wxyz")
    temporary = destination.with_suffix(".working.npz")
    np.savez_compressed(temporary, **base)
    try:
        clip = MotionClip(temporary, model)
        commands = annotate(clip.values, 50.0, cutoff, anticipation)
    finally:
        temporary.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    joint_ids = [model.joint(n).id for n in JOINT_NAMES]
    limits = model.jnt_range[joint_ids]
    penetration = 0.0
    for i in range(len(result)):
        data.qpos[:7] = result[i, :7]
        data.qpos[model.jnt_qposadr[joint_ids]] = result[i, 7:]
        mujoco.mj_forward(model, data)
        penetration = max(penetration, -min([c.dist for c in data.contact if model.geom("floor").id in c.geom] + [0.0]))
    original = source.stem.removesuffix("_mirrored")
    metadata = dict(
        dataset="RAI Institute / Boston Dynamics HOLD",
        license="CC-BY-4.0",
        revision=REVISION,
        source=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        recording_group=original,
        start=start,
        end=float(target_times[-1]),
        height_offset=height_offset,
        base_mapping="exported base -> local pelvis; provisional geometry audit below",
        annotation=dict(version=1, cutoff_hz=cutoff, max_anticipation_seconds=anticipation),
        audit=dict(
            max_sampled_penetration_m=penetration,
            joint_limit_violations=int(
                np.sum((result[:, 7:] < limits[:, 0] - 1e-5) | (result[:, 7:] > limits[:, 1] + 1e-5))
            ),
            joint_limit_tolerance_rad=1e-5,
            frames_checked=len(result),
            full_quality_validation=False,
        ),
    )
    np.savez_compressed(destination, **base, commands=commands, metadata=json.dumps(metadata))
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, nargs="+")
    p.add_argument("--clips", nargs="+", default=["walk/S0_normal_walk_1.h5"])
    p.add_argument("--output", type=Path, default=Path("data/hold"))
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--seconds", type=float)
    p.add_argument("--height-offset", type=float, default=0.0, help="Explicit, recorded vertical translation in metres")
    p.add_argument("--cutoff", type=float, default=3.0)
    p.add_argument("--anticipation", type=float, default=0.15)
    args = p.parse_args()
    sources = args.source or [fetch(c, args.output / "raw") for c in args.clips]
    for source in sources:
        out = args.output / f"{source.stem}.npz"
        metadata = convert(
            source,
            out,
            start=args.start,
            seconds=args.seconds,
            height_offset=args.height_offset,
            cutoff=args.cutoff,
            anticipation=args.anticipation,
        )
        print(f"{out}: {metadata['audit']}")


if __name__ == "__main__":
    main()

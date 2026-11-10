"""Download pinned Unitree LAFAN1 CSV and convert it to validated G1 references."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from zest.envs.g1.config import JOINT_NAMES
from zest.motion.library import validate_motion
from zest.utils.math import slerp

DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"


def fetch(relative, directory):
    target = directory / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{relative}", temporary)
        temporary.replace(target)
    return target


def convert(source, destination, *, input_fps=30.0, output_fps=50.0, start=0.0, seconds=None, downloaded=False):
    source, destination = Path(source), Path(destination)
    raw = np.loadtxt(source, delimiter=",", ndmin=2)
    if raw.shape[1] != 36 or len(raw) < 3 or not np.isfinite(raw).all():
        raise ValueError("Expected headerless Unitree G1 CSV: root xyz, quaternion xyzw, 29 joint angles")
    if input_fps <= 0 or output_fps <= 0 or start < 0 or (seconds is not None and seconds <= 0):
        raise ValueError("Invalid FPS or crop interval")
    qpos = raw.copy()
    qpos[:, 3:7] = raw[:, [6, 3, 4, 5]]
    lengths = np.linalg.norm(qpos[:, 3:7], axis=-1)
    if np.any(np.abs(lengths - 1) > 0.01):
        raise ValueError("Input contains invalid root quaternions")
    qpos[:, 3:7] /= lengths[:, None]
    end = (len(raw) - 1) / input_fps
    if seconds is not None:
        end = min(end, start + seconds)
    times = start + np.arange(int(np.floor((end - start) * output_fps)) + 1) / output_fps
    if len(times) < 3:
        raise ValueError("Crop must contain at least 3 output frames")
    index = times * input_fps
    lo = np.floor(index).astype(int)
    hi = np.minimum(lo + 1, len(qpos) - 1)
    f = (index - lo)[:, None]
    resampled = qpos[lo] * (1 - f) + qpos[hi] * f
    resampled[:, 3:7] = slerp(qpos[lo, 3:7], qpos[hi, 3:7], f)
    # Translation does not change the skill and keeps coordinates close to the origin.
    resampled[:, :2] -= resampled[0, :2]
    validate_motion(resampled, output_fps, JOINT_NAMES)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        source=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_fps=input_fps,
        output_fps=output_fps,
        start=start,
        end=float(times[-1]),
        dataset=DATASET if downloaded else "user-supplied Unitree-format CSV",
        revision=REVISION if downloaded else None,
        license="LAFAN1: CC-BY-NC-ND-4.0; see source dataset" if downloaded else "See source data terms",
    )
    np.savez_compressed(
        destination,
        version=1,
        qpos=resampled,
        fps=output_fps,
        joint_names=np.array(JOINT_NAMES),
        quaternion_order="wxyz",
        metadata=json.dumps(metadata),
    )
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local Unitree-format G1 CSV (skips download)")
    parser.add_argument("--clips", nargs="+", default=["walk1_subject1"])
    parser.add_argument("--all", action="store_true", help="Download and convert every G1 clip")
    parser.add_argument("--output", type=Path, default=Path("data/lafan1"))
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, help="Crop duration; default: full clip")
    args = parser.parse_args()
    if args.source:
        sources = [args.source]
    else:
        raw = args.output / "raw"
        fetch("README.md", raw)
        fetch("LICENSE", raw)
        clips = args.clips
        if args.all:
            with urllib.request.urlopen(
                f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}"
            ) as response:
                info = json.load(response)
            clips = [
                Path(s["rfilename"]).stem
                for s in info["siblings"]
                if s["rfilename"].startswith("g1/") and s["rfilename"].endswith(".csv")
            ]
        if any(Path(c).name != c for c in clips):
            parser.error("Clip names must be bare filenames without directories")
        sources = [fetch(f"g1/{c.removesuffix('.csv')}.csv", raw) for c in clips]
    for source in sources:
        out = convert(
            source,
            args.output / f"{source.stem}.npz",
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            start=args.start,
            seconds=args.seconds,
            downloaded=not args.source,
        )
        print(out)


if __name__ == "__main__":
    main()

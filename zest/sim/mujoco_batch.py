# SPDX-License-Identifier: BSD-3-Clause
"""CPU batch simulation, adapted from legged-gym-mujoco's model/batch structure."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mujoco import batch

from zest.envs.g1.config import BODY_NAMES, JOINT_NAMES

ASSET = Path(__file__).resolve().parents[1] / "resources/robots/g1/g1_mjlab.xml"


def asset_digest():
    digest = hashlib.sha256(ASSET.read_bytes())
    for mesh in sorted(ET.parse(ASSET).getroot().findall("asset/mesh"), key=lambda x: x.get("file")):
        digest.update((ASSET.parent / "meshes" / mesh.get("file")).read_bytes())
    return digest.hexdigest()


def load_model(cfg):
    root = ET.parse(ASSET).getroot()
    root.find("compiler").set("meshdir", str(ASSET.parent / "meshes"))
    ET.SubElement(
        root,
        "option",
        timestep=str(cfg.sim_dt),
        integrator="implicitfast",
        cone="elliptic",
        solver="Newton",
        iterations="30",
        gravity="0 0 -9.81",
    )
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        friction="0.8 0.005 0.0001",
        rgba="0.22 0.26 0.3 1",
        condim="3",
    )
    ET.SubElement(world, "light", name="task_light", pos="0 0 4", dir="0 0 -1", directional="true")
    for geom in world.iter("geom"):
        if geom.get("name", "").endswith("_collision"):
            geom.attrib.update(contype="1", conaffinity="1", condim="3", priority="1", friction="0.8 0.005 0.0001")
    armature, kp, kd, effort, _ = cfg.motor_parameters()
    actuators = ET.SubElement(root, "actuator")
    for name, arm, p, d, limit in zip(JOINT_NAMES, armature, kp, kd, effort):
        world.find(f".//joint[@name='{name}']").set("armature", str(arm))
        ET.SubElement(
            actuators,
            "position",
            name=name,
            joint=name,
            kp=str(p),
            kv=str(d),
            forcelimited="true",
            forcerange=f"{-limit} {limit}",
            ctrllimited="false",
        )
    sensors = root.find("sensor")
    for body in BODY_NAMES:
        for kind in ("framelinvel", "frameangvel"):
            ET.SubElement(sensors, kind, name=f"{kind}_{body}", objtype="xbody", objname=body)
        ET.SubElement(
            sensors,
            "contact",
            name=f"contact_{body}",
            body1=body,
            geom2="floor",
            data="force",
            reduce="netforce",
            num="1",
        )
    if hasattr(cfg, "stage"):
        ET.SubElement(
            sensors,
            "contact",
            name="ground_depth",
            subtree1="pelvis",
            geom2="floor",
            data="dist",
            reduce="mindist",
            num="1",
        )
        ET.SubElement(
            sensors,
            "contact",
            name="ground_peak",
            subtree1="pelvis",
            geom2="floor",
            data="force",
            reduce="maxforce",
            num="1",
        )
        ET.SubElement(
            sensors,
            "contact",
            name="feet_collision",
            body1="left_ankle_roll_link",
            body2="right_ankle_roll_link",
            data="found",
            num="1",
        )
        for body in BODY_NAMES[2:6]:
            ET.SubElement(
                sensors,
                "contact",
                name=f"wrench_{body}",
                body1=body,
                geom2="floor",
                data="force torque",
                reduce="netforce",
                num="1",
            )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def nominal_inertia(model):
    """Whole-body mass, torso-link inertia, and whole-body COM offset in the torso frame.

    S6 specifies whole-body mass and nominal base inertia, not the locked whole-body
    rotational inertia. Using the latter with a wrench on the articulated torso
    overestimates its rotational inertia and destabilizes the assistive controller.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = model.body("torso_link").id
    origin = data.xpos[root]
    torso_rotation = data.xmat[root].reshape(3, 3)
    rotation = torso_rotation.T @ data.ximat[root].reshape(3, 3)
    inertia = rotation @ np.diag(model.body_inertia[root]) @ rotation.T
    mass = model.body_mass.sum()
    com = torso_rotation.T @ (np.sum(model.body_mass[:, None] * data.xipos, axis=0) / mass - origin)
    return mass, inertia, com


def composite_inertia(model, nominal):
    """Locked whole-body inertia about torso origin, expressed in the torso frame."""
    data = mujoco.MjData(model)
    ids = [model.joint(n).id for n in JOINT_NAMES]
    data.qpos[model.jnt_qposadr[ids]] = nominal
    mujoco.mj_forward(model, data)
    torso = model.body("torso_link").id
    rotation = data.xmat[torso].reshape(3, 3).T
    offsets = (data.xipos - data.xpos[torso]) @ rotation.T
    inertia = np.zeros((3, 3))
    for i in range(1, model.nbody):
        r = rotation @ data.ximat[i].reshape(3, 3)
        delta = offsets[i]
        inertia += r @ np.diag(model.body_inertia[i]) @ r.T
        inertia += model.body_mass[i] * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    mass = model.body_mass.sum()
    return mass, inertia, (model.body_mass[:, None] * offsets).sum(0) / mass


class MujocoBatch:
    def __init__(self, cfg, rng):
        self.model = load_model(cfg)
        self.asset_digest = asset_digest()
        self.mass, self.inertia, self.com = nominal_inertia(self.model)
        self.batch = batch.Batch(self.model, nsim=cfg.num_envs, nthread=cfg.num_threads)
        self.body_ids = np.array([self.model.body(n).id for n in BODY_NAMES])
        self.joint_ids = np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.limits = self.model.jnt_range[self.joint_ids].copy()
        for name in (
            "qpos",
            "qvel",
            "ctrl",
            "xfrc_applied",
            "xpos",
            "xquat",
            "xipos",
            "sensordata",
            "qacc",
            "actuator_force",
        ):
            setattr(self, name, self.batch.bind(name))
        self.lin_cols = self._columns("framelinvel")
        self.ang_cols = self._columns("frameangvel")
        self.contact_cols = self._columns("contact")
        if cfg.randomize:
            friction = self.batch.expand("geom_friction")
            friction[:, :, 0] = rng.uniform(0.6, 1.0, (cfg.num_envs, 1))
            mass = self.batch.expand("body_mass")
            inertia = self.batch.expand("body_inertia")
            scales = rng.uniform(0.9, 1.1, mass.shape)
            mass *= scales
            inertia *= scales[..., None]
            self.batch.set_const()
        self.batch.reset()

    def _columns(self, prefix):
        return np.array(
            [
                np.arange(self.model.sensor(f"{prefix}_{n}").adr[0], self.model.sensor(f"{prefix}_{n}").adr[0] + 3)
                for n in BODY_NAMES
            ]
        )

    @property
    def linear_velocity(self):
        return self.sensordata[:, self.lin_cols]

    @property
    def angular_velocity(self):
        return self.sensordata[:, self.ang_cols]

    @property
    def contacts(self):
        # body1=robot, geom2=floor: the sensor gives force on the second body.
        # The critic requires the equal-and-opposite force acting on the robot.
        return -self.sensordata[:, self.contact_cols]

    def step(self, targets):
        self.ctrl[:] = targets
        self.batch.step(stage=mujoco.mjtStage.mjSTAGE_VEL)
        warnings = [
            int(mujoco.mjtWarning.mjWARN_BADQPOS),
            int(mujoco.mjtWarning.mjWARN_BADQVEL),
            int(mujoco.mjtWarning.mjWARN_BADQACC),
        ]
        if not np.isfinite(self.qpos).all() or np.any(self.batch.warning[:, warnings, 1] > 0):
            raise FloatingPointError("MuJoCo reported invalid dynamics")

    def forward(self, ids=None):
        self.batch.forward(ids)

    def reset(self, ids):
        self.batch.reset(ids, stage=mujoco.mjtStage.mjSTAGE_NONE)

    def close(self):
        for name in (
            "qpos",
            "qvel",
            "ctrl",
            "xfrc_applied",
            "xpos",
            "xquat",
            "xipos",
            "sensordata",
            "qacc",
            "actuator_force",
        ):
            setattr(self, name, None)
        self.batch = None

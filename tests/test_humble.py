import copy

import h5py
import mujoco
import numpy as np
import pytest
import torch

from zest.envs.g1.config import JOINT_NAMES
from zest.envs.g1.humble import G1Humble
from zest.envs.g1.humble_config import HumbleConfig, joint_parameters
from zest.motion.commands import CommandLibrary
from zest.scripts.hold_data import convert
from zest.scripts.humble_evaluate import evaluation_env
from zest.sim.mujoco_batch import composite_inertia, load_model
from zest.utils.humble_runner import HumbleRunner, config_for, load_policy, normalize_by_task, read_checkpoint


@pytest.fixture
def humble_motion(motion):
    with np.load(motion) as source:
        data = dict(source)
    data["commands"] = np.zeros((len(data["qpos"]), 3), dtype=np.float32)
    np.savez(motion, **data)
    return motion


def test_hold_timestamp_and_named_joint_conversion(tmp_path):
    source = tmp_path / "recording.h5"
    model = load_model(HumbleConfig(randomize=False))
    q = np.repeat(model.qpos0[None], 40, axis=0)
    t = np.arange(40) * 0.007453
    q[:, 0] = 2 * t
    q[:, 7:] = np.arange(29) * 0.001
    with h5py.File(source, "w") as f:
        f["robot_data/meta_data/quat_order"] = np.array(list("wxyz"), dtype="S1")
        root = "robot_data/trajectory_data/"
        f[root + "time"] = t + 17  # Absolute export timestamps must be rebased.
        f[root + "bodies/labels"] = np.array(["base"], dtype="S4")
        f[root + "bodies/pos_w"] = q[:, None, :3]
        f[root + "bodies/quat_w"] = q[:, None, 3:7]
        f[root + "joints/labels"] = np.array(JOINT_NAMES[::-1], dtype="S40")
        f[root + "joints/pos"] = q[:, 7:][:, ::-1]
    destination = tmp_path / "out.npz"
    meta = convert(source, destination, height_offset=0.01, anticipation=0)
    with np.load(destination) as result:
        np.testing.assert_allclose(result["qpos"][:, 0], np.arange(len(result["qpos"])) * 0.04, atol=1e-10)
        np.testing.assert_allclose(result["qpos"][0, 7:], q[0, 7:])
        assert result["commands"].shape[1] == 3
    assert meta["height_offset"] == 0.01
    with pytest.raises(FileExistsError):
        convert(source, destination)


def test_composite_inertia_matches_rigid_rotation_energy():
    model = load_model(HumbleConfig(randomize=False))
    nominal = joint_parameters()[2]
    mass, inertia, com = composite_inertia(model, nominal)
    data = mujoco.MjData(model)
    ids = [model.joint(n).id for n in JOINT_NAMES]
    data.qpos[model.jnt_qposadr[ids]] = nominal
    mujoco.mj_forward(model, data)
    torso = model.body("torso_link").id
    rotation = data.xmat[torso].reshape(3, 3)
    omega = np.array([0.5, -0.8, 0.2])
    world_omega = rotation @ omega
    energy = 0.0
    for i in range(1, model.nbody):
        w_local = data.ximat[i].reshape(3, 3).T @ world_omega
        v = np.cross(world_omega, data.xipos[i] - data.xpos[torso])
        energy += 0.5 * (np.sum(model.body_inertia[i] * w_local**2) + model.body_mass[i] * (v @ v))
    np.testing.assert_allclose(0.5 * omega @ inertia @ omega, energy, rtol=1e-12)
    assert mass > 20 and np.all(np.linalg.eigvalsh(inertia) > 0) and np.isfinite(com).all()


def test_observation_privilege_and_nominal_actions(humble_motion):
    env = G1Humble(
        [humble_motion],
        HumbleConfig(num_envs=4, num_threads=1, stage="finetune", randomize=False, observation_noise=False),
    )
    try:
        old = env.compute_observations()
        env.library.clips[0].values["qpos"][:, 7:] += 0.1
        new = env.compute_observations()
        torch.testing.assert_close(old["student"], new["student"])
        assert not torch.equal(old["teacher"], new["teacher"])
        assert old["student"].shape == (4, 96)
        np.testing.assert_allclose(env.joint_targets(None), np.tile(joint_parameters()[2], (4, 1)))
        assert not env.beta.any()
        env.sim.qpos[:, 2] -= 1
        env.sim.forward()
        assert np.all(env.sensor("ground_depth") < -0.03)
        env.cfg.observation_noise = True
        a, b = env.compute_observations(), env.compute_observations()
        torch.testing.assert_close(a["critic"], b["critic"])
        assert not torch.equal(a["student"], b["student"])
    finally:
        env.close()


def test_task_advantages_do_not_mix_reward_scales():
    advantages = torch.tensor([[[1.0], [100.0]], [[3.0], [300.0]]])
    tasks = torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])
    result = normalize_by_task(advantages, tasks)
    torch.testing.assert_close(result, torch.tensor([[[-1.0], [-1.0]], [[1.0], [1.0]]]))


def test_goal_initialization_uses_nearest_supported_commands(humble_motion, model):
    library = CommandLibrary([humble_motion], model)
    clips, times = library.initial_states(np.array([[99.0, -100.0, 50.0]]), np.random.default_rng(1))
    assert clips[0] == 0 and 0 <= times[0] <= library.durations[0]


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_all_learning_stages_freeze_transfer_resume(humble_motion, tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(19)
    torch.set_num_threads(2)
    previous = None
    for stage in ("teacher", "distill", "warmup", "finetune"):
        cfg = HumbleConfig(
            num_envs=2, num_threads=1, stage=stage, episode_seconds=0.06, randomize=False, observation_noise=False
        )
        env = G1Humble([humble_motion], cfg)
        try:
            train = config_for(stage, tiny=True)
            train["num_steps_per_env"] = 4
            runner = HumbleRunner(env, train, device)
            if previous:
                runner.initialize(previous)
            before = copy.deepcopy(runner.alg.actor.state_dict())
            critic_before = copy.deepcopy(runner.alg.critic.state_dict())
            expert = copy.deepcopy(runner.expert.state_dict()) if runner.expert else None
            metrics = runner.collect_update()
            assert all(np.isfinite(value) for value in metrics.values())
            after = runner.alg.actor.state_dict()
            if stage == "teacher":
                frozen = copy.deepcopy(after)
                simulation = env.sim.qpos.copy()
                report = runner.evaluate_references(tmp_path)
                assert 0 <= report["Unassisted/completion"] <= 1
                assert all(torch.equal(frozen[k], runner.alg.actor.state_dict()[k]) for k in frozen)
                np.testing.assert_array_equal(simulation, env.sim.qpos)
            if stage == "warmup":
                assert all(torch.equal(before[k], after[k]) for k in before)
                assert any(not torch.equal(critic_before[k], runner.alg.critic.state_dict()[k]) for k in critic_before)
            else:
                assert any(not torch.equal(before[k], after[k]) for k in before if "mlp" in k)
            if expert:
                assert all(torch.equal(expert[k], runner.expert.state_dict()[k]) for k in expert)
            checkpoint = tmp_path / f"{device}_{stage}.pt"
            runner.iteration = 1
            runner.save(checkpoint)
            resumed = HumbleRunner(env, train, "cpu")
            resumed.resume(checkpoint)
            assert resumed.iteration == 1
            for k, value in after.items():
                torch.testing.assert_close(value.cpu(), resumed.alg.actor.state_dict()[k].cpu())
            previous = checkpoint
        finally:
            env.close()
    saved = read_checkpoint(previous)
    env = evaluation_env(saved, num_envs=1, commands=True)
    try:
        policy = load_policy(previous, env, "cpu")
        env.set_command([0.2, 0, 0.1])
        env.stand()
        with torch.inference_mode():
            _, reward, _, _ = env.step(policy(env.obs))
        assert torch.isfinite(reward).all()
    finally:
        env.close()

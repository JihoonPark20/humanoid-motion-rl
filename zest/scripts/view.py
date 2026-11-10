"""Viser playback for G1 references and ZEST/HuMBLE policies; no training updates."""

import argparse
import queue
import signal
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from zest.envs.g1.config import G1Config
from zest.envs.g1.tracking import G1Tracking
from zest.scripts.humble_evaluate import evaluation_env
from zest.utils.humble_runner import load_policy
from zest.utils.runner import ZestRunner, resolve_device


class Playback:
    """Single-thread-owned simulation state, independent of the web UI."""

    def __init__(self, checkpoint=None, motion=None, device="cpu", reference=False):
        self.device, self.policy = device, None
        self.command_policy = False
        self.family = "Reference"
        if checkpoint:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "humble" in saved:
                stage = saved["humble"]["env_config"]["stage"]
                self.family = f"HuMBLE / {stage}"
                self.command_policy = stage != "teacher" and motion is None
                self.env = evaluation_env(
                    saved, num_envs=1, threads=1, commands=self.command_policy, motions=[motion] if motion else None
                )
                try:
                    self.policy = load_policy(checkpoint, self.env, device)
                except Exception:
                    self.env.close()
                    raise
            else:
                self.family = "ZEST"
                meta = saved["infos"]["zest"]
                cfg = G1Config(**meta["env_config"])
                cfg.num_envs = cfg.num_threads = 1
                cfg.assistance = cfg.randomize = cfg.observation_noise = False
                motions = [motion] if motion else [m["path"] for m in meta["contract"]["motions"]]
                self.env = G1Tracking(motions, cfg, evaluation=True)
                try:
                    runner = ZestRunner(self.env, meta["train_config"], device=device)
                    runner.load(checkpoint, restore_training=False)
                    self.policy = runner.get_inference_policy()
                except Exception:
                    self.env.close()
                    raise
        else:
            if motion is None:
                raise ValueError("Provide --motion or --checkpoint")
            self.env = G1Tracking(
                [motion],
                G1Config(num_envs=1, num_threads=1, assistance=False, randomize=False, observation_noise=False),
                evaluation=True,
            )
        self.reference_available = not self.command_policy
        if reference and not self.reference_available:
            self.env.close()
            raise ValueError("Reference playback needs --motion for a command-only student")
        self.mode = "Reference" if reference or self.policy is None else "Policy"
        self.data = mujoco.MjData(self.env.sim.model)
        self.clip_id, self.time, self.failures = 0, 0.0, 0
        self.reset()

    @property
    def duration(self):
        return float(self.env.library.durations[self.clip_id])

    def reset(self):
        self.time, self.failures = 0.0, 0
        if self.command_policy:
            self.env.stand()
        else:
            self.env.reset(clip_ids=np.array([self.clip_id]))
        self.refresh()

    def refresh(self):
        if self.mode == "Reference":
            ref = self.env.library.sample(np.array([self.clip_id]), np.array([self.time]))
            self.data.qpos[:7] = ref["qpos"][0, :7]
            self.data.qpos[self.env.sim.qpos_ids] = ref["qpos"][0, 7:]
            self.data.qvel[:6] = ref["qvel"][0, :6]
            self.data.qvel[self.env.sim.qvel_ids] = ref["qvel"][0, 6:]
        else:
            self.data.qpos[:] = self.env.sim.qpos[0]
            self.data.qvel[:] = self.env.sim.qvel[0]
        mujoco.mj_forward(self.env.sim.model, self.data)

    def advance(self, command=(0.5, 0.0, 0.0)):
        if self.mode == "Reference":
            self.time = (self.time + self.env.dt) % self.duration
        else:
            if self.command_policy:
                self.env.set_command(command)
            with torch.inference_mode():
                _, _, done, info = self.env.step(self.policy(self.env.obs.to(self.device)))
            self.time += self.env.dt
            self.failures += int(info["failed"].sum())
            if done.any():
                if self.command_policy:
                    self.env.stand()
                else:
                    self.env.reset(clip_ids=np.array([self.clip_id]))
        self.refresh()

    def close(self):
        self.env.close()


def serve(playback, host="127.0.0.1", port=8080, paused=False, duration=None):
    try:
        import viser
        from mjviser import ViserMujocoScene
    except ImportError as error:
        raise RuntimeError("Viser playback needs Python 3.11+ and: uv sync --extra viewer") from error

    server = viser.ViserServer(host=host, port=port, label="G1 Motion Lab")
    events = queue.SimpleQueue()
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        server.gui.configure_theme(control_width="medium", show_share_button=False)
        server.gui.main_panel.dock_right()
        server.scene.set_up_direction("+z")
        scene = ViserMujocoScene(server, playback.env.sim.model, num_envs=1)
        with server.gui.add_folder("Playback", expand_by_default=True):
            server.gui.add_markdown(f"**{playback.family}** · Unitree G1\n\nPlayback only · assistance off")
            options = (["Reference"] if playback.reference_available else []) + (["Policy"] if playback.policy else [])
            mode = server.gui.add_dropdown("Mode", options, initial_value=playback.mode)
            names = [f"{i}: {clip.path.stem}" for i, clip in enumerate(playback.env.library.clips)]
            clip = server.gui.add_dropdown("Motion", names, visible=playback.reference_available)
            playing = server.gui.add_checkbox("Playing", initial_value=not paused)
            reset = server.gui.add_button("Reset")
            speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
            timeline = server.gui.add_slider(
                "Time (s)",
                min=0.0,
                max=playback.duration,
                step=playback.env.dt,
                initial_value=0.0,
                disabled=playback.mode != "Reference",
                visible=playback.reference_available,
            )
            follow = server.gui.add_checkbox("Follow robot", initial_value=True)
            status = server.gui.add_markdown("")
        command = []
        if playback.command_policy:
            with server.gui.add_folder("Velocity command"):
                for label, low, high, value in (
                    ("Forward (m/s)", -0.8, 1.2, 0.5),
                    ("Sideways (m/s)", -0.6, 0.6, 0.0),
                    ("Turn (rad/s)", -1.5, 1.5, 0.0),
                ):
                    command.append(server.gui.add_slider(label, min=low, max=high, step=0.05, initial_value=value))
        scene.create_visualization_gui()

        # GUI callbacks run in background threads. Only enqueue requests; the
        # main loop owns MuJoCo and PyTorch and applies them between steps.
        @mode.on_update
        def on_mode(event):
            if event.client_id is not None:
                events.put(("mode", mode.value))

        @clip.on_update
        def on_clip(event):
            if event.client_id is not None:
                events.put(("clip", names.index(clip.value)))

        @reset.on_click
        def on_reset(event):
            events.put(("reset", None))

        @timeline.on_update
        def on_time(event):
            if event.client_id is not None:
                events.put(("time", timeline.value))

        @server.on_client_connect
        def on_connect(client):
            client.camera.position = (2.5, -3.0, 1.8)
            client.camera.look_at = (0.0, 0.0, 0.8)

        print(f"Viser playback: http://{host}:{server.get_port()} (no training)", flush=True)
        begin = last = time.monotonic()
        accumulator = 0.0
        camera_origin = playback.data.qpos[:3].copy()
        while not stopped and (duration is None or time.monotonic() - begin < duration):
            now = time.monotonic()
            elapsed, last = min(now - last, 0.1), now
            while not events.empty():
                kind, value = events.get()
                if kind == "mode":
                    playback.mode = value
                    playback.reset()
                elif kind == "clip":
                    playback.clip_id = value
                    playback.reset()
                    timeline.max = playback.duration
                elif kind == "time" and playback.mode == "Reference":
                    playing.value = False
                    playback.time = float(np.clip(value, 0.0, playback.duration))
                    playback.refresh()
                elif kind == "reset":
                    playback.reset()
                accumulator = 0.0
            timeline.disabled = playback.mode != "Reference"
            if playing.value:
                accumulator += elapsed * speed.value
                for _ in range(min(int(accumulator / playback.env.dt), 10)):
                    playback.advance(tuple(slider.value for slider in command) if command else (0.5, 0.0, 0.0))
                    accumulator -= playback.env.dt
            else:
                accumulator = 0.0
            with server.atomic():
                scene.update_from_mjdata(playback.data)
                if playback.mode == "Reference":
                    timeline.value = playback.time
                status.content = f"Time: {playback.time:.2f} s · Failures: {playback.failures}"
                delta = playback.data.qpos[:3] - camera_origin
                if follow.value:
                    for client in server.get_clients().values():
                        client.camera.position = np.asarray(client.camera.position) + delta
                        client.camera.look_at = np.asarray(client.camera.look_at) + delta
                camera_origin = playback.data.qpos[:3].copy()
            time.sleep(1 / 60)
    finally:
        server.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--motion", type=Path)
    p.add_argument("--reference", action="store_true", help="Start in kinematic reference mode")
    p.add_argument("--paused", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--duration", type=float, help="Optional wall-clock viewer lifetime in seconds")
    args = p.parse_args()
    if not args.motion and not args.checkpoint:
        p.error("Provide --motion or --checkpoint")
    if not 0 <= args.port <= 65535 or (args.duration is not None and args.duration <= 0):
        p.error("Invalid port or duration")
    torch.set_num_threads(2)
    playback = Playback(args.checkpoint, args.motion, resolve_device(args.device), args.reference)
    try:
        serve(playback, args.host, args.port, args.paused, args.duration)
    finally:
        playback.close()


if __name__ == "__main__":
    main()

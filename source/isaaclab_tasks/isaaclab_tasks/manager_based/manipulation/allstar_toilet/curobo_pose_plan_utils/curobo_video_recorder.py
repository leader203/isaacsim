from __future__ import annotations

import datetime
from pathlib import Path


class ObserverVideoRecorder:
    def __init__(self, env, *, output_dir: str | Path, fps: int, every_n_steps: int):
        self.env = env
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.fps = max(1, int(fps))
        self.every_n_steps = max(1, int(every_n_steps))
        self.frame_count = 0
        self.step_count = 0
        self.writer = None
        self.path = None

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"run_curobo_pose_plan_test_observer_{timestamp}.mp4"
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=8)
        print(f"[Video] recording observer view to {self.path}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            print(f"[Video] saved observer mp4: {self.path} frames={self.frame_count}", flush=True)

    def after_step(self):
        self.step_count += 1
        if self.step_count % self.every_n_steps == 0:
            self.capture()

    def capture(self, *, force: bool = False):
        if self.writer is None:
            return
        frame = self.env.render(recompute=force)
        if frame is None:
            return
        self.writer.append_data(frame)
        self.frame_count += 1


def attach_video_recorder(
    env,
    *,
    record_video: bool,
    output_dir: str | Path,
    fps: int,
    every_n_steps: int,
) -> ObserverVideoRecorder | None:
    if not record_video:
        return None
    recorder = ObserverVideoRecorder(
        env,
        output_dir=output_dir,
        fps=fps,
        every_n_steps=every_n_steps,
    )
    original_step = env.step

    def recording_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        recorder.after_step()
        return result

    env.step = recording_step
    recorder.__enter__()
    return recorder

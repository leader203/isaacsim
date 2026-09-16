# RecordWrapper — manual demonstration recording.
# Ported from allstar_sim; ROS2 dependency removed.
# Drives recorder_manager to export HDF5 episodes on operator command.

import os

from ..utils.state_machine import LoopStateMachine, StepperStateMachine

from .wrapper_base import Any, ManagerBasedEnvCfg, Wrapper, torch

_CONFIG_SUB_NAME = "record"
_CONFIG_DATASET_FILE = "dataset_file"
_CONFIG_NUM_DEMOS = "num_demos"


class RecordWrapper(Wrapper):
    """Manual demonstration recording wrapper.

    Responsibilities:
    1. Configure RecorderManager export path before env creation.
    2. Manage recording state machine at runtime.
    3. Concatenate teleop action + force-control before passing to env.
    4. Drive recorder_manager to export successful episodes.

    Key bindings (exposed through teleop_func for callers that register them):
        R  — reset environment
        P  — advance task state machine (prepare → start → deal → finish)
        M  — toggle constant z-axis force injection
    """

    reset_state: StepperStateMachine
    task_state: StepperStateMachine
    force_control_state: LoopStateMachine
    teleop_func: dict
    current_recorded_demo_count: int

    @staticmethod
    def crop_env_cfg(env_cfg: ManagerBasedEnvCfg, task_cfg: dict, **kwargs):
        task_cfg = task_cfg.get(_CONFIG_SUB_NAME, {})

        if not isinstance(task_cfg.get(_CONFIG_DATASET_FILE), str):
            raise ValueError("Must have dataset_file as a str, file path to export recorded demos.")
        dataset_file = task_cfg[_CONFIG_DATASET_FILE]
        output_dir = os.path.dirname(dataset_file)
        output_file_name = os.path.splitext(os.path.basename(dataset_file))[0]
        setattr(env_cfg.recorders, "dataset_export_dir_path", output_dir)
        setattr(env_cfg.recorders, "dataset_filename", output_file_name)

        if not isinstance(task_cfg.get(_CONFIG_NUM_DEMOS), int):
            raise ValueError("Must have num_demos as an int. Set to 0 for unlimited.")

        if hasattr(env_cfg, "crop_4_command"):
            env_cfg.crop_4_command("record")

    def __init__(self, env: Any, task_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(env, task_cfg.get(_CONFIG_SUB_NAME, {}) if task_cfg else task_cfg)

        self.reset_state = StepperStateMachine(["running", "resetting"])
        self.task_state = StepperStateMachine(["prepare", "start", "deal", "finish"])
        self.force_control_state = LoopStateMachine(["disable", "enable"])

        self.teleop_func = getattr(self._env, "teleop_func", {})
        self.teleop_func.update({
            self.reset_state.next: {"keyboard": "R", "spacemouse": "R", "gamepad": "R", "handtracking": "R"},
            self.task_state.next: {"keyboard": "P", "spacemouse": "L", "gamepad": "L", "handtracking": "P"},
            self.force_control_state.next: {"keyboard": "M"},
        })

        self.current_recorded_demo_count = 0

    def step(self, action: torch.Tensor, **kwargs):
        force_control = torch.zeros(self._env.num_envs, 6).to(self._env.device)
        if self.force_control_state.state == "enable":
            force_control[..., :] = torch.Tensor([0, 0, 5, 0, 0, 0])

        self._unwrapped.cfg.last_force_control = force_control  # type: ignore

        action_dim = self._unwrapped.action_manager.total_action_dim
        if action_dim == action.shape[-1] + force_control.shape[-1]:
            full_action = torch.concat([action, force_control], dim=-1)
        else:
            full_action = action[..., :action_dim]

        obs, reward, terminated, truncated, extra = self._env.step(full_action)

        if self.task_state.state == "prepare":
            self._unwrapped.recorder_manager.reset()
        elif self.task_state.state == "start":
            self._unwrapped.recorder_manager.reset()
            self._unwrapped.recorder_manager.record_post_reset([0])
            self.task_state.next()
        elif self.task_state.state == "deal":
            pass
        elif self.task_state.state == "finish":
            self._unwrapped.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            self._unwrapped.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=self._env.device)
            )
            self._unwrapped.recorder_manager.export_episodes([0])
            self.reset_state.next()

        if self.reset_state.state == "resetting":
            self._unwrapped.recorder_manager.reset()
            self._env.reset()
            self.task_state.reset()
            self.reset_state.reset()

        exported = self._unwrapped.recorder_manager.exported_successful_episode_count
        if exported > self.current_recorded_demo_count:
            self.current_recorded_demo_count = exported
            print(f"Recorded {self.current_recorded_demo_count} successful demonstrations.")

        if self._task_cfg[_CONFIG_NUM_DEMOS] > 0 and exported >= self._task_cfg[_CONFIG_NUM_DEMOS]:
            extra.update({
                "task_finished": True,
                "task_description": f"All {self._task_cfg['num_demos']} demonstrations recorded.",
            })

        return obs, extra

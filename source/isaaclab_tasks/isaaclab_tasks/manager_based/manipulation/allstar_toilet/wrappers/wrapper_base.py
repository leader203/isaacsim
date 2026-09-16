# Lightweight wrapper base class for allstar toilet task.
# Replaces allstar_sim.base.envs.Wrapper with a standalone version
# that depends only on isaaclab and standard Python.

import torch
from typing import Any, Tuple

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import ManagerBasedRLEnvCfg as ManagerBasedEnvCfg  # type alias for convenience


class Wrapper:
    """Base wrapper for Isaac Lab environments in the allstar toilet task.

    Provides the same chaining interface as the original allstar_sim Wrapper
    but without ROS2 or allstar_sim infrastructure dependencies.

    Attributes:
        _env: Directly wrapped object (env or inner wrapper).
        _unwrapped: The innermost Isaac Lab env, bypassing all wrappers.
        _task_cfg: Config fragment consumed by this wrapper.
    """

    _env: Any
    _unwrapped: ManagerBasedEnv
    _task_cfg: dict

    @staticmethod
    def crop_env_cfg(env_cfg: ManagerBasedEnvCfg, task_cfg: dict, **kwargs):
        """Pre-instantiation hook: modify env_cfg before the env is created."""
        pass

    def __init__(self, env: Any, task_cfg: dict | None = None, **kwargs) -> None:
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except Exception:
            self._unwrapped = env
        self._task_cfg = task_cfg if task_cfg is not None else {}

    def __getattr__(self, key: str) -> Any:
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def unwrapped(self) -> Any:
        return self._unwrapped

    def reset(self, **kwargs) -> Tuple[torch.Tensor, Any]:
        return self._env.reset()

    def step(self, action: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Any]:
        obs, reward, terminated, truncated, extra = self._env.step(action)
        return obs, extra

    def close(self) -> None:
        return self._env.close()

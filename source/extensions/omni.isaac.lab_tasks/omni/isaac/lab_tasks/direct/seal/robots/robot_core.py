# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.assets import Articulation


class RobotCore:
    def __init__(
        self,
        robot_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
    ):
        """Initializes the robot core.

        Args:
            robot_cfg: The configuration of the robot.
            robot_uid: The unique id of the robot.
            num_envs: The number of environments.
            device: The device on which the tensors are stored."""

        # Unique task identifier, used to differentiate between tasks with the same name
        self._robot_uid = robot_uid
        # Number of environments and device to be used
        self._num_envs = num_envs
        self._device = device

        # Robot
        self._robot: Articulation = MISSING

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        # Buffers
        if env_ids is None:
            self._env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        else:
            self._env_ids = env_ids

    def run_setup(self, robot: Articulation) -> None:
        """Loads the robot into the task. After it has been loaded."""
        self._robot = robot

    ##
    # Derived properties.
    ##

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation root's center of mass frame.
        """
        return self._robot.data.root_state_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the articulation root.
        """
        return self._robot.data.root_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the articulation root.
        """
        return self._robot.data.root_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of
        mass frame.
        """
        return self._robot.data.root_vel_w

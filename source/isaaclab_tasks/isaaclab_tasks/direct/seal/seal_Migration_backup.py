# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math

from .seal_core import SealEnv, SealEnvCfg

from isaaclab.utils.math import subtract_frame_transforms, quat_from_matrix, normalize, quat_from_euler_xyz


@torch.jit.script
def calculate_desired_orientation(robot_pos: torch.Tensor, goal_pos: torch.Tensor, device: str) -> torch.Tensor:
    """
    Calculate the desired orientation quaternion from the robot's position to the goal position.

    Parameters:
    - robot_pos: The current position of the robot. Shape is (N, 3).
    - goal_pos: The goal position. Shape is (N, 3).
    - device: The device on which tensors are allocated.

    Returns:
    - desired_orientation: The desired orientation quaternion. Shape is (N, 4).
    """
    # Calculate the direction vector from the robot to the goal
    direction_vector = goal_pos - robot_pos
    direction_vector = normalize(direction_vector)  # Normalize

    # Create the rotation matrix
    forward_vector = torch.tensor([1, 0, 0], device=device).expand_as(direction_vector)
    up_vector = torch.tensor([0, 0, 1], device=device).expand_as(direction_vector)
    right_vector = torch.cross(forward_vector, up_vector, dim=1)

    # Combine the vectors to create a rotation matrix
    rotation_matrix = torch.stack([right_vector, up_vector, direction_vector], dim=-1)

    # Convert the rotation matrix to a quaternion
    desired_orientation = quat_from_matrix(rotation_matrix)

    return desired_orientation


class SealMigrationEnvCfg(SealEnvCfg):
    # Here you can define task-specific configurations
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space_task = (
            3  # Task-specific observations. For Migration task, we need to observe the relative desired position in the body frame, X, Y, Z
        )
        self.observation_space = self.observation_space_robot + self.observation_space_task
        self.time_reward_scale = -10.0
        self.prog_reward_scale = 1.0
        self.perc_reward_scale = 0.02
        self.ang_diff_constant = -10.0
        self.cmd_reward_scale = -1e-4
        self.crash_reward_scale = -5.0  # default -5.0


class SealMigrationEnv(SealEnv):
    cfg: SealMigrationEnvCfg

    def __init__(self, cfg: SealEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "time_penalty",  # time penalty
                "prog",  # progress towards goal
                "perc",  # angle between current and desired direction
                "cmd",  # for smooth action
                "crash",  # out of bounds
            ]
        }

        self.distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) + 1e6
        # Initialize previous distance and action values
        self._prev_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Initialize target orientation (roll, pitch, yaw)
        self._goal_orientation = torch.zeros(self.num_envs, 3, device=self.device)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )

        elevation = self._robot.data.root_link_state_w[:, 2]
        # change elevation dimension to 2
        elevation = elevation.unsqueeze(1).expand(-1, 2)

        # print(f"body_com_state_w: ", self._robot.data.body_com_state_w)

        joint_pos_indices = [self._robot.find_joints(name)[0][0] for name in self.cfg.joints_pos_required]
        joint_vel_indices = [self._robot.find_joints(name)[0][0] for name in self.cfg.joints_vel_required]
        joint_pos_obs = self._robot.data.joint_pos[:, joint_pos_indices]
        joint_vel_obs = self._robot.data.joint_vel[:, joint_vel_indices]

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                elevation,
                joint_pos_obs,
                joint_vel_obs,
                desired_pos_b,
                self._previous_actions,
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Compute distance to goal
        self.distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)

        # Compute progress reward
        r_prog = torch.where(self._prev_distance > 0.0, (self._prev_distance - self.distance_to_goal) * self.cfg.prog_reward_scale, 0.0)

        # Update the previous distance to goal
        self._prev_distance = self.distance_to_goal

        # Compute smooth action reward
        action_diff = torch.norm(self._actions - self._previous_actions, dim=1)
        r_cmd = self.cfg.cmd_reward_scale * (action_diff**2)

        # Update the previous actions
        self._previous_actions = self._actions.clone()

        # Compute crash penalty
        crash_penalty = torch.where(self._robot.data.root_pos_w[:, 2] < -9.0, self.cfg.crash_reward_scale, 0.0)

        # time penalty
        r_time = self.cfg.time_reward_scale * torch.ones_like(r_prog)

        # Total reward
        rewards = {
            "time_penalty": r_time * self.step_dt,
            "prog": r_prog,
            "cmd": r_cmd,
            "crash": crash_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bound = torch.logical_or(self._robot.data.root_pos_w[:, 2] < -10.0, self._robot.data.root_pos_w[:, 2] > 10.0)
        goal_reached = torch.logical_and(self.distance_to_goal > 0.0, self.distance_to_goal < 0.5)
        died = torch.logical_or(out_of_bound, goal_reached)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._prev_distance[env_ids] = torch.zeros(len(env_ids), device=self.device)

        # Sample new target positions
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(99.5, 100.5)
        self._desired_pos_w[env_ids, 1] = torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(-0.5, 0.5)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(-3.0, 3.0)

        default_root_state = self._robot.data.default_root_state[env_ids]
        # Randomize the robot's initial position
        random_offsets = torch.zeros_like(self._terrain.env_origins[env_ids])
        random_offsets[:, 0] = torch.FloatTensor(len(env_ids)).uniform_(-100.5, -99.5)  # Randomize x position
        random_offsets[:, 1] = torch.FloatTensor(len(env_ids)).uniform_(-0.5, 0.5)  # Randomize y position
        random_offsets[:, 2] = torch.FloatTensor(len(env_ids)).uniform_(-5.0, 0.0)  # Randomize z position
        default_root_state[:, :3] += self._terrain.env_origins[env_ids] + random_offsets

        # Initialize orientation with roll pitch, and yaw
        random_yaw = torch.zeros_like(self._goal_orientation[env_ids, 2]).uniform_(-math.pi, math.pi)
        random_pitch = torch.zeros_like(self._goal_orientation[env_ids, 1]).uniform_(-math.pi, math.pi)
        random_roll = torch.zeros_like(self._goal_orientation[env_ids, 0]).uniform_(-math.pi, math.pi)
        initial_orientation = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
        default_root_state[:, 3:7] = initial_orientation

        """
        # Add small random velocities
        random_lin_vel = torch.zeros_like(default_root_state[:, 7:10]).uniform_(-0.5, 0.5)
        random_ang_vel = torch.zeros_like(default_root_state[:, 10:13]).uniform_(-0.1, 0.1)
        
        default_root_state[:, 7:10] += random_lin_vel
        default_root_state[:, 10:13] += random_ang_vel
        """

        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)

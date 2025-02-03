# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat, matrix_from_quat

"""
Following Fossen's Equation,
Fossen, T. I. (1991). Nonlinear modeling and control of Underwater Vehicles. Doctoral thesis, Department of Engineering Cybernetics, Norwegian Institute of Technology (NTH), June 1991.
"""


@configclass
class BuoyancyCfg:
    """Configuration for default buoyancy."""

    gravity: float = -9.81  # m/s^2
    mass: float = MISSING  # Kg


class Buoyancy:
    def __init__(self, num_envs, device, cfg: BuoyancyCfg):

        self.cfg = cfg

        self.num_envs = num_envs
        self.device = device

        # Buoyancy
        self.archimedes_force_global = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_torque_global = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_force_local = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_torque_local = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        return

    def compute_archimedes_metacentric_global(self, rpy):
        roll, pitch = rpy[:, 0], rpy[:, 1]  # roll and pitch are given in global frame

        # compute buoyancy force
        self.archimedes_force_global[:, 2] = -self.cfg.gravity * self.cfg.mass

        # TODO: now, all torques are zero. Implement the torque calculation for COB change.
        self.archimedes_torque_global = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        return self.archimedes_force_global, self.archimedes_torque_global

    def compute_archimedes_metacentric_local(self, position, quaternions):

        roll, pitch, yaw = euler_xyz_from_quat(quaternions)
        euler = torch.stack((roll, pitch, yaw), dim=1)

        self.compute_archimedes_metacentric_global(euler)
        # print computed self.archimedes_force_global
        # print(f"self.archimedes_force_global: {self.archimedes_force_global}")

        # get rotation matrix from quaternions in world frame, size is (3*num_envs, 3)
        R = matrix_from_quat(quaternions)
        # print computed R
        # print(f"R: {R}")

        # Arobot = Rworld * Aworld. Resulting matrix should be size (3*num_envs, 3) * (num_envs,3) =(num_envs,3)
        self.archimedes_force_local = torch.bmm(
            R.mT, torch.unsqueeze(self.archimedes_force_global, 1).mT
        )  # add batch dimension to tensor and transpose it
        self.archimedes_force_local = self.archimedes_force_local.mT.squeeze(1)  # remove batch dimension to tensor
        # print computed self.archimedes_force_local
        # print(f"self.archimedes_force_local: {self.archimedes_force_local}")

        # TODO: now, all torques are zero. Implement the torque calculation for COB change.
        self.archimedes_torque_local = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        return torch.hstack(
            [
                self.archimedes_force_local,
                self.archimedes_torque_local,
            ]
        )

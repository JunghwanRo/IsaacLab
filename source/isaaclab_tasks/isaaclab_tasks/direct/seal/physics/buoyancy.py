# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from math import pi
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat

"""
Following Fossen's Equation,
Fossen, T. I. (1991). Nonlinear modeling and control of Underwater Vehicles. Doctoral thesis, Department of Engineering Cybernetics, Norwegian Institute of Technology (NTH), June 1991.
"""


@configclass
class BuoyancyCfg:
    """Configuration for default buoyancy."""

    gravity: float = -9.81  # m/s^2
    mass: float = MISSING  # Kg
    total_water_mass: float = MISSING  # Kg
    buoyancy_spheres: list[tuple[str, float]] = MISSING  # [(sphere_name, sphere_volume)]


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

        # Compute the buoyancy sphere radius, depend on the total_water_mass and the volume of each sphere.
        buoyancy_spheres = self.cfg.buoyancy_spheres
        # convert replaced water mass to m^3
        total_water_volume = self.cfg.total_water_mass / 1000
        # Add all relative volume of spheres
        total_relative_volume = sum([sphere[1] for sphere in buoyancy_spheres])
        self.volume_list = [sphere[1] * total_water_volume / total_relative_volume for sphere in buoyancy_spheres]
        # Compute the radius of each sphere. radius = ((3*volume)/(4*pi))^(1/3)
        self.radius_list = [((3 * volume) / (4 * pi)) ** (1 / 3) for volume in self.volume_list]

        return

    def compute_archimedes_metacentric_global(self, rpy, positions_z, idx, charged_water_mass=None):
        roll, pitch = rpy[:, 0], rpy[:, 1]  # roll and pitch are given in global frame

        water_density = 1000.0  # 1000 Kg/m^3

        # For the water-tank sphere (assumed index 0), update the volume if water_tank_level is provided.
        if charged_water_mass is not None:
            # Calculate additional volume from water tank level (volume in m³ = mass (kg) / density)
            extra_volume = charged_water_mass / water_density
            effective_volume = self.volume_list[idx] + extra_volume
            effective_radius = ((3 * effective_volume) / (4 * pi)) ** (1 / 3)
        else:
            effective_volume = self.volume_list[idx]
            effective_radius = self.radius_list[idx]

        F_full = water_density * effective_volume * (-self.cfg.gravity)

        # Compute the submerged fraction based on effective radius.
        d = positions_z  # shape (num_envs,)
        fraction = torch.where(
            d <= -effective_radius,
            torch.ones_like(d),
            torch.where(
                d >= effective_radius, torch.zeros_like(d), 1.0 - (0.5 + 0.75 * (d / effective_radius) - 0.25 * (d**3) / (effective_radius**3))
            ),
        )
        Fz = F_full * fraction
        self.archimedes_force_global[:, 2] = Fz
        # Torque computation can be extended as needed.
        self.archimedes_torque_global = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        return self.archimedes_force_global, self.archimedes_torque_global

    def compute_archimedes_metacentric_local(self, position, quaternions, positions_z, idx, charged_water_mass=None):

        roll, pitch, yaw = euler_xyz_from_quat(quaternions)
        euler = torch.stack((roll, pitch, yaw), dim=1)

        self.compute_archimedes_metacentric_global(euler, positions_z, idx, charged_water_mass)
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

        # buoyancy = 0 when the robot is overwater
        mask = (positions_z < 0.0).float().unsqueeze(1)
        self.archimedes_force_local = self.archimedes_force_local * mask
        self.archimedes_torque_local = self.archimedes_torque_local * mask

        return torch.hstack(
            [
                self.archimedes_force_local,
                self.archimedes_torque_local,
            ]
        )

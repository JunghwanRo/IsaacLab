# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import Articulation
from isaaclab_tasks.direct.seal.physics.buoyancy import Buoyancy
from isaaclab_tasks.direct.seal.physics.hydrodynamics import Hydrodynamics
from isaaclab_tasks.direct.seal.physics.aerodynamics import Aerodynamics
from isaaclab_tasks.direct.seal.physics.lift import Lift
from isaaclab_tasks.direct.seal.physics.addedmass import AddedMass
from isaaclab.utils import math as math_utils

from isaaclab_tasks.direct.seal.robots_cfg.squidbot_cfg import SquidbotRobotCfg

from .robot_core import RobotCore


class SquidbotRobot(RobotCore):

    def __init__(
        self,
        robot_cfg: SquidbotRobotCfg,
        robot_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        sim_dt: float = 0.02,
    ):
        super().__init__(robot_uid=robot_uid, num_envs=num_envs, device=device)
        self._robot_cfg = robot_cfg

        # Buoyancy, Hydrodynamics, Aerodynamics and Thruster Dynamics
        self._buoyancy = Buoyancy(num_envs, device, self._robot_cfg.buoyancy_cfg)
        self._hydrodynamics = Hydrodynamics(num_envs, device, self._robot_cfg.hydrodynamics_cfg)
        self._aerodynamics = Aerodynamics(num_envs, device, self._robot_cfg.aerodynamics_cfg)
        self._lift = Lift(num_envs, device, self._robot_cfg.lift_cfg)
        self._addedmass = AddedMass(num_envs, device, sim_dt, self._robot_cfg.addedmass_cfg)

        # Buffers
        self.initialize_buffers()

    def initialize_buffers(self, env_ids=None):
        super().initialize_buffers(env_ids)
        self._buoyancy_force = torch.zeros(
            (self._num_envs, len(self._robot_cfg.buoyancy_cfg.buoyancy_spheres), 6), device=self._device, dtype=torch.float32
        )
        self._hydrodynamic_force = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)
        self._aerodynamic_force = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)
        self._drag_force = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)
        self._lift_force = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)
        self._addedmass_force = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)
        self._combined_force_root = torch.zeros((self._num_envs, 1, 6), device=self._device, dtype=torch.float32)

    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._root_idx, _ = self._robot.find_bodies([self._robot_cfg.root_id_name])
        # cob no more used, switched to distrituted buoyancy spheres
        # self._cob_idx, _ = self._robot.find_bodies(["cob"])
        self._ac_idx, _ = self._robot.find_bodies(["ac"])

        # Save the buoyancy sphere names along with their IDs.
        self._buoyancy_sphere_ids = []
        self._buoyancy_sphere_names = []
        for sphere_name, _ in self._robot_cfg.buoyancy_cfg.buoyancy_spheres:
            sphere_id = self._robot.find_bodies(sphere_name)[0][0]
            self._buoyancy_sphere_ids.append(sphere_id)
            self._buoyancy_sphere_names.append(sphere_name)

    def compute_physics(self, water_tank_level=None):
        # For each buoyancy sphere, compute the corresponding forces.
        for idx, sphere_id in enumerate(self._buoyancy_sphere_ids):
            # If this is the water tank sphere, pass the water_tank_level.
            # TODO: now "buoyancy_00" is hard coded, need to add it in the cfg file.
            if self._buoyancy_sphere_names[idx] == self._robot_cfg.water_tank_b_name and water_tank_level is not None:
                self._buoyancy_force[:, idx, :] = self._buoyancy.compute_archimedes_metacentric_local(
                    self._robot.data.body_link_pos_w[:, sphere_id],
                    self._robot.data.body_link_quat_w[:, sphere_id],
                    self._robot.data.body_link_pos_w[:, sphere_id, 2],
                    idx,
                    charged_water_mass=water_tank_level * (self._robot_cfg.water_tank_mass / 100.0),
                )
            else:
                self._buoyancy_force[:, idx, :] = self._buoyancy.compute_archimedes_metacentric_local(
                    self._robot.data.body_link_pos_w[:, sphere_id],
                    self._robot.data.body_link_quat_w[:, sphere_id],
                    self._robot.data.body_link_pos_w[:, sphere_id, 2],
                    idx,
                )

        # Compute hydrodynamics
        self._hydrodynamic_force[:, 0, :] = self._hydrodynamics.ComputeHydrodynamicsEffects(self.root_quat_w, self.root_vel_w, self.root_pos_w[:, 2])

        # Compute aerodynamics
        self._aerodynamic_force[:, 0, :] = self._aerodynamics.ComputeAerodynamicsEffects(self.root_quat_w, self.root_vel_w, self.root_pos_w[:, 2])

        # Compute lift forces (lift + lift induced drag)
        self._lift_force[:, 0, :] = self._lift.compute_lift_forces(self.root_quat_w, self.root_vel_w, self.root_pos_w[:, 2])

        # Compute added mass converted forces
        self._addedmass_force[:, 0, :] = self._addedmass.compute_added_mass_force(self.root_vel_b, self.root_pos_w[:, 2])
        # print(f"self.root_vel_b: {self.root_vel_b}")
        # print(f"addedmass_force: {self._addedmass_force}")

    def apply_physics(self, articulations: Articulation):
        # If the function is called with empty forces and torques, then this function disables the application of external wrench to the simulation. So we must check if it is not empty.
        robot_z_positions = self._robot.data.body_com_state_w[:, self._root_idx, 2]
        # Check if the robot is underwater. if it is overwater, no buoyancy force is applied to the environment.
        # Match robot_z_position dimension before using it in torch.where with self._buoyancy_force
        robot_z_positions = robot_z_positions.unsqueeze(1).repeat(1, 1, 6)
        # print robot_z_positions
        # print(f"robot_z_positions: {robot_z_positions}")
        # print _buoyancy_force
        # print(f"_buoyancy_force: {self._buoyancy_force}")
        if torch.any(self._buoyancy_force[..., :6] != 0.0):
            # print applied buoyancy force
            # print(f"applied buoyancy force: {self._buoyancy_force}")
            # Apply buoyancy force, and zero torque
            articulations.set_external_force_and_torque(
                self._buoyancy_force[..., :3], self._buoyancy_force[..., 3:], body_ids=self._buoyancy_sphere_ids
            )
        if torch.any(self._lift_force[..., :6] != 0.0):
            articulations.set_external_force_and_torque(self._lift_force[..., :3], self._lift_force[..., 3:], body_ids=self._ac_idx)

        # self._combined_force_root[..., :6] = self._drag_force[..., :6]
        # self._combined_force_root[..., :6] = self._drag_force[..., :6] + self._lift_force[..., :6]
        # TODO: below is correct.
        self._combined_force_root[..., :6] = self._aerodynamic_force[..., :6] + self._hydrodynamic_force[..., :6] + self._addedmass_force[..., :6]
        # print(f"self._combined_force_root: {self._combined_force_root}")
        if torch.any(self._combined_force_root != 0.0):
            # print applied hydrodynamic force
            # print(f"applied hydrodynamic force: {self._hydrodynamic_force}")
            # Apply hydrodynamic forces and torques
            # articulations.set_external_force_and_torque(self._drag_force[..., :3], self._drag_force[..., 3:], body_ids=self._root_idx)
            articulations.set_external_force_and_torque(
                self._combined_force_root[..., :3], self._combined_force_root[..., 3:], body_ids=self._root_idx
            )

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation root's center of mass frame.
        """
        return self._robot.data.body_link_state_w[:, self._root_idx].squeeze(1)

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the articulation root.

        body_com_pos_w: Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).
        """
        return self._robot.data.body_link_pos_w[:, self._root_idx].squeeze(1)

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the articulation root.

        body_com_quat_w: Orientation (w, x, y, z) of the prinicple axies of inertia of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).
        """
        return self._robot.data.body_link_quat_w[:, self._root_idx].squeeze(1)

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of
        mass frame.

        body_com_vel_w: Velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 6).
        """
        return self._robot.data.body_link_vel_w[:, self._root_idx].squeeze(1)

    @property
    def root_vel_b(self) -> torch.Tensor:
        """Root velocity in body frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of
        mass frame. We make it by merging self._rotot.data.root_lin_vel_b and self._robot.data.root_ang_vel_b.

        property root_lin_vel_b: torch.Tensor
        Root linear velocity in base frame. Shape is (num_instances, 3).
        This quantity is the linear velocity of the articulation root’s center of mass frame relative to the world with respect to the articulation root’s actor frame.

        property root_ang_vel_b: torch.Tensor
        Root angular velocity in base world frame. Shape is (num_instances, 3).
        This quantity is the angular velocity of the articulation root’s center of mass frame relative to the world with respect to the articulation root’s actor frame.
        """
        return torch.hstack([self._robot.data.root_lin_vel_b, self._robot.data.root_ang_vel_b])

    # Public properties to access the protected force data
    @property
    def buoyancy_force(self) -> torch.Tensor:
        return self._buoyancy_force

    @property
    def hydrodynamic_force(self) -> torch.Tensor:
        return self._hydrodynamic_force

    @property
    def aerodynamic_force(self) -> torch.Tensor:
        return self._aerodynamic_force

    @property
    def lift_force(self) -> torch.Tensor:
        return self._lift_force

    @property
    def addedmass_force(self) -> torch.Tensor:
        return self._addedmass_force

    @property
    def combined_force_root(self) -> torch.Tensor:
        return self._combined_force_root

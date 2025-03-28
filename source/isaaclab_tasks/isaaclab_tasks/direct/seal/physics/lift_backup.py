# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from math import pi
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_rotate_inverse

"""
Following Fossen's Equation,
Fossen, T. I. (1991). Nonlinear modeling and control of Underwater Vehicles. Doctoral thesis, Department of Engineering Cybernetics, Norwegian Institute of Technology (NTH), June 1991.
"""


@configclass
class LiftCfg:
    """
    Configuration for a simplified lift + induced drag model.

    This is a stripped-down approach similar to PyFly,
    but specialized for a rigid body with no control surfaces.
    """

    # Air density (kg/m^3) for aerodynamic environment
    air_density: float = 1.225
    # Reference wing area (m^2)
    reference_area: float = 0.5
    # Lift coefficient at alpha=0
    CL0: float = 0.2
    # Lift slope per rad of alpha (CL_alpha)
    CL_alpha: float = 5.7
    # Induced/Parasitic drag coefficient at alpha=0
    CD0: float = 0.02
    # Induced drag factor (K) if you want something like CD = CD0 + K * (CL^2)
    K_induced: float = 0.05

    # Stall angle in radians (e.g., 15 deg ~ 0.26 rad)
    alpha_stall: float = 0.3
    # Maximum angle of attack for which any lift is produced (e.g., 90 deg ~ 1.57 rad)
    alpha_max: float = 0.6

    # Switch or scaling factor if you only apply lift above water, etc.
    apply_if_z_above: float = 0.0


class Lift:
    def __init__(self, num_envs: int, device: str, cfg: LiftCfg):
        """
        A simplified lift + induced drag class for rigid bodies.

        Args:
            num_envs: Number of parallel environments
            device: Torch device, e.g. 'cuda' or 'cpu'
            cfg: LiftCfg with aerodynamic parameters
        """
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg

        # Output buffer for storing [Fx, Fy, Fz, Mx, My, Mz] in local body frame
        self.lift_forces = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)

    def compute_lift_forces(self, quaternions: torch.Tensor, velocities: torch.Tensor, positions_z: torch.Tensor):
        """
        Compute and return local-body lift + induced drag forces on each environment.

        Args:
            quaternions: (num_envs, 4) root-body orientation in world frame (w, x, y, z).
            velocities: (num_envs, 6) linear and angular velocity in world frame. We only need linear part for lift.
            positions_z: (num_envs,) The z-coordinate of the robot root link in world frame (to check water/air).
                        If you only want lift above a certain z, you can check that here.

        Returns:
            self.lift_forces: (num_envs, 6) local-body [Fx, Fy, Fz, Mx, My, Mz].
                              For this simple approach, we compute only F-lift & F-drag, torque=0.
        """
        # 1) Convert velocity to local body frame
        #    We'll use only the linear velocity for alpha calculation.
        local_lin_vel = quat_rotate_inverse(quaternions, velocities[:, :3])
        # Debug:
        # print(f"local_lin_vel: {local_lin_vel}")

        # 2) Compute speed Va and angle of attack alpha
        #    For our simple 2D approach in the x-z plane:
        #    local_lin_vel = [u, v, w] where u and w are chosen according to the simulation's convention.
        u = local_lin_vel[:, 2]  # forward component
        w = local_lin_vel[:, 0]  # vertical component
        Va = torch.sqrt(u * u + w * w + 1e-6)

        # Angle of attack: defined as alpha = atan2(w, u)
        alpha = torch.atan2(w, u)

        # 3) Compute lift coefficient with stall behavior:
        #    For |alpha| below alpha_stall use the linear model:
        #         CL = CL0 + CL_alpha * alpha
        #    For |alpha| above alpha_stall, let CL decay linearly to zero at alpha_max.
        alpha_stall = self.cfg.alpha_stall  # e.g. 0.26 rad (15 deg)
        alpha_max = self.cfg.alpha_max  # e.g. 1.57 rad (90 deg)
        CL_linear = self.cfg.CL0 + self.cfg.CL_alpha * alpha
        # Maximum lift coefficient achieved at stall
        CL_max = self.cfg.CL0 + self.cfg.CL_alpha * alpha_stall

        # Compute lift coefficient for post-stall regime:
        #   For |alpha| between alpha_stall and alpha_max,
        #   assume a linear decay from CL_max to zero.
        #   The decay factor is computed as:
        #       CL_decay = CL_max * (1 - ((|alpha| - alpha_stall) / (alpha_max - alpha_stall)))
        CL_decay = CL_max * (1 - ((torch.abs(alpha) - alpha_stall) / (alpha_max - alpha_stall)))
        # Ensure that CL_decay does not go negative:
        CL_decay = torch.clamp(CL_decay, min=0.0)
        # Choose the appropriate CL based on alpha:
        CL = torch.where(torch.abs(alpha) <= alpha_stall, CL_linear, CL_decay * torch.sign(alpha))

        # 4) Compute dynamic pressure: q_dyn = 0.5 * rho * Va^2
        q_dyn = 0.5 * self.cfg.air_density * Va * Va

        # 5) Compute lift force magnitude (L = q_dyn * reference_area * CL)
        L = q_dyn * self.cfg.reference_area * CL

        # 6) Compute induced (or total) drag coefficient.
        #    Here we ignore the parasitic drag (CD0) for simplicity.
        CD = self.cfg.K_induced * (CL * CL)
        D = q_dyn * self.cfg.reference_area * CD

        # 7) Determine the force directions in the local x-z plane:
        #    With our coordinate assignment (u forward, w vertical):
        small_eps = 1e-6
        # Drag is aligned opposite to the velocity vector in the x-z plane.
        drag_z = -(u / (Va + small_eps))  # along local z
        drag_x = -(w / (Va + small_eps))  # along local x
        # Lift is perpendicular to the velocity in the x-z plane:
        lift_z = -drag_x
        lift_x = drag_z

        # 8) Compute the force components:
        Fx = D * drag_x + L * lift_x
        Fz = D * drag_z + L * lift_z

        # Apply the force only if the position's z-coordinate is above the threshold.
        apply_mask = positions_z > self.cfg.apply_if_z_above  # shape: (num_envs,)
        apply_mask_f = apply_mask.float()  # 1 if True, else 0
        Fx = Fx * apply_mask_f
        Fz = Fz * apply_mask_f

        # 9) Store the computed forces (no moments in this simplified model)
        self.lift_forces[:, 0] = Fx  # local x-force
        self.lift_forces[:, 1] = 0.0  # local y-force
        self.lift_forces[:, 2] = Fz  # local z-force
        self.lift_forces[:, 3] = 0.0  # roll moment
        self.lift_forces[:, 4] = 0.0  # pitch moment
        self.lift_forces[:, 5] = 0.0  # yaw moment

        # print(f"self.lift_forces: {self.lift_forces}")

        return self.lift_forces

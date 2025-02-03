# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_rotate_inverse

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

    # If you want to limit the maximum absolute alpha
    alpha_max: float = 1.5  # [rad], i.e ~ 86 deg

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
        #    We'll use only local_lin_vel for alpha calculation.
        local_lin_vel = quat_rotate_inverse(quaternions, velocities[:, :3])

        # 2) Compute speed Va and angle of attack alpha
        #    alpha = atan2(w, u), ignoring v for a simple 2D approach
        #    local_lin_vel = [u, v, w]
        #    Note that u and w was defined carefully in the isaac sim modeling.
        u = local_lin_vel[:, 2]
        w = local_lin_vel[:, 0]
        Va = torch.sqrt(u * u + w * w + 1e-6)

        alpha = torch.atan2(w, u)
        # clamp alpha to alpha_max
        alpha = torch.clamp(alpha, -self.cfg.alpha_max, self.cfg.alpha_max)

        # 3) Compute lift coefficient
        #    CL = CL0 + CL_alpha * alpha
        CL = self.cfg.CL0 + (self.cfg.CL_alpha * alpha)

        # 4) Compute dynamic pressure and partial factors
        #    q_dyn = 0.5 * rho * Va^2
        q_dyn = 0.5 * self.cfg.air_density * Va * Va

        # 5) Lift force magnitude (in local frame, ignoring sign)
        #    L = q_dyn * reference_area * CL
        L = q_dyn * self.cfg.reference_area * CL

        # 6) Induced (or total) drag coefficient
        #    For simplicity: CD = CD0 + K*(CL^2)
        #    Then D = q_dyn * reference_area * CD
        # CD = self.cfg.CD0 + self.cfg.K_induced * (CL * CL)
        # For now, we ignore the parasitic drag CD0, since it is handled in the aerodynamics model.
        CD = self.cfg.K_induced * (CL * CL)
        D = q_dyn * self.cfg.reference_area * CD

        # 7) The direction in the local x-z plane
        #    Typically, lift is perpendicular to velocity and drag is aligned (opposite sign).
        #    local_xz velocity vector: [u, w]
        #    drag direction: negative the velocity
        #    lift direction: negative 90 deg from velocity, in the x-z plane
        #    Let's compute unit vectors in x-z:
        #    speed = Va
        #    drag_unit = -[u, 0, w]/Va
        #    lift_unit = cross( [0,1,0], drag_unit ) or rotate by +90 deg in x-z plane
        #    We'll do a simpler approach: if velocity is in +u direction, alpha>0 => lift is negative local z
        #                                 we can do angle-based approach:

        # drag_x = - (u / Va)
        # drag_z = - (w / Va)
        # lift_x = -drag_z
        # lift_z =  drag_x
        # sign if alpha>0 => lift is negative z, if alpha<0 => lift is positive z ?? We'll keep consistent with angle of attack definition

        small_eps = 1e-6
        # Because now 'u' is in z, 'w' is in x, the “drag” direction is negative the velocity in (z, x).
        drag_z = -(u / (Va + small_eps))  # if forward is z
        drag_x = -(w / (Va + small_eps))  # if normal is x

        # The lift is perpendicular to velocity in the x–z plane:
        lift_z = -drag_x
        lift_x = drag_z

        # 8) Multiply by magnitudes L and D
        #    local lift = L * [lift_x, 0, lift_z]
        #    local drag = D * [drag_x, 0, drag_z]
        #    Then sum them
        #    We'll assume no side force, so y=0
        Fx = D * drag_x + L * lift_x
        Fz = D * drag_z + L * lift_z
        # store in local-body array
        # also check if we want to apply if z is above water_surf
        # we can do a mask
        apply_mask = positions_z > self.cfg.apply_if_z_above  # shape: (num_envs,)
        apply_mask_f = apply_mask.float()  # 1 if True, else 0

        # If you only want the object to get lift if it's above water (or below?), you can invert or adjust.
        # For example, if we only want lift above water, you'd do:
        # apply_mask = positions_z > 0.0
        # or some threshold.

        # Combine force with mask
        Fx = Fx * apply_mask_f
        Fz = Fz * apply_mask_f

        # 9) For a simple approach, no torque in this model:
        self.lift_forces[:, 0] = Fx  # local x
        self.lift_forces[:, 1] = 0.0  # local y
        self.lift_forces[:, 2] = Fz  # local z
        self.lift_forces[:, 3] = 0.0  # roll torque
        self.lift_forces[:, 4] = 0.0  # pitch torque
        self.lift_forces[:, 5] = 0.0  # yaw torque

        return self.lift_forces

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
    Configuration for a simplified aerodynamic force and moment model with a simple stall.

    Local coordinate system:
      - x (index 0): upward
      - y (index 1): lateral (to the right)
      - z (index 2): forward
    """

    air_density: float = 1.225
    reference_area: float = 0.08  # Wing (or reference) area, in m².

    # Lift coefficient parameters (for force in the vertical plane)
    CL0: float = 0.0
    CL_alpha: float = 10.0
    alpha_stall: float = 0.3  # Angle (radians) at which stall begins.
    alpha_max: float = 0.6  # Maximum effective angle (radians).

    # Induced drag: simple quadratic model.
    K_induced: float = 0.05

    # Side force coefficient (assumed linear with sideslip beta)
    CY_beta: float = -0.5

    # Geometric parameters (for moment computations)
    chord: float = 0.08  # Reference chord length (used for pitch moment).
    wing_span: float = 0.5  # Reference wing span (used for roll and yaw moments).

    # Moment coefficients (defined in a standard aircraft frame):
    # Pitching moment (about Y_std)
    CM0: float = 0.0
    CM_alpha: float = -1.0
    # Rolling moment (about X_std)
    Cl0: float = 0.0
    Cl_beta: float = -0.1
    # Yawing moment (about Z_std)
    Cn0: float = 0.0
    Cn_beta: float = 0.1

    # Additional stall parameters for moments:
    # For pitch moment (using alpha):
    CM_alpha_stall: float = 0.3  # Stall onset for pitch moment (radians)
    CM_alpha_max: float = 0.6  # Maximum effective angle for pitch moment (radians)

    # For roll and yaw moments (using beta):
    Cl_beta_stall: float = 0.1  # Stall onset for roll moment (radians)
    Cl_beta_max: float = 0.2  # Maximum effective angle for roll moment (radians)
    Cn_beta_stall: float = 0.1  # Stall onset for yaw moment (radians)
    Cn_beta_max: float = 0.2  # Maximum effective angle for yaw moment (radians)

    # Apply aerodynamic forces only if the world z-coordinate exceeds this threshold.
    apply_if_z_above: float = 0.0


class Lift:
    def __init__(self, num_envs: int, device: str, cfg: LiftCfg):
        """
        Initializes the aerodynamic force and moment model.

        Args:
            num_envs: Number of environments.
            device: Torch device (e.g. 'cpu' or 'cuda').
            cfg: Lift configuration parameters.
        """
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg

        # Output buffer for storing [Fx, Fy, Fz, Mx, My, Mz] in the local body frame.
        self.lift_forces = torch.zeros((num_envs, 6), device=device, dtype=torch.float32)

    def compute_lift_forces(self, quaternions: torch.Tensor, velocities: torch.Tensor, positions_z: torch.Tensor):
        """
        Computes the aerodynamic force and moment vector in the local body frame.

        Args:
            quaternions: (num_envs, 4) orientation (w, x, y, z) in world frame.
            velocities: (num_envs, 6) linear and angular velocities in world frame.
            positions_z: (num_envs,) world-frame z-coordinate (used for conditional application).

        Returns:
            lift_forces: (num_envs, 6) [Fx, Fy, Fz, Mx, My, Mz] in the local body frame.
        """
        eps = 1e-6

        # 1) Convert linear velocity from world to local body frame.
        local_lin_vel = quat_rotate_inverse(quaternions, velocities[:, :3])
        # Convention: index 0 = upward, index 1 = lateral, index 2 = forward.
        w = local_lin_vel[:, 0]
        v = local_lin_vel[:, 1]
        u = local_lin_vel[:, 2]

        # 2) Compute airspeed and aerodynamic angles.
        Va = torch.sqrt(u * u + v * v + w * w + eps)
        alpha = torch.atan2(w, u)  # Angle of attack.
        beta = torch.asin(v / (Va + eps))  # Sideslip angle.

        # 3) Compute dynamic pressure.
        q_dyn = 0.5 * self.cfg.air_density * Va * Va

        # 4) Compute lift coefficient with simple stall.
        CL_linear = self.cfg.CL0 + self.cfg.CL_alpha * alpha
        CL_max = self.cfg.CL0 + self.cfg.CL_alpha * self.cfg.alpha_stall
        CL_decay = CL_max * (1 - ((torch.abs(alpha) - self.cfg.alpha_stall) / (self.cfg.alpha_max - self.cfg.alpha_stall)))
        CL_decay = torch.clamp(CL_decay, min=0.0)
        CL = torch.where(torch.abs(alpha) <= self.cfg.alpha_stall, CL_linear, CL_decay * torch.sign(alpha))

        # 5) Compute drag and side force coefficients.
        CD = self.cfg.K_induced * (CL * CL)
        CY = self.cfg.CY_beta * beta

        # 6) Compute force magnitudes.
        L = q_dyn * self.cfg.reference_area * CL  # Lift.
        D = q_dyn * self.cfg.reference_area * CD  # Drag.
        Y = q_dyn * self.cfg.reference_area * CY  # Side force.

        # 7) Decompose lift and drag in the x-z plane.
        denom = torch.sqrt(u * u + w * w + eps)
        drag_z = -(u / denom)
        drag_x = -(w / denom)
        # Lift is perpendicular to drag.
        lift_x = -drag_z
        lift_z = drag_x

        # 8) Force components in the x-z plane.
        Fx_plane = D * drag_x + L * lift_x  # x (upward) component.
        Fz_plane = D * drag_z + L * lift_z  # z (forward) component.
        # Lateral force (y) comes directly from the side force.
        Fy = Y

        # 9) Assemble the force vector.
        Fx = Fx_plane
        Fz = Fz_plane

        # 10) Compute aerodynamic moments in a standard aircraft frame.
        # --- Pitch Moment (using alpha with a stall-like nonlinearity) ---
        CM_linear = self.cfg.CM0 + self.cfg.CM_alpha * alpha
        CM_max = self.cfg.CM0 + self.cfg.CM_alpha * self.cfg.CM_alpha_stall
        CM_decay = CM_max * (1 - ((torch.abs(alpha) - self.cfg.CM_alpha_stall) / (self.cfg.CM_alpha_max - self.cfg.CM_alpha_stall)))
        CM_decay = torch.clamp(CM_decay, min=0.0)
        CM = torch.where(torch.abs(alpha) <= self.cfg.CM_alpha_stall, CM_linear, CM_decay * torch.sign(alpha))
        M_pitch_std = q_dyn * self.cfg.reference_area * self.cfg.chord * CM

        # --- Roll Moment (using beta with a stall-like nonlinearity) ---
        Cl_linear = self.cfg.Cl0 + self.cfg.Cl_beta * beta
        Cl_max = self.cfg.Cl0 + self.cfg.Cl_beta * self.cfg.Cl_beta_stall
        Cl_decay = Cl_max * (1 - ((torch.abs(beta) - self.cfg.Cl_beta_stall) / (self.cfg.Cl_beta_max - self.cfg.Cl_beta_stall)))
        Cl_decay = torch.clamp(Cl_decay, min=0.0)
        Cl = torch.where(torch.abs(beta) <= self.cfg.Cl_beta_stall, Cl_linear, Cl_decay * torch.sign(beta))
        M_roll_std = q_dyn * self.cfg.reference_area * self.cfg.wing_span * Cl

        # --- Yaw Moment (using beta with a stall-like nonlinearity) ---
        Cn_linear = self.cfg.Cn0 + self.cfg.Cn_beta * beta
        Cn_max = self.cfg.Cn0 + self.cfg.Cn_beta * self.cfg.Cn_beta_stall
        Cn_decay = Cn_max * (1 - ((torch.abs(beta) - self.cfg.Cn_beta_stall) / (self.cfg.Cn_beta_max - self.cfg.Cn_beta_stall)))
        Cn_decay = torch.clamp(Cn_decay, min=0.0)
        Cn = torch.where(torch.abs(beta) <= self.cfg.Cn_beta_stall, Cn_linear, Cn_decay * torch.sign(beta))
        M_yaw_std = q_dyn * self.cfg.reference_area * self.cfg.wing_span * Cn

        Mx = -M_yaw_std  # Moment about x (local upward axis) corresponds to –yaw moment.
        My = M_pitch_std  # Moment about y (local lateral axis) corresponds to pitch moment.
        Mz = M_roll_std  # Moment about z (local forward axis) corresponds to roll moment.

        # 11) Apply the condition: only apply forces/moments if world z-coordinate > threshold.
        mask = (positions_z > self.cfg.apply_if_z_above).float()
        Fx = Fx * mask
        Fy = Fy * mask
        Fz = Fz * mask
        Mx = Mx * mask
        My = My * mask
        Mz = Mz * mask

        # 12) Assemble the final 6D wrench vector in the local frame.
        self.lift_forces[:, 0] = Fx  # upward force.
        self.lift_forces[:, 1] = Fy  # lateral force.
        self.lift_forces[:, 2] = Fz  # forward force.
        self.lift_forces[:, 3] = Mx  # moment about x (yaw moment).
        self.lift_forces[:, 4] = My  # moment about y (pitch moment).
        self.lift_forces[:, 5] = Mz  # moment about z (roll moment).

        return self.lift_forces

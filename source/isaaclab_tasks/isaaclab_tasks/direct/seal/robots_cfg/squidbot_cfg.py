# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.seal import SEAL_CFG

from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.direct.seal.physics.buoyancy import BuoyancyCfg
from isaaclab_tasks.direct.seal.physics.hydrodynamics import HydrodynamicsCfg
from isaaclab_tasks.direct.seal.physics.aerodynamics import AerodynamicsCfg
from isaaclab_tasks.direct.seal.physics.lift import LiftCfg
from isaaclab_tasks.direct.seal.physics.addedmass import AddedMassCfg
from isaaclab.utils import configclass

from .robot_core_cfg import RobotCoreCfg


@configclass
class SquidbotRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_cfg: ArticulationCfg = SEAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    root_id_name = "base_link"

    # Buoyancy
    buoyancy_cfg: BuoyancyCfg = BuoyancyCfg()
    buoyancy_cfg.mass = 0.5730299949645996  # Kg
    buoyancy_cfg.total_water_mass = 0.5730299949645996  # Kg
    buoyancy_cfg.buoyancy_spheres = [("buoyancy_00", 1.0), ("buoyancy_01", 1.0), ("buoyancy_02", 0.5)]

    # Hydrodynamics
    hydrodynamics_cfg: HydrodynamicsCfg = HydrodynamicsCfg()
    # Damping
    hydrodynamics_cfg.linear_damping = [3.0, 3.0, 1.0, 0.05, 0.05, 0.02]
    # linear Nominal [16.44998712, 15.79776044, 100, 13, 13, 6]
    # linear SID [0.0, 99.99, 99.99, 13.0, 13.0, 0.82985084]
    hydrodynamics_cfg.quadratic_damping = [0.2, 0.2, 0.05, 0.001, 0.001, 0.001]
    # quadratic Nominal [2.942, 2.7617212, 10, 5, 5, 5]
    # quadratic SID [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    # Damping randomization
    hydrodynamics_cfg.use_drag_randomization = False
    hydrodynamics_cfg.linear_damping_rand = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    hydrodynamics_cfg.quadratic_damping_rand = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Aerodynamics
    aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
    # Damping
    aerodynamics_cfg.linear_damping = [0.05, 0.05, 0.02, 0.001, 0.001, 0.001]
    aerodynamics_cfg.quadratic_damping = [0.003, 0.003, 0.001, 0.0001, 0.0001, 0.0001]
    # Damping randomization
    aerodynamics_cfg.use_drag_randomization = False
    aerodynamics_cfg.linear_damping_rand = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    aerodynamics_cfg.quadratic_damping_rand = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Add a new lift config
    lift_cfg: LiftCfg = LiftCfg()
    lift_cfg.air_density = 1.225
    lift_cfg.reference_area = 0.08  # Wing (or reference) area, in m².
    # Lift coefficient parameters (for force in the vertical plane)
    lift_cfg.CL0 = 0.0
    lift_cfg.CL_alpha = 10.0
    lift_cfg.alpha_stall = 0.35  # Angle (radians) at which stall begins.
    lift_cfg.alpha_max = 1.57  # Maximum effective angle (radians).
    # Induced drag: simple quadratic model.
    lift_cfg.K_induced = 0.05
    # Side force coefficient (assumed linear with sideslip beta)
    lift_cfg.CY_beta = -0.5
    # Geometric parameters (for moment computations)
    lift_cfg.chord = 0.08  # Reference chord length (used for pitch moment).
    lift_cfg.wing_span = 0.5  # Reference wing span (used for roll and yaw moments).
    # Moment coefficients (defined in a standard aircraft frame):
    # Pitching moment (about Y_std) (- indicates pitch stability)
    lift_cfg.CM0 = 0.0
    lift_cfg.CM_alpha = -0.1
    # Rolling moment (about X_std) (- indicates roll stability)
    lift_cfg.Cl0 = 0.0
    lift_cfg.Cl_beta = -0.1  # Default -0.1
    # Yawing moment (about Z_std) (+ indicates yaw stability, align with the wind)
    lift_cfg.Cn0 = 0.0
    lift_cfg.Cn_beta = 0.1

    # Additional stall parameters for moments:
    # For pitch moment (using alpha):
    CM_alpha_stall = 0.35  # Stall onset for pitch moment (radians)
    CM_alpha_max = 1.57  # Maximum effective angle for pitch moment (radians)
    # For roll and yaw moments (using beta):
    Cl_beta_stall = 0.35  # Stall onset for roll moment (radians)
    Cl_beta_max = 1.57  # Maximum effective angle for roll moment (radians)
    Cn_beta_stall = 0.35  # Stall onset for yaw moment (radians)
    Cn_beta_max = 1.57  # Maximum effective angle for yaw moment (radians)
    # Apply aerodynamic forces only if the world z-coordinate exceeds this threshold.
    lift_cfg.apply_if_z_above = 0.0

    # Added-Mass configuration.
    addedmass_cfg: AddedMassCfg = AddedMassCfg()
    # Example coefficients: choose values according to your vehicle's hydrodynamic properties.
    # addedmass_cfg.added_mass = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    addedmass_cfg.added_mass = [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]
    addedmass_cfg.alpha = 0.5
    # TODO: added mass dt should match with core dt. later it should be not here.
    addedmass_cfg.sim_dt = 0.02

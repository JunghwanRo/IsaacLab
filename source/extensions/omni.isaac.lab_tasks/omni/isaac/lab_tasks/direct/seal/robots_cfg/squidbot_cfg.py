# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab_assets.seal import SEAL_CFG

from omni.isaac.lab.assets import ArticulationCfg

from omni.isaac.lab_tasks.direct.seal.physics.buoyancy import BuoyancyCfg
from omni.isaac.lab_tasks.direct.seal.physics.hydrodynamics import HydrodynamicsCfg
from omni.isaac.lab_tasks.direct.seal.physics.aerodynamics import AerodynamicsCfg
from omni.isaac.lab_tasks.direct.seal.physics.lift import LiftCfg
from omni.isaac.lab.utils import configclass

from .robot_core_cfg import RobotCoreCfg


@configclass
class SquidbotRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_cfg: ArticulationCfg = SEAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    root_id_name = "base_link"

    # Buoyancy
    buoyancy_cfg: BuoyancyCfg = BuoyancyCfg()
    buoyancy_cfg.mass = 0.5119999647140503  # Kg

    # Hydrodynamics
    hydrodynamics_cfg: HydrodynamicsCfg = HydrodynamicsCfg()
    # Damping
    hydrodynamics_cfg.linear_damping = [0.5, 0.5, 0.5, 0.03, 0.03, 0.03]
    # linear Nominal [16.44998712, 15.79776044, 100, 13, 13, 6]
    # linear SID [0.0, 99.99, 99.99, 13.0, 13.0, 0.82985084]
    hydrodynamics_cfg.quadratic_damping = [0.5, 0.5, 0.5, 0.003, 0.003, 0.003]
    # quadratic Nominal [2.942, 2.7617212, 10, 5, 5, 5]
    # quadratic SID [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    # Damping randomization
    hydrodynamics_cfg.use_drag_randomization = False
    hydrodynamics_cfg.linear_damping_rand = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    hydrodynamics_cfg.quadratic_damping_rand = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Aerodynamics
    aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
    # Damping
    aerodynamics_cfg.linear_damping = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    aerodynamics_cfg.quadratic_damping = [0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001]
    # Damping randomization
    aerodynamics_cfg.use_drag_randomization = False
    aerodynamics_cfg.linear_damping_rand = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    aerodynamics_cfg.quadratic_damping_rand = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Add a new lift config
    lift_cfg: LiftCfg = LiftCfg()
    # Example parameter values:
    lift_cfg.air_density = 1.225
    lift_cfg.reference_area = 0.0  # 0.1
    lift_cfg.CL0 = 0.0
    lift_cfg.CL_alpha = 0.0  # 5.7
    lift_cfg.CD0 = 0.0
    lift_cfg.K_induced = 0.0  # 0.05
    lift_cfg.alpha_max = 0.0  # 1.5
    # If you only want lift above water (z>0), set apply_if_z_below to -9999
    # or if you only want lift below z=0, set 0 or something. Example:
    lift_cfg.apply_if_z_above = 0.0  # effectively always apply

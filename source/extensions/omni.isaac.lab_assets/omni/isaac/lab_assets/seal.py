# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the fourbar drone."""

from __future__ import annotations

import os
import pathlib
import yaml

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Determine the IsaacLab directory
ISAACLAB_DIR = pathlib.Path(__file__).parents[6]


# Function to read YAML configuration
def read_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Read the configuration
config_path = os.path.join(ISAACLAB_DIR, "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/seal/seal_cfg.yaml")
config = read_config(config_path)

# Extract the model and usd_path from the configuration
seal_model = config["seal_model"]
seal_usd_path = os.path.join(ISAACLAB_DIR, config["seal_usd_path"])

##
# Configuration
##

SEAL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=seal_usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,  # if true, the root link is fixed in the world frame
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,  # None to use value in the USD file
            damping=None,  # None to use value in the USD file
        ),
    },
)

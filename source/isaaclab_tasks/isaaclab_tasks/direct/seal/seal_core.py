# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
import gymnasium as gym
import numpy as np
import torch
import os
import math
import glob
import shutil
import yaml
import pathlib
import csv
import sys
from datetime import datetime
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_tasks.direct.seal.dynamics.rotordynamics import create_rotor
from isaaclab_tasks.direct.seal.dynamics.jointdynamics import create_joint
from isaaclab.markers import CUBOID_MARKER_CFG, CYLINDER_MINUS_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.direct.seal.robots.squidbot import SquidbotRobot
from isaaclab_tasks.direct.seal.robots_cfg.squidbot_cfg import SquidbotRobotCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.seal import SEAL_CFG  # isort: skip


# Determine the IsaacLab directory
ISAACLAB_DIR = pathlib.Path(__file__).parents[5]


# Function to read YAML configuration
def read_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected configuration to be a dictionary, but got {type(config).__name__}")
    return config


class RobotState:
    """States for the robot"""

    IDLE = wp.constant(0)
    JETTING = wp.constant(1)
    CHARGING = wp.constant(2)


class TankState:
    """States for the water tank."""

    """Currently not used in the code."""

    EMPTY = wp.constant(0)
    CHARGING = wp.constant(1)
    FULL = wp.constant(2)
    JETTING = wp.constant(3)


class SealEnvWindow(BaseEnvWindow):
    """Window manager for the Seal environment."""

    def __init__(self, env: SealEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class EventCfg:
    """Configuration for randomization."""

    # TODO: Update this to match the SEAL environment.


@configclass
class SealEnvCfg(DirectRLEnvCfg):
    ####################################################################################################
    # parameters from yaml
    ####################################################################################################
    # Read the configuration
    config_path = os.path.join(ISAACLAB_DIR, "source/isaaclab_tasks/isaaclab_tasks/direct/seal/seal_cfg.yaml")
    config = read_config(config_path)

    robot_cfg: SquidbotRobotCfg = SquidbotRobotCfg()

    save_csv: bool = config.get("save_csv", False)
    save_files: list[str] = config.get("save_files", [])

    joints_pos_required: list[str] = config.get("joints_pos_required", [])
    joints_vel_required: list[str] = config.get("joints_vel_required", [])
    joints_define_morph: list[str] = config.get("joints_define_morph", [])
    rotors: list[tuple[str, str, str, bool, float, float, float, float, float, float]] = config.get("rotors", [])
    pro_joints: list[tuple[str, list[float]]] = config.get("pro_joints", [])
    active_joints: list[str] = config.get("active_joints", [])
    all_same_rotors: bool = config.get("all_same_rotors", True)
    thrust_to_weight: float = config.get("thrust_to_weight_ratio", 2.0)
    use_act_noise: bool = config.get("use_action_noise", False)
    act_noise_std: float = config.get("action_noise_std", 0.00)
    act_noise_bias_std: float = config.get("action_noise_bias_std", 0.00)
    use_obs_noise: bool = config.get("use_observation_noise", False)
    obs_noise_std: float = config.get("observation_noise_std", 0.00)
    obs_noise_bias_std: float = config.get("observation_noise_bias_std", 0.00)
    # time delay constant alpha, for the joints.
    Td_joint = 2.0  # with 2, takes 5 steps to reach 90%

    # if use_act_noise is True, add noise to the actions
    if use_act_noise:
        # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
        action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=act_noise_std, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=act_noise_bias_std, operation="abs"),
        )
    if use_obs_noise:
        # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
        observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=obs_noise_std, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=obs_noise_bias_std, operation="abs"),
        )

    # Add disturbance configuration
    disturbances_cfg: dict = config.get("disturbances", {})
    # print(f"Disturbances: {disturbances_cfg}")
    force_disturbance: dict = disturbances_cfg.get("forces", {})
    # print(f"Force Disturbance: {force_disturbance}")
    torque_disturbance: dict = disturbances_cfg.get("torques", {})
    # print(f"Torque Disturbance: {torque_disturbance}")
    ####################################################################################################
    # End of parameters from yaml
    ####################################################################################################
    # env
    episode_length_s = 100.0
    decimation = 2  # change when changing the dt
    action_scale = 0.5
    # -----------------------------
    # IMPORTANT: Use a dictionary with one discrete dimension and one continuous dimension
    # -----------------------------
    # Example discrete dimension: 3 => {0: NOTHING, 1: JETTING, 2: CHARGING}
    # Example continuous dimension: remainder for rotors + joints
    # Define the action_apce, one with set (3 elements) and len(active_joints)
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3 + len(active_joints),),
        dtype=np.float32,
    )
    # -----------------------------
    observation_space_robot = 9 + len(joints_pos_required) + len(joints_vel_required) + 6  # TODO: change hardcoded observation space numbers
    observation_space_task: int = 3  # default 3, will be updated in the env
    observation_space = observation_space_robot + observation_space_task
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        collision_group=-1,
        usd_path="/home/julia/IsaacLab/usd/water_terrain_long.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/omniverse://localhost/Projects/R3AMA/water.usd",
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = SEAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    """
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    """

    # reward scales
    lin_vel_reward_scale = -0.05  # default -0.05
    ang_vel_reward_scale = -0.01  # default -0.01
    distance_to_goal_reward_scale = 10.0
    gamma_reward_scale = 10.0
    yaw_reward_scale = 10.0
    goal_reached_reward_scale = 0.0
    time_penalty_second = -10.0  # - will be penalty
    out_of_bounds_penalty = -10000.0  # - will be penalty, prevent the robot from going out of the boundary


class SealEnv(DirectRLEnv):
    cfg: SealEnvCfg

    def __init__(self, cfg: SealEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot_api.run_setup(self._robot)

        # Instead of storing one big _actions tensor, we store separate buffers
        # for the discrete command and for the continuous motor command:
        self._discrete_cmd = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._continuous_cmd = torch.zeros(self.num_envs, len(self.cfg.active_joints), device=self.device)
        # Similarly, store previous actions for logging
        self._prev_discrete_cmd = torch.zeros_like(self._discrete_cmd)
        self._prev_continuous_cmd = torch.zeros_like(self._continuous_cmd)

        self._vec_joints = torch.zeros(self.num_envs, len(self.cfg.active_joints), device=self.device)
        self._temp_joint_targets = torch.zeros(self.num_envs, len(self.cfg.active_joints), device=self.device)
        self._joint_targets = torch.zeros(self.num_envs, len(self.cfg.active_joints), device=self.device)
        self._thrusts = torch.zeros(self.num_envs, len(self._robot.find_bodies(".*")[0]), 3, device=self.device)
        self._moments = torch.zeros(self.num_envs, len(self._robot.find_bodies(".*")[0]), 3, device=self.device)
        # Print out found bodies
        print(f"Found bodies: {self._robot.find_bodies('.*')}")
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # water tank and robot state
        self._water_tank_level = torch.zeros(self.num_envs, device=self.device)  # starts at 0
        self._robot_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=IDLE by default
        # Tweak these as desired:
        self._max_tank = 100.0  # Full tank
        self._min_jet_amount = 100.0
        self._charge_rate = 125.0  # per second
        self._jet_depletion_rate = 125.0  # per second
        self._water_surface_z = 0.0  # water surface at z=0

        # initialize the default base_link mass of robot. Call the _initialize_default_base_mass() function
        self._initialize_default_base_mass()

        if self.cfg.save_csv and self.num_envs <= 1:
            print("Saving data to CSV")
            os.makedirs("logs", exist_ok=True)
            # Get the current date and time to append to the filename
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Store the file in the 'logs' directory with date and time
            self.csv_file = os.path.join("logs", f"TEST_Seal_{current_time}.csv")
            self.header_written = False
            self.obs_log = None
            self.thrust_log = torch.zeros(self.num_envs, len(self.cfg.rotors), device=self.device)
        # --- Determine the experiment folder ---
        # We assume that the training script creates experiment folders under "logs/skrl/seal_direct"
        # and that the most recently modified folder in that directory is the current experiment folder.
        logs_base = os.path.join(os.getcwd(), "logs", "skrl", "seal_direct")
        experiment_folders = glob.glob(os.path.join(logs_base, "*"))
        if experiment_folders:
            # Choose the folder with the most recent modification time
            target_dir = max(experiment_folders, key=os.path.getmtime)
            print(f"[DEBUG] Determined experiment folder: {target_dir}")
        else:
            # Fallback if no experiment folder is found; use the current working directory.
            target_dir = os.getcwd()
            print(f"[DEBUG] No experiment folder found in {logs_base}; using {target_dir}")

        # --- Copy the important files into the experiment folder ---
        if self.cfg.save_files and "train.py" in " ".join(sys.argv):
            for rel_path in self.cfg.save_files:
                # Construct the absolute source path assuming ISAACLAB_DIR is defined as your project root.
                src_path = os.path.join(ISAACLAB_DIR, rel_path)
                dst_path = os.path.join(target_dir, os.path.basename(rel_path))
                try:
                    shutil.copy(src_path, dst_path)
                    print(f"[INFO] Copied {src_path} to {dst_path}")
                except Exception as e:
                    print(f"[ERROR] Could not copy {src_path} to {dst_path}: {e}")

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices for propeller hubs to apply thrust/torque
        self.rotor_body_ids = [self._robot.find_bodies(rotor_name)[0][0] for _, rotor_name, _, _, _, _, _, _, _, _ in self.cfg.rotors]
        self.rotorhub_body_ids = [self._robot.find_bodies(hub_name)[0][0] for _, _, hub_name, _, _, _, _, _, _, _ in self.cfg.rotors]
        self.rotor_Ds = torch.tensor([rotor[-1] for rotor in self.cfg.rotors], device=self.device)  # Assuming rotor_D is the last element
        print(f"Rotor body IDs: {self.rotor_body_ids}")
        print(f"Rotor hub body IDs: {self.rotorhub_body_ids}")

        # Initialize list for active joints (id, name, lower_limit, upper_limit)
        self.active_joints = []

        # Fetch joints from configuration
        joint_limits = self._robot.data.joint_limits
        for joint_name in self.cfg.active_joints:
            joint_id = self._robot.find_joints(joint_name)[0][0]
            lower_limit = joint_limits[0, joint_id, 0]
            upper_limit = joint_limits[0, joint_id, 1]
            print(f">>>>>>>> Active Joint: ID={joint_id}, Name={joint_name}, Limits=({lower_limit}, {upper_limit})")
            self.active_joints.append((joint_id, joint_name, lower_limit, upper_limit))

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        print(f"Robot mass: {self._robot_mass}")
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Initialize rotors
        self.rotors = []
        # If all rotors are the same, assign calculated max_thrust to all rotors
        if self.cfg.all_same_rotors:
            print(f"All rotors are same. max_thrust: {self.cfg.thrust_to_weight * self._robot_weight / len(self.cfg.rotors)}")
            for rotor_type, rotor_name, hub_name, is_ccw, Td_rotor, min_thrust, max_thrust, k_tau, k_tau_neg, rotor_D in self.cfg.rotors:
                rotor = create_rotor(
                    rotor_type=rotor_type,
                    num_envs=self.num_envs,
                    num_rotors=len(self.cfg.rotors),
                    is_ccw=is_ccw,
                    k_tau=k_tau,
                    k_tau_neg=k_tau_neg,
                    max_thrust=(self.cfg.thrust_to_weight * self._robot_weight) / len(self.cfg.rotors),
                    min_thrust=min_thrust,
                    Td_rotor=Td_rotor,
                    act_dt=self.cfg.sim.dt * self.cfg.decimation,
                    device=self.device,
                )
                self.rotors.append(rotor)
        # If rotors are different, assign max_thrust to each rotor
        else:
            print("Rotors are different. Initializing with different max_thrust.")
            for rotor_type, rotor_name, hub_name, is_ccw, Td_rotor, min_thrust, max_thrust, k_tau, k_tau_neg, rotor_D in self.cfg.rotors:
                rotor = create_rotor(
                    rotor_type=rotor_type,
                    num_envs=self.num_envs,
                    num_rotors=len(self.cfg.rotors),
                    is_ccw=is_ccw,
                    k_tau=k_tau,
                    k_tau_neg=k_tau_neg,
                    max_thrust=max_thrust,
                    min_thrust=min_thrust,
                    Td_rotor=Td_rotor,
                    act_dt=self.cfg.sim.dt * self.cfg.decimation,
                    device=self.device,
                )
                self.rotors.append(rotor)

        # Initialize joint dynamics
        self.pro_joints = []
        for joint_name, pos_to_effort_graph in self.cfg.pro_joints:
            pro_joint_id = self._robot.find_joints(joint_name)[0][0]
            joint = create_joint(
                joint_type="linearint",
                num_int_points=1000,
                num_envs=self.num_envs,
                pos_to_effort_graph=torch.tensor(pos_to_effort_graph, device=self.device),
                device=self.device,
            )
            self.pro_joints.append((pro_joint_id, joint))

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.robot_api = SquidbotRobot(self.cfg.robot_cfg, robot_uid=0, num_envs=self.num_envs, device=self.device, sim_dt=self.cfg.sim.dt)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process the raw policy actions, enforce tank constraints, apply thrust & joints."""
        # ---------------------------------------------------------------
        # 1) Extract discrete vs. continuous from the raw action
        # ---------------------------------------------------------------
        discrete_logits = actions[:, :3].to(self.device)
        continuous_cmd = actions[:, 3:].to(self.device)

        # The candidate discrete command is argmax of the first 3
        candidate_cmd = discrete_logits.argmax(dim=1).long()
        self._continuous_cmd[:] = continuous_cmd.clamp(-1.0, 1.0)

        # ---------------------------------------------------------------
        # 2) Vectorized "state machine" for JET / CHARGE constraints
        # ---------------------------------------------------------------
        dt = self.cfg.sim.dt * self.cfg.sim.render_interval

        # Identify which envs are underwater
        root_z = self._robot.data.root_pos_w[:, 2]
        underwater_mask = root_z < self._water_surface_z

        # Are we IDLE vs. JETTING vs. CHARGING right now?
        curr_state = self._robot_state
        next_state = curr_state.clone()

        is_idle = curr_state == int(RobotState.IDLE)
        is_jetting = curr_state == int(RobotState.JETTING)
        is_charging = curr_state == int(RobotState.CHARGING)

        # candidate_cmd meaning: 0 => do nothing, 1 => jet, 2 => charge
        want_jet = candidate_cmd == 1
        want_charge = candidate_cmd == 2

        # Start JET if IDLE + want_jet + underwater + tank > min
        can_start_jet = is_idle & want_jet & underwater_mask & (self._water_tank_level >= self._min_jet_amount)
        next_state = torch.where(can_start_jet, torch.tensor(int(RobotState.JETTING), device=self.device), next_state)

        # Start CHARGE if IDLE + want_charge + underwater + tank not full
        can_start_charge = is_idle & want_charge & underwater_mask & (self._water_tank_level < self._max_tank)
        next_state = torch.where(can_start_charge, torch.tensor(int(RobotState.CHARGING), device=self.device), next_state)

        # Remain in JETTING if currently JETTING and tank not empty
        # If empty => revert to IDLE
        empty_tank = self._water_tank_level <= 0.0
        stop_jet = is_jetting & empty_tank
        # If empty => go IDLE
        next_state = torch.where(stop_jet, torch.tensor(int(RobotState.IDLE), device=self.device), next_state)

        # Remain in CHARGING if currently CHARGING, tank not full, and still underwater
        # If tank full or left water => go IDLE
        full_tank = self._water_tank_level >= self._max_tank
        stop_charge = is_charging & (full_tank | (~underwater_mask))
        next_state = torch.where(stop_charge, torch.tensor(int(RobotState.IDLE), device=self.device), next_state)

        # Write next state
        self._robot_state = next_state

        # Derive the "actual" discrete command from final state
        new_discrete_cmd = torch.zeros_like(self._discrete_cmd)
        new_discrete_cmd = torch.where(next_state == int(RobotState.JETTING), torch.tensor(1, device=self.device), new_discrete_cmd)
        new_discrete_cmd = torch.where(next_state == int(RobotState.CHARGING), torch.tensor(2, device=self.device), new_discrete_cmd)

        self._discrete_cmd[:] = new_discrete_cmd

        # ---------------------------------------------------------------
        # 3) Update water tank level
        # ---------------------------------------------------------------
        is_charging = self._robot_state == int(RobotState.CHARGING)
        is_jetting = self._robot_state == int(RobotState.JETTING)

        # Charging => add
        self._water_tank_level = torch.where(is_charging, self._water_tank_level + (self._charge_rate * dt), self._water_tank_level)
        # Jetting => subtract
        self._water_tank_level = torch.where(is_jetting, self._water_tank_level - (self._jet_depletion_rate * dt), self._water_tank_level)
        # Clamp
        self._water_tank_level.clamp_(0.0, self._max_tank)
        # Update robot base_link mass based on water level
        self._update_base_link_mass()

        # ---------------------------------------------------------------
        # 4) Use the final discrete command to apply thrust, etc.
        # ---------------------------------------------------------------
        # For demonstration, we feed self._full_actions=1.0 to rotor, but only apply
        # if _discrete_cmd == 1 => JET. You can refine to scale thrust or similar.
        self._full_actions = torch.ones_like(self._discrete_cmd, dtype=torch.float, device=self.device)

        # Clear thrust / torque
        self._thrusts[:] = 0.0
        self._moments[:] = 0.0

        # For each rotor, compute thrust
        for idx, rotor in enumerate(self.rotors):
            thr, mom = rotor.get_thrust_and_torque(self._full_actions)
            self._thrusts[:, self.rotor_body_ids[idx], 2] = thr
            self._moments[:, self.rotorhub_body_ids[idx], 2] = mom

        # Now zero out thrust if not in "JET" (discrete_cmd == 1)
        jet_mask = self._discrete_cmd == 1
        jet_mask_3d = jet_mask.view(-1, 1, 1).expand(-1, self._thrusts.shape[1], 3)
        self._thrusts = torch.where(jet_mask_3d, self._thrusts, torch.zeros_like(self._thrusts))
        self._moments = torch.where(jet_mask_3d, self._moments, torch.zeros_like(self._moments))

        if self.cfg.save_csv and self.num_envs <= 1:
            for idx, rotor in enumerate(self.rotors):
                self.thrust_log[:, idx] = self._thrusts[:, self.rotor_body_ids[idx], 2]
        # Make thrust to 0.0 if the rotor’s world z-position is above 0.0
        # But note that actually the thrust works in the air as well.
        """
        for idx, rotor in enumerate(self.rotors):
            rotor_z_positions = self._robot.data.body_com_state_w[:, idx, 2]
            self._thrusts[:, self.rotor_body_ids[idx], 2] = torch.where(
                rotor_z_positions <= 0.0,
                self._thrusts[:, self.rotor_body_ids[idx], 2],
                torch.zeros_like(self._thrusts[:, self.rotor_body_ids[idx], 2]),
            )
        """
        # Compute the physics
        self.robot_api.compute_physics(water_tank_level=self._water_tank_level)

        # Handle joint actions if there are any active joints
        if len(self.active_joints) > 0:
            joint_actions = self._continuous_cmd
            # Check if we have active_joints
            for idx, (joint_id, joint_name, lower_limit, upper_limit) in enumerate(self.active_joints):
                # Map action from [-1, 1] to [lower_limit, upper_limit]
                self._temp_joint_targets[:, idx] = lower_limit + (joint_actions[:, idx] + 1) * (upper_limit - lower_limit) / 2.0
            # Apply smoothing
            self._vec_joints += (self._temp_joint_targets - self._vec_joints) / self.cfg.Td_joint
            self._joint_targets = self._vec_joints
            # debug: print joint_actions and joint_targets
            # print(f"Joint actions: {joint_actions[0]}")
            # print(f"Joint targets: {self._joint_targets[0]}")

        # Handle joint dynamics (programmable joints)
        self.joint_efforts = torch.zeros(self.num_envs, len(self.pro_joints), device=self.device)
        for idx, joint in enumerate(self.pro_joints):
            # print(self._robot.data.joint_pos[:, joint[0]])
            self.joint_efforts[:, idx] = joint[1].get_joint_effort(self._robot.data.joint_pos[:, joint[0]])

        # log csv
        if self.cfg.save_csv and self.num_envs <= 1:
            self._log_csv()

    def _apply_action(self):
        # debug self._actions & self._thrusts & self._moments
        # print(f"actions: {self._actions[0]}")
        # print(f"Thrusts: {self._thrusts[0]}")
        # print(f"Moments: {self._moments[0]}")
        self._robot.set_external_force_and_torque(self._thrusts, self._moments)

        # Apply computed physics forces
        self.robot_api.apply_physics(self._robot)

        if len(self.active_joints) > 0:
            self._robot.set_joint_position_target(self._joint_targets, joint_ids=[joint[0] for joint in self.active_joints])
        if len(self.pro_joints) > 0:
            self._robot.set_joint_effort_target(self.joint_efforts, joint_ids=[joint[0] for joint in self.pro_joints])

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )

        joint_pos_indices = [self._robot.find_joints(name)[0][0] for name in self.cfg.joints_pos_required]
        joint_vel_indices = [self._robot.find_joints(name)[0][0] for name in self.cfg.joints_vel_required]
        joint_pos_obs = self._robot.data.joint_pos[:, joint_pos_indices]
        joint_vel_obs = self._robot.data.joint_vel[:, joint_vel_indices]
        tank_level_obs = self._water_tank_level.unsqueeze(-1)  # shape (num_envs, 1)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                joint_pos_obs,
                joint_vel_obs,
                desired_pos_b,
                self._prev_discrete_cmd,
                self._prev_continuous_cmd,
                tank_level_obs,
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        if self.cfg.save_csv and self.num_envs <= 1:
            self.obs_log = {
                "root_lin_vel_b": self._robot.data.root_lin_vel_b,
                "root_ang_vel_b": self._robot.data.root_ang_vel_b,
                "projected_gravity_b": self._robot.data.projected_gravity_b,
                "joint_pos_obs": joint_pos_obs,
                "joint_vel_obs": joint_vel_obs,
                "desired_pos_b": desired_pos_b,
                "prev_discrete_cmd": self._prev_discrete_cmd,
                "prev_continuous_cmd": self._prev_continuous_cmd,
            }

        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        self._prev_discrete_cmd = self._discrete_cmd.clone()
        self._prev_continuous_cmd = self._continuous_cmd.clone()

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < -10.0, self._robot.data.root_pos_w[:, 2] > 10.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and len(env_ids) != 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._continuous_cmd[env_ids] = 0.0
        self._prev_continuous_cmd[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # Randomize the robot's initial position
        random_offsets = torch.zeros_like(self._terrain.env_origins[env_ids])
        random_offsets[:, 0] = torch.FloatTensor(len(env_ids)).uniform_(-0.0, 0.0)  # Randomize x position
        random_offsets[:, 1] = torch.FloatTensor(len(env_ids)).uniform_(-0.0, 0.0)  # Randomize y position
        random_offsets[:, 2] = torch.FloatTensor(len(env_ids)).uniform_(-7.0, -5.0)  # Randomize z position
        default_root_state[:, :3] += random_offsets
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._water_tank_level[env_ids] = 0.0
        self._robot_state[env_ids] = int(RobotState.IDLE)

        # Logging
        final_distance_to_goal = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Glass/Red_Glass.mdl",
                )
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            if not hasattr(self, "thrust_visualizer"):
                marker_cfg = CYLINDER_MINUS_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/thrust"
                self.thrust_visualizer = VisualizationMarkers(marker_cfg)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "thrust_visualizer"):
                self.thrust_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        thrust_pos_w = self._robot.data.body_link_pos_w[:, self.rotor_body_ids[0], :]
        thrust_quat = self._robot.data.body_link_quat_w[:, self.rotor_body_ids[0], :]
        # scales: Scale applied before any rotation is applied. Shape is (M, 3).
        thrust_scale = torch.ones_like(self._thrusts[:, self.rotor_body_ids[0], :]) * 0.03
        thrust_scale[:, 2] = self._thrusts[:, self.rotor_body_ids[0], 2] * 0.03
        self.thrust_visualizer.visualize(thrust_pos_w, thrust_quat, thrust_scale)

    def _log_csv(self):
        # if there is not self.obs_log yet, skip
        if self.obs_log is None:
            print("No observation log found. Skipping CSV logging.")
            return
        if not self.header_written:
            self.header_written = True
            headers = [
                "time",
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_w",
                "root_ang_vel_w",
                "discrete actions",
                "continuous actions",
                "thrusts",
                "joint_targets",
                "body_com_state_w",
                "buoyancy_force",
                "hydrodynamic_force",
                "aerodynamic_force",
                "lift_force",
                "addedmass_force",
                "combined_force_root",
            ]
            for key, obs_data in self.obs_log.items():
                headers.append(key)
            with open(self.csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)

        self.time_log = self.episode_length_buf[0].item() * self.cfg.sim.dt * self.cfg.sim.render_interval

        # Prepare the thrust_log
        thrust_log_list = self.thrust_log.tolist()
        # Check if thrust_log_list is a double list with only one inner list
        if isinstance(thrust_log_list, list) and len(thrust_log_list) == 1 and isinstance(thrust_log_list[0], list):
            thrust_log_list = thrust_log_list[0]  # Remove the outer list

        # Get additional force data from the robot_api
        buoyancy_force = self.robot_api.buoyancy_force[0].tolist() if hasattr(self.robot_api, "_buoyancy_force") else "N/A"
        hydrodynamic_force = self.robot_api.hydrodynamic_force[0].tolist() if hasattr(self.robot_api, "_hydrodynamic_force") else "N/A"
        aerodynamic_force = self.robot_api.aerodynamic_force[0].tolist() if hasattr(self.robot_api, "_aerodynamic_force") else "N/A"
        lift_force = self.robot_api.lift_force[0].tolist() if hasattr(self.robot_api, "_lift_force") else "N/A"
        addedmass_force = self.robot_api.addedmass_force[0].tolist() if hasattr(self.robot_api, "_addedmass_force") else "N/A"
        combined_force_root = self.robot_api.combined_force_root[0].tolist() if hasattr(self.robot_api, "_combined_force_root") else "N/A"

        log_data = [
            self.time_log,
            self._robot.data.root_pos_w[0].tolist(),
            self._robot.data.root_quat_w[0].tolist(),
            self._robot.data.root_lin_vel_w[0].tolist(),
            self._robot.data.root_ang_vel_w[0].tolist(),
            self._discrete_cmd[0].tolist(),
            self._continuous_cmd[0].tolist(),
            thrust_log_list,
            self._joint_targets[0].tolist(),
            self._robot.data.body_com_state_w[0].tolist(),
            buoyancy_force,
            hydrodynamic_force,
            aerodynamic_force,
            lift_force,
            addedmass_force,
            combined_force_root,
        ]

        # Append the observation logs as strings
        for key, obs_data in self.obs_log.items():
            if len(obs_data.shape) > 1:
                log_data.append(str(obs_data[0].tolist()))  # Convert list to string
            else:
                log_data.append(str([obs_data[0].item()]))  # Wrap single value in list and convert to string

        with open(self.csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(log_data)

    def _initialize_default_base_mass(self):
        # Retrieve the index of the base_link body
        base_link_idx = self._robot.find_bodies("base_link")[0][0]
        # Store the default base_link mass once (assuming _robot.root_physx_view.get_masses() returns a tensor of shape [num_envs, num_bodies])
        self._default_base_mass = self._robot.root_physx_view.get_masses().to(self.device)[:, base_link_idx].clone()

    def _update_base_link_mass(self):
        # Retrieve the index of the base_link body
        base_link_idx = self._robot.find_bodies("base_link")[0][0]

        # Compute the new mass for base_link using the stored default mass plus the water tank level (assumed to be in kg)
        new_base_mass = self._default_base_mass + self._water_tank_level * (
            self.cfg.robot_cfg.water_tank_mass / 100.0
        )  # both tensors of shape (num_envs,)

        # Obtain the full mass tensor from the physics view
        current_masses = self._robot.root_physx_view.get_masses().to(self.device)
        updated_masses = current_masses.clone()

        # Update only the base_link mass with the new value
        updated_masses[:, base_link_idx] = new_base_mass

        # Apply the updated mass tensor to the simulation (transfer to CPU if required by the API)
        self._robot.root_physx_view.set_masses(updated_masses.to("cpu"), torch.arange(self.num_envs, device="cpu"))


wp.init()

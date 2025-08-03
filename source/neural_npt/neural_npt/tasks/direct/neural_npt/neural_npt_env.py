# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import numpy
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_error_magnitude,
    combine_frame_transforms,
    matrix_from_quat,
    subtract_frame_transforms,
    euler_xyz_from_quat)

from isaaclab.sensors import Camera
from isaaclab.markers import VisualizationMarkers

from .neural_npt_env_cfg import NeuralNPTEnvCfg


class NeuralNPTEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: NeuralNPTEnvCfg

    def __init__(self, cfg: NeuralNPTEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._device = self.sim.device
        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_default_jointPos = self._robot.data.default_joint_pos[:, :].clone()

        self.tray_link_idx = self._robot.find_bodies("tray")[0][0]
        self.robot_root_pos_w = self._robot.data.root_link_state_w[:, :3]
        self.robot_root_quat_w = self._robot.data.root_link_state_w[:, 3:7]
        self.default_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat((self.num_envs, 1))

        assets_sizes = torch.tensor(self.cfg.assets_sizes, device=self.device, dtype=torch.float32)
        self.assets_sizes = assets_sizes[:self.num_envs]


        if self.cfg.evaluating:
            self.targetPositions = torch.tensor(self.cfg.targetPositions, device=self.device, dtype=torch.float32)
            self.initJointPositions = torch.tensor(self.cfg.initJointPositions, device=self.device, dtype=torch.float32)

        self.BetaDist = torch.distributions.Beta(4, 1)
        self.init_state_buffers()
        self.set_goalPos_ranges()
        self.reset_command_Pos(self._robot._ALL_INDICES)

        self.rgb, self.depth, self.segment, self.cam_info = [], [], [], []
        self.scale = 0.0

                

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._stand = RigidObject(self.cfg.stand)
        self._block = RigidObject(self.cfg.block)
        self.scene.rigid_objects["block"] = self._block

        if self.cfg.visualizeGoals:
            self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
            #self.ee_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)


        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process Jerk actions from  to joint position"""
        self.actions = actions.clone()
        self.actions = self.last_action + self.cfg.AccRate*self.actions
        self.cmd_acc = self.actions.clone()* self.cfg.action_scale

        # For safer real robot deployment, we limit the maximum acceleration to 20 m/s^2
        #  limit max velocity to 3 m/s
        #velEst = self.joint_qd.clone() + self.cmd_acc.clone() * (self.cfg.sim.dt*self.cfg.decimation)
        #high_velEst = (torch.abs(velEst) > 3.0).float()
        #clipped_acc = torch.sign(self.cmd_acc)*torch.abs(3.0-torch.abs(self.joint_qd.clone()))/self.cfg.sim.dt/self.cfg.decimation
        #self.cmd_acc = self.cmd_acc.clone()* (1-high_velEst) + high_velEst*clipped_acc
        
        # set acceleration to zero if the goal is reached
        #acc_notreached = self.cmd_acc*(1-self.reachSuccess.float()).unsqueeze(-1) 
        #acc_reached = -self.reachSuccess.float().unsqueeze(-1)*(self.joint_qd/self.cfg.sim.dt/self.cfg.decimation)
        #acc_reached = torch.clamp(acc_reached, -0.5#, 0.5)
        #self.cmd_acc = acc_reached + acc_notreached#

            
    def _apply_action(self):
        """Apply the joint position targets to the robot"""
        dt = self.cfg.sim.dt 
        # update the joint positions and velocities using integrated acceleration (we don't use the physics engine for this)
        self.joint_q = self.joint_q + self.joint_qd * (dt) + 0.5 * self.cmd_acc * (dt) ** 2
        self.joint_qd = self.joint_qd + self.cmd_acc*(dt)
        commanded_positions = self.joint_q.clone()

        self.robot_dofPos_targets = torch.clamp(commanded_positions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        
        self._robot.set_joint_position_target(self.robot_dofPos_targets)

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        truncated = self.episode_length_buf >= (self.max_episode_length)
        if self.cfg.evaluating:
            terminated = truncated.clone()
        else:
            terminated = self.is_dropped(self.eePos_w, self.block_pos_ee, self.eeQuat_w, self.block_quat_w)

        updateGoalIds = (self.episode_length_buf % self.cfg.maxSteps == 0).nonzero().reshape(1, -1)
        nElements = updateGoalIds.numel()
        if nElements > 0:
            self.reset_command_Pos(updateGoalIds)

        return terminated, truncated

    ################################################## RESET #######################################################
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        self.sample_env_state(env_ids)
        self.reset_command_Pos(env_ids)  # Target Positions

        self.ee_maxVel[env_ids] = 0.0
        self.last_action[env_ids] = torch.zeros_like(self.last_action[env_ids])
        self.joint_q[env_ids] = self._robot.data.joint_pos[env_ids].clone()
        self.joint_qd[env_ids] = self._robot.data.joint_vel[env_ids].clone()
        self.joint_qdd[env_ids] = torch.zeros_like(self.joint_qdd[env_ids])


        self.block_mass = self._block.root_physx_view.get_masses().clone().to(device=self.device)
        block_props= self._block.root_physx_view.get_material_properties().clone().to(device=self.device)
        self.block_staticFriction  = block_props[:, 0, 0].reshape(-1, 1)
        self.block_dynamicFriction = block_props[:, 0, 1].reshape(-1, 1)
        self.block_com = self._block.root_physx_view.get_coms().clone()[:,:3].to(device=self.device)

        
        # update rewards scales based on the current episode length
        self.update_curriculum()
       

    def compute_observation(self, target="policy") -> torch.Tensor:
        # Compute the observation based on the target type for assymetric actor-critic
        relStep = (self.episode_length_buf % self.cfg.maxSteps).float() / self.cfg.maxSteps
        relStep = relStep.reshape(-1, 1)


        eePos = self.eePos_w - self.robot_root_pos_w
        block_pos = self.block_pos_w - self.robot_root_pos_w
        
        ee_rotMat = matrix_from_quat(self.eeQuat_w)
        ee_rotMat = ee_rotMat[:, :, :2].reshape(-1, 6)


        block_rotMat = matrix_from_quat(self.block_quat_w)
        block_rotMat = block_rotMat[:, :, :2].reshape(-1, 6)
       

        if target == "policy":
            obs = torch.cat(
                (
                    self.joint_q ,              # 6 +
                    self.joint_qd ,             # 6 + --> 12
                    self.pos_command_b,         # 3 + --> 15
                    self.assets_sizes,          # 3 + --> 18
                    self.last_action ,          # 6 + --> 24
                    self.block_mass,            # 1 + --> 25
                    self.block_staticFriction,  # 1 + --> 26
                    self.block_dynamicFriction, # 1 + --> 27
                    relStep,                    # 1 + --> 28    -- useful for stable learning (check if needed)
                    self.last_cmd_acc,          # 6 + --> 34    -- redundant with last action (check if needed)

                ),
                dim=-1,
            )

        if target == "critic":
            obs = torch.cat(
                (
                    self.joint_pos,             # 6 +
                    self.joint_qd,              # 6 + --> 12
                    eePos,                      # 3 + --> 15
                    ee_rotMat,                  # 6 + --> 21
                    self.eeVel,                 # 6 + --> 27
                    block_pos,                  # 3 + --> 30
                    block_rotMat,               # 6 + --> 36
                    self.block_vel,             # 6 + --> 42
                    self.pos_command_b,         # 3 + --> 45
                    self.assets_sizes,          # 3 + --> 48
                    self.last_action ,          # 6 + --> 54
                    self.block_mass,            # 1 + --> 55
                    self.block_staticFriction,  # 1 + --> 56
                    self.block_dynamicFriction, # 1 + --> 57
                    relStep,                    # 1 + --> 58  -- useful for stable learning
                    self.last_cmd_acc,          # 6 + --> 64  -- redundant with last action (check if needed)
                ),
                dim=-1,
            )

            # Append center of mass if randomization is enabled
            if self.cfg.RandCOM:
                obs = torch.cat((obs, self.block_com), dim=-1)
        
        return obs.clone()
    
    def _get_observations(self) -> dict:   
        self._compute_intermediate_values()
        obs = self.compute_observation("policy")
        state = self.compute_observation("critic")
        
        observations = {"policy": obs, "critic": state}
        return observations


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # ee State
        eeState_w = self._robot.data.body_link_state_w[env_ids, self.tray_link_idx].clone()
        self.eePos_w[env_ids] = eeState_w[:, :3]
        self.eeQuat_w[env_ids] = eeState_w[:, 3:7]
        self.eeVel[env_ids] = eeState_w[:, 7:]

        # block State
        self.block_pos_w[env_ids] = self._block.data.body_link_pos_w[env_ids, 0].clone()
        self.block_quat_w[env_ids] = self._block.data.body_link_quat_w[env_ids, 0].clone()
        self.block_vel[env_ids] = self._block.data.body_link_vel_w[env_ids, 0].clone()

        self.block_pos_ee[env_ids] = self.block_pos_w[env_ids].clone()
        self.block_pos_ee[env_ids, 2] -= self.assets_sizes[env_ids, 2] / 2

        # joint state
        self.joint_pos[env_ids] = self._robot.data.joint_pos[env_ids].clone()
        self.joint_vel[env_ids] = self._robot.data.joint_vel[env_ids].clone()


        self.ee_maxVel[env_ids] = torch.maximum(self.ee_maxVel[env_ids], torch.linalg.vector_norm(self.eeVel[env_ids,:3], dim=1, ord=2))

        
    def is_dropped(self, eePos_w, block_pos_ee, eeQuat_w, block_quat_w) -> torch.Tensor:
        block_posError = torch.linalg.vector_norm(eePos_w - block_pos_ee, dim=1, ord=1)
        block_oriError = quat_error_magnitude(eeQuat_w, block_quat_w)

        is_dropped = (block_posError > 0.2) | (block_oriError > 1.0)
        return is_dropped

    def _get_rewards(self) -> torch.Tensor:
        relStep = (self.episode_length_buf % self.cfg.maxSteps).float() / self.cfg.maxSteps
        steady_state = relStep > 0.5

        """Reaching"""
        # Dense Rewards:
        ## Position Error
        ee_posError = torch.linalg.vector_norm(self.eePos_w - self.pos_command_w, dim=1, ord=2)

        ## orientation not neccessary
        ##################################################################################
        # Regularization
        ## Action Smoothness
        action_norm = torch.linalg.vector_norm(self.actions, dim=1, ord=1)
        action_rate_penalty = torch.linalg.vector_norm((self.actions - self.last_action), dim=1, ord=1)
        self.last_action = self.actions.clone()

        jointAcc_norm = torch.linalg.vector_norm(self.cmd_acc, dim=1, ord=1)
        jointAcc_rate_penalty = torch.linalg.vector_norm((self.cmd_acc - self.last_cmd_acc), dim=1, ord=1)
        self.last_cmd_acc = self.cmd_acc.clone()

        

        # Stability and Dynamic Grasping
        # block pos distance
        block_posError = torch.linalg.vector_norm(self.eePos_w - self.block_pos_ee, dim=1, ord=2)
        block_posError_fine_grained = logistic_kernel(block_posError, std=0.05)
        # block quat error
        block_oriError = quat_error_magnitude(self.eeQuat_w, self.block_quat_w)
        block_oriError_fine_grained = logistic_kernel(block_oriError, std=0.05)



        # task_success
        is_success = (block_posError < 0.1) & (block_oriError < 0.1) & (ee_posError < 0.1) 

        # Uncomment to add additional dynamic grasping reward if desired
        success_reward = is_success.float() * (logistic_kernel(-(ee_posError), std=0.05))# +\
                                                #logistic_kernel(-(block_posError), std=0.01)*\
                                                #logistic_kernel(-(block_oriError), std=0.01))

        dropped = self.is_dropped(self.eePos_w, self.block_pos_ee, self.eeQuat_w, self.block_quat_w)



        self.reachSuccess = (block_posError < 0.02) & (block_oriError < 0.02) & (ee_posError < 0.01)
        

        # add further regularization as success reward increases
        reg_scale = success_reward.mean()

        rewards = (
            self.cfg.live_reward
            - self.cfg.ee_posError_rewScale * ee_posError
            # uncomment to add dynamic grasping rewards
            #- self.cfg.block_posError_rewScale * block_posError
            #- self.cfg.block_oriError_rewScale * block_oriError

            + self.cfg.success_rewScale * success_reward
            
            - self.cfg.action_penalty_rewScale * jointAcc_norm* reg_scale
            - self.cfg.action_penalty_rewScale * jointAcc_rate_penalty*reg_scale

            - self.cfg.action_penalty_rewScale * action_norm* reg_scale*10
            - self.cfg.action_penalty_rewScale * action_rate_penalty*reg_scale*10
        )



        self.extras["log"] = {
            "action_penalty": (action_norm).mean(),
            "ee_maxVel": (self.ee_maxVel).mean(),
            "success_reward": (success_reward).mean(),
            "scale": self.scale,
            "step": relStep,
            "Dropped": dropped,
            "is_success": is_success.float(), 
            "block_state":  [],
            "eePos_Error": ee_posError,
            "blockPos_Error": block_posError,
            "blockOri_Error": block_oriError,
            "eeVel": torch.linalg.vector_norm(self.eeVel[:, :3], dim=1, ord=2),
            "JointAcc": self.cmd_acc,
        }

        return rewards

    ### Helper Functions ####
    def set_goalPos_ranges(self):
        """Set the ranges for sampling the goal positions"""
        self.tgt_radius_range = (0.6, 0.9)
        self.tgt_yaw_range = (-math.pi / 2, math.pi / 2)
        self.tgt_pitch_range = (-math.pi / 6, 2*math.pi /6)


    def init_state_buffers(self):
        """Initialize the state buffers for the environment"""
        self.init_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.init_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.block_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.block_pos_ee = torch.zeros((self.num_envs, 3), device=self.device)
        self.block_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.block_vel = torch.zeros((self.num_envs, 6), device=self.device)

        self.eePos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.eeQuat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.eeVel = torch.zeros((self.num_envs, 6), device=self.device)

        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.pos_command_b= torch.zeros((self.num_envs, 3), device=self.device)
        self.pos_command_w = torch.zeros((self.num_envs, 3), device=self.device)

        self.ee_maxVel = torch.zeros((self.num_envs,), device=self.device)

        self.last_action = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.joint_q = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.joint_qd = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.joint_qdd = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.cmd_acc = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.last_cmd_acc = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)


        self.reachSuccess = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)


        
    def update_curriculum(self):
        avg_ep_length = 1.1 * self.episode_length_buf.float().mean() / (self.cfg.maxSteps / 2)
        scale = min(avg_ep_length, 1)
        scale = 0.1 + 0.9*scale
        self.scale = self.scale + 0.001 if scale > self.scale else self.scale
        self.scale = min(self.scale, 1.0)
        self.cfg.live_reward = 1.0*(1.1 - self.scale)
        self.cfg.action_penalty_rewScale = 0.5* self.scale
        self.cfg.joint_vel_rewScale = 0.1 * self.scale
        self.cfg.ee_posError_rewScale = 1.0 * self.scale
        self.cfg.block_posError_rewScale = 1.0 * self.scale
        self.cfg.block_oriError_rewScale = 1.0 * self.scale
        self.cfg.success_rewScale = 50.0 * self.scale
        

    def sample_env_state(self, env_ids):
        # Sample initial joint positions and velocities for the robot
        # Then set the block pose and velocity

        self.init_joint_pos[env_ids] = self._robot.data.default_joint_pos[env_ids]
        
        self.init_joint_pos[env_ids] = self.initJointPositions[env_ids] if self.cfg.evaluating else self._sample_init_joint(self.init_joint_pos[env_ids])
        self._robot.write_joint_state_to_sim(self.init_joint_pos[env_ids], self.init_joint_vel[env_ids], env_ids=env_ids)


        block_pose = self._robot.data.body_link_state_w[env_ids, self.tray_link_idx][:, :7].clone()
        # shift block pose up by half its height
        block_pose[:, 2] += self.assets_sizes[env_ids, 2] / 2

        block_vel_envIds = torch.zeros((len(env_ids), 6), device=self.device)
        # Randomize the block position if enabled
        if self.cfg.RandObjPos and self.cfg.training:
            radius = 0.9*torch.rand(len(env_ids), device=self.device)*0.1
            theta = 2*math.pi*torch.rand(len(env_ids), device=self.device)
            block_pose[:, 0] += radius * torch.cos(theta)
            block_pose[:, 1] += radius * torch.sin(theta)
            
        self._block.write_root_pose_to_sim(block_pose, env_ids=env_ids)
        self._block.write_root_velocity_to_sim(block_vel_envIds, env_ids=env_ids)
        
    def _sample_init_joint(self, init_joint_pos):
        """Sample joints ensuring initial horizontal orientation and randomize the position"""
        n = init_joint_pos.shape[0]
        joint1_shift1 = torch.pi * (self.BetaDist.sample((n,)).to(self.device).squeeze() - 0.5)
        joint1_shift2 = -torch.pi * (self.BetaDist.sample((n,)).to(self.device).squeeze() - 0.5)
        sampledShift = torch.where(torch.rand(n,device=self.device) > 0.5, joint1_shift1, joint1_shift2)
        init_joint_pos[:, 0] = init_joint_pos[:, 0] + sampledShift
        
        # Sample random angle between -45 and 45 degrees
        angle = math.pi / 2 * (torch.rand_like(init_joint_pos[:, 0]) - 0.5) 
        coeff1 = 2 * torch.rand_like(init_joint_pos[:, 0]) - 1
        coeff2 = 2 * torch.rand_like(init_joint_pos[:, 0]) - 1
        coeff3 = coeff1 + coeff2
        init_joint_pos[:, 1] += coeff1 * angle
        init_joint_pos[:, 3] += coeff2 * angle
        init_joint_pos[:, 2] -= coeff3 * angle

        return init_joint_pos

    def reset_command_Pos(self, env_ids):
        """Reset the target position for the robot end-effector"""
        # sample a target yaw angle that is either random, pi/2 or -pi/2, or the farthest goal
        if self.cfg.evaluating:
            self.pos_command_b[env_ids] = self.targetPositions[env_ids]
        else:
            env_ids = env_ids.reshape(-1)
            n = len(env_ids)
            tgt_radius1 = self.tgt_radius_range[0] + (self.tgt_radius_range[1]-self.tgt_radius_range[0])* self.BetaDist.sample((n,)).to(self.device).squeeze()
            tgt_radius2 = self.tgt_radius_range[1] + (self.tgt_radius_range[0]-self.tgt_radius_range[1])* self.BetaDist.sample((n,)).to(self.device).squeeze()
            tgt_radius = torch.where(torch.rand(n,device=self.device) > 0.5, tgt_radius1, tgt_radius2)

            tgt_pitch1 = self.tgt_pitch_range[0] + (self.tgt_pitch_range[1] - self.tgt_pitch_range[0]) * self.BetaDist.sample((n,)).to(self.device).squeeze()
            tgt_pitch2 = self.tgt_pitch_range[1] + (self.tgt_pitch_range[0] - self.tgt_pitch_range[1]) * self.BetaDist.sample((n,)).to(self.device).squeeze()
            tgt_pitch = torch.where(torch.rand(n, device=self.device) > 0.5, tgt_pitch1, tgt_pitch2)
            
            if torch.rand(1) < 0.33:
                # random sample
                tgt_yaw = torch.empty(n, device=self.device).uniform_(*self.tgt_yaw_range)
            elif torch.rand(1) < 0.66:
                # sample pi/2 or -pi/2
                joint1_pos = self.init_joint_pos[env_ids, 0]
                tgt_yaw = -torch.sign(joint1_pos) * torch.pi / 2
            else:
                # sample farthest goal
                tgt_yaw_set = torch.empty((n,10), device=self.device).uniform_(*self.tgt_yaw_range)
                joint1_pos = self.init_joint_pos[env_ids, 0].unsqueeze(1).repeat(1, 10)
                distance = torch.abs(tgt_yaw_set - joint1_pos)
                tgt_yaw = tgt_yaw_set[torch.arange(n), distance.argmax(dim=-1)]

            self.pos_command_b[env_ids, 0] = tgt_radius * torch.cos(tgt_yaw) * torch.cos(tgt_pitch)
            self.pos_command_b[env_ids, 1] = tgt_radius * torch.sin(tgt_yaw) * torch.cos(tgt_pitch)
            self.pos_command_b[env_ids, 2] = tgt_radius * torch.sin(tgt_pitch)

        self.pos_command_w[env_ids], _ = combine_frame_transforms(
            self.robot_root_pos_w[env_ids], self.robot_root_quat_w[env_ids], self.pos_command_b[env_ids]
        )

        self.joint_q[env_ids] = self._robot.data.joint_pos[env_ids].clone()
        self.joint_qd[env_ids] = self._robot.data.joint_vel[env_ids].clone()
        if self.cfg.visualizeGoals:
            self.goal_markers.visualize(self.pos_command_w, self.default_quat)




@torch.jit.script
def logistic_kernel(x: torch.Tensor, std: float):
    a = 1 / std
    return (2.0) / (torch.exp(x * a) + torch.exp(-x * a))


    

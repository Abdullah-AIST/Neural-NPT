# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import CameraCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg, NoiseModelCfg
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
import BalanceReach_Real.tasks.direct.balancereach_noPose.mdp as mdp

from gymnasium import spaces

from .multi_object import MultiAssetCfg

import os
import math
import torch
from torch.distributions import Beta


def sample_beta(n: int, x_min:float, x_max:float, alpha=1, beta=1) -> torch.Tensor:
    """Sample from a beta distribution and scale to a given range."""
    # Sample from a beta distribution
    beta_dist = Beta(alpha, beta)
    samples = beta_dist.sample((n,1))
    # Scale the samples to the desired range
    scaled_samples = samples * (x_max - x_min) + x_min
    return scaled_samples



@configclass
class EventCfg:
    """Configuration for randomization."""

    


    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=0, # reset every episode
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("block"),
            "static_friction_range": (0.1, 0.1),      # 0.1, 1.0  #0.1 friction is safe for most objects but slower
            "dynamic_friction_range": (0.1, 0.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 500,
            "make_consistent": True,
        },
    )


    #object_abs_mass = EventTerm(
    #    func=mdp.randomize_rigid_body_mass,
    #    min_step_count_between_reset=0,
    #    mode="reset",
    #    params={
    #        "asset_cfg": SceneEntityCfg("block"),
    #        "mass_distribution_params": (0.75, 0.75), # 0.1, 2.5
    #        "operation": "abs",
    #        "distribution": "uniform",
    #        "recompute_inertia": True, 
    #    },
    #)
#

    #object_com = EventTerm(
    #    func=mdp.randomize_rigid_body_COM,
    #    mode="reset",                       #"startup","reset"
    #    params={
    #        "asset_cfg": SceneEntityCfg("block"),
    #    },
    #)




@configclass
class BalanceReachEnvCfg(DirectRLEnvCfg):
    RandCOM = False
    RandObjPos = False
    AccRate = 0.25              #0.5 seems to streak a balance
    smooth_acc = False      

    training = False

    evaluating = (not training)
    visualizeGoals = True 

    sim_Freq = 120
    ctrl_Freq = 20
    decimation = int(sim_Freq / ctrl_Freq)
    nTrajsPerEpisode = 1 # if not evaluating else 1
    TrajTime = 4.8 #if (not training or  RandCOM or RandObjPos) else 3.0
    episode_length_s = nTrajsPerEpisode* TrajTime #if not evaluating else 3 # 48*3/60=2.4
    action_space = 6#spaces.Box(-2, 2, shape=(6,))
    observation_space =  34 
    state_space =   64  + 3*RandCOM
    maxEp_steps = int(TrajTime*ctrl_Freq) # 48*3=144
    max_Steps = int(episode_length_s * ctrl_Freq)  # freq(120)/decimation(2)

    print("max_Steps: ", max_Steps)
    print("maxEp_steps: ", maxEp_steps)
    
    # num of instances
    n_training = 4096
    n_dataset = 512
    n_eval = 512

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, #1/120
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        use_fabric=True, #set to False for visual debugging
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=False)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="./assets/ur5eTray.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -math.pi / 4,
                "elbow_joint": math.pi / 2, 
                "wrist_1_joint": -math.pi / 4,
                "wrist_2_joint": math.pi / 2,
                "wrist_3_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=6000,
                damping=250,
               )
        },
    )

    stand = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Stand",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.4), rot=(1.0, 0.0, 0.0, 0.0)),
    )


        # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    goal_object_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=0.1, height=0.005, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.0))
            ),
        },
    )


    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/tray/camera",
        update_period=1/60,
        height= 240,
        width= 424,
        data_types=["rgb", "distance_to_image_plane","instance_id_segmentation_fast"],  # "instance_id_segmentation_fast"
        colorize_instance_id_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.0,
            horizontal_aperture=2.0059041707667853,
            vertical_aperture=1.1354174551510106,
            horizontal_aperture_offset=-0.02758591325410637,
            vertical_aperture_offset=0.003188630686549119,
            clipping_range=(0.007, 1e0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.375, -0.0, 0.148), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),

    )


    # eval states
    evalStates_path = "./scripts/Eval/eval_states/Evaluation_States.pt"
    #evalStates_path = "./scripts/Eval/eval_states/Evaluation_States_UnseenTall.pt"
    #evalStates_path = "./scripts/Eval/eval_states/Evaluation_States_real.pt"

    evalStates_loaded = False
    if os.path.exists(evalStates_path):
        evalStates_loaded = True
        evalStates = torch.load(evalStates_path)
        print("Loaded eval states from: ", evalStates_path)
        
    # block
    height_offset = 0.000

    assets_ratios = torch.empty((n_training + n_dataset, 3), device="cuda")

    if evaluating:
        assets_ratios = evalStates[:, 9:12]
        assets_colors = evalStates[:, 12:15]
        assets_frictions = evalStates[:, 15:]
    else:
        assets_ratios[:, 0] = sample_beta(n_training + n_dataset, 0.05, 0.6, alpha=1.5, beta=1.1).squeeze()     # encourage taller objects up to 0.6
        assets_ratios[:, 1] = sample_beta(n_training + n_dataset, 0.20,  3.0, alpha=1.1, beta=5.0).squeeze()     # encourage thinner objects up to 0.2 ratio (0.2*0.6 = 0.12)
        assets_ratios[:, 2] = sample_beta(n_training + n_dataset, 0.25, 4.0, alpha=1.0, beta=1.0).squeeze()       # Uniform distribution for the width/length ratio

        assets_colors = torch.rand((n_training + n_dataset, 3), device="cuda")
        assets_frictions = torch.rand((n_training + n_dataset, 1), device="cuda") * 0.9 + 0.1

        if training:
            assets_ratios = assets_ratios[:n_training]
            assets_colors = assets_colors[:n_training]
            assets_frictions = assets_frictions[:n_training]
        else:
            assets_ratios = assets_ratios[n_training:]
            assets_colors = assets_colors[n_training:]
            assets_frictions = assets_frictions[n_training:]

    assets_sizes = assets_ratios.clone()
    assets_sizes[:,2] = assets_ratios[:,0].clone()                                        # set height to the first ratio
    assets_sizes[:,0] = torch.clip(assets_ratios[:,1]*assets_ratios[:,0],max=0.15).clone()  # set width relative to height
    assets_sizes[:,1] = torch.clip(assets_ratios[:,2]*assets_ratios[:,1]*assets_ratios[:,0],max=0.15).clone()


    initJointPositions = evalStates[:,:6].tolist()
    targetPositions = evalStates[:,6:9].tolist()
    target = "woodBlock"
    
    YCB_tragets = {"dexCube": [0.1, 0.1, 0.1],
                    "crackerBox": [0.06, 0.158, 0.21], 
                    "chips": [0.075, 0.075, 0.250],
                    "mustardBottle": [0.058, 0.095, 0.190], 
                    "masterChefCan": [0.102, 0.102, 0.139],
                    "soupCan": [0.066, 0.066, 0.101],
                    "spam": [0.050, 0.097, 0.082],
                    "sugarBox": [0.038, 0.089, 0.175],
                    "tunaCan": [0.085, 0.085, 0.033],
                    "gelatin": [0.028, 0.085, 0.073],
                    "pudding": [0.035, 0.110, 0.089],
                    "apple": [0.075, 0.075, 0.075],
                    "pitcher": [0.108,0.108,0.235],
                    "bowl": [0.159, 0.159, 0.053],
                    "mug": [0.080, 0.080, 0.082],
                    "bleachCleanser": [0.065, 0.098, 0.250],
                    "powerDrill": [0.05, 0.15, 0.184],
                    "woodBlock": [0.085, 0.085, 0.20],
                    "customBlock": [0.15, 0.15, 0.56], #Tall
                    #"customBlock": [0.15, 0.05, 0.2],
                    #"customBlock": [0.158, 0.06, 0.21], # crakerBox_rotated
                    #"customBlock": [0.1, 0.1, 0.35], # pitcher
                    #"customBlock": [0.1, 0.1, 0.5], # dexCube
                    #"customBlock": [0.15, 0.15, 0.6],
                    #"customBlock": [0.065, 0.065, 0.13], # cup
                       }

    if target == "cylinder":
        assets_sizes[:,1] = assets_sizes[:,0]

    if target in YCB_tragets.keys():
        print("Evaluating YCB object: ", target)
        assets_sizes = torch.ones_like(assets_sizes)
        assets_sizes[:,0] = assets_sizes[:,0]*YCB_tragets[target][0]
        assets_sizes[:,1] = assets_sizes[:,1]*YCB_tragets[target][1]
        assets_sizes[:,2] = assets_sizes[:,2]*YCB_tragets[target][2]


    
    #change from torch to list
    assets_sizes = assets_sizes.tolist()
    assets_colors = assets_colors.tolist()
    assets_frictions = assets_frictions.tolist()

    if target == "block" or target == "customBlock":
        block = RigidObjectCfg(
            prim_path="/World/envs/env_.*/block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0], lin_vel=[0, 0, 0], ang_vel=[0, 0, 0]
            ),
            spawn=MultiAssetCfg(
                assets_cfg=[
                    sim_utils.CuboidCfg( #CuboidCfg, CylinderCfg, SphereCfg
                        #height = 1,
                        #radius=0.5,
                        size=(1, 1, 1),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(density=100),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            friction_combine_mode="multiply",
                            restitution_combine_mode="multiply",
                            static_friction=0.3,
                            dynamic_friction=0.3,
                            restitution=0.0,
                        ),
                    )
                ],
                assets_sizes=assets_sizes,
                assets_colors=assets_colors,
            ),
        )
    elif target == "cylinder":
        block = RigidObjectCfg(
            prim_path="/World/envs/env_.*/block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0], lin_vel=[0, 0, 0], ang_vel=[0, 0, 0]
            ),
            spawn=MultiAssetCfg(
                assets_cfg=[
                    sim_utils.CylinderCfg( #CuboidCfg, CylinderCfg, SphereCfg
                        height = 1,
                        radius=0.5,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(density=100),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            friction_combine_mode="multiply",
                            restitution_combine_mode="multiply",
                            static_friction=0.3,
                            dynamic_friction=0.3,
                            restitution=0.0,
                        ),
                    )
                ],
                assets_sizes=assets_sizes,
                assets_colors=assets_colors,
            ),
        )
    else:
        block = RigidObjectCfg(
            prim_path="/World/envs/env_.*/block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0], lin_vel=[0, 0, 0], ang_vel=[0, 0, 0]
            ),
            spawn=sim_utils.UsdFileCfg(usd_path=f"./assets/{target}2.usd",) 

        )
    

    


    action_scale = 20.0
    # reward scales
    # goal Pose
    ee_posError_rewScale = 1.0
    # regularization
    action_penalty_rewScale = 0.1
    joint_vel_rewScale = 0.1
    # block State
    block_posError_rewScale = 1.0
    block_oriError_rewScale = 1.0
    # survival/completion
    success_rewScale = 50.0
    live_reward = 1.0

    evalStates = []
    idx = []
    assets_ratios = []



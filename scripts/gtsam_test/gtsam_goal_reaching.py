# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Launch Isaac Sim Simulator first."""


import esther.launch as sim_lancher
simulation_app, args_cli, debug_drawer = sim_lancher.setup()
# start the simulator
import numpy as np
import bisect
import time
import csv
import torch
import math
import torch.nn.functional as F
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_angle_axis
from esther.env import EstherEnvCfg
from fgmp.controller import WheelOnlyController

from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

def create_start_goal_visualizer() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "start": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.5, 0.5, 0.5),
            ),
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.5, 0.5, 0.5),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

def create_waypoint_frames_visualizer() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            f"wp_{idx}": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ) 
         for idx in range(50)}
    )
    return VisualizationMarkers(marker_cfg)

class EstherController(WheelOnlyController):
    def __init__(self, r_wheel=0.320, l_wheel=0.690, th_wheel=22/180*torch.pi, N = 50):
        super().__init__(N)
        self.r_wheel = r_wheel
        self.l_wheel = l_wheel
        self.th_wheel = th_wheel

    def cmd2wheel(self, cmd):
        '''
        cmd: [vx, vtheta]
        return: [vleft, vright]
        '''
        v, w = cmd
        w_l = (v - w *self.l_wheel/2 - w * self.r_wheel*math.sin(self.th_wheel)) / self.r_wheel
        w_r = (v + w *self.l_wheel/2 + w * self.r_wheel*math.sin(self.th_wheel)) / self.r_wheel

        # # clip 
        # norm = torch.sqrt(w_l**2 + w_r**2)
        # if norm > self.max_speed_wheel:
        #     w_l = w_l / norm * self.max_speed_wheel
        #     w_r = w_r / norm * self.max_speed_wheel

        return w_l, w_r
    
    def wheel2cmd(self, wheel_vel):
        '''
        convert the left and right wheel angular velocity to the linear velocity and angular velocity
        '''
        w_l, w_r = wheel_vel
        v = (w_l + w_r) / 2 * self.r_wheel 
        w = (w_r - w_l) / (self.l_wheel + 2*self.r_wheel*math.sin(self.th_wheel)) * self.r_wheel
        return v, w


def _wait_awhile(env, action, duration= 1.0):
    num_steps = int(duration / env.step_dt)
    for _ in range(num_steps):
        env.step(action)
    

def main():
    """Main function."""
    # parse the arguments
    env_cfg = EstherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    curr_time = 0.0
    action = torch.zeros_like(env.action_manager.action, device=env.device)
    controller = EstherController()
    start_goal_visualizer = create_start_goal_visualizer()
    waypoint_frames_visualizer = create_waypoint_frames_visualizer()



    _wait_awhile(env, action, duration=1.0)

    result = None
    goal_reaching_duration = 20.0
    while simulation_app.is_running():
        # wait for a while to stable the initial dynamics

        print('curr_time: ', curr_time)
        print('curr_action: ', action[0,:2].cpu().numpy())
        obs, _ = env.step(action)
        curr_time += env.step_dt

        # get the current states
        odom = obs["policy"]['odom'][:, :7]      # [x, y, z, quat] per environment
        wheel_vel = obs["policy"]['vel'][:, -2:]  # left, right wheel velocities
        arm_vel = obs["policy"]['vel'][:, 2:]
        arm_pos = obs["policy"]['pos']

        print('curr_obs: ', wheel_vel[0].cpu().numpy())
        # control

        start_xy = odom[:, :2].cpu().numpy()
        start_th = euler_xyz_from_quat(odom[:, 3:7])[2].cpu().numpy()
        start_th = (start_th + np.pi)% (2 * np.pi) - np.pi
        start = np.array([start_xy[0,0], start_xy[0,1], start_th[0]])

        v_start = controller.wheel2cmd(wheel_vel[0].cpu().numpy())
        goal = np.array([5.0, 5.0, 0])
        v_goal = np.array([0.0, 0.0])
        
        print('start: ', start)
        print('v_start: ', v_start)
        print('goal: ', goal)
        print('v_goal: ', v_goal)
        print('goal_reaching_duration: ', goal_reaching_duration - curr_time)

        
        result = controller.inference(start, v_start, goal, v_goal, goal_reaching_duration - curr_time , initial_estimate=None)
        velocities = controller.get_velocity(result)
        graph_times = np.linspace(0, goal_reaching_duration - curr_time, controller.N)
        next_velocity_id = 0
        wl, wr = controller.cmd2wheel(velocities[next_velocity_id])


        # test_cmd = [2.4, np.pi/6]
        # wl, wr = controller.cmd2wheel(test_cmd)
        # # wl, wr = 10,5
        action[:,0:2] = torch.tensor([[wl, wr]], device=env.device)
        # # test_cmd = controller.wheel2cmd([wl, wr])
        # print('wl, wr: ', wl, wr)
        # print('set cmd: ', test_cmd)
        # print('obs v: ', torch.linalg.norm(obs["policy"]['odom'][0, 7:10]))
        # print('obs w: ', obs["policy"]['odom'][0, 10:13])


        # debug path
        if not args_cli.headless:
            # path
            debug_drawer.clear_lines()
            poses = controller.get_pose(result)
            point_list_1 = [(pose[0], pose[1], 0.0) for pose in poses[:-1]]
            point_list_2 = [(pose[0], pose[1], 0.0) for pose in poses[1:]]
            colors = [(0,1,0,1) for _ in range(len(poses)-1)]
            sizes = [10 for _ in range(len(poses)-1)]
            debug_drawer.draw_lines(point_list_1, point_list_2, colors, sizes)

            # start and goal
            yaw_start  = start[2]
            yaw_goal = goal[2]
            orientations = quat_from_angle_axis(torch.tensor([yaw_start, yaw_goal]), torch.tensor([0.0, 0.0, 1.0]))
            locations = torch.tensor([[start[0], start[1], 0.0], [goal[0], goal[1], 0.0]])
            start_goal_visualizer.visualize(locations, orientations, marker_indices=torch.tensor([0, 1]))

            # # waypoint frames
            # yaws = torch.tensor([pose[2] for pose in poses])
            # orientations = quat_from_angle_axis(yaws, torch.tensor([0.0, 0.0, 1.0]))
            # locations = torch.tensor([[pose[0], pose[1], 0.0] for pose in poses])
            # waypoint_frames_visualizer.visualize(locations, orientations, marker_indices=torch.arange(len(poses)))
            
            # time.sleep(1)

    env.close()
    simulation_app.close()
 

if __name__ == "__main__":
    main()
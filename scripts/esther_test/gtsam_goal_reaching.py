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
import time
import csv
import torch
import math
import torch.nn.functional as F
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.utils.math import euler_xyz_from_quat
from esther.env import EstherEnvCfg

from fgmp.controller import WheelOnlyController


class EstherController(WheelOnlyController):
    def __init__(self, r_wheel=0.320, l_wheel=0.690, th_wheel=20/180*torch.pi, N = 50):
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



    _wait_awhile(env, action, duration=1.0)

    result = None
    duration = 5.0
    while simulation_app.is_running():
        # wait for a while to stable the initial dynamics
        # print(action)
        obs, _ = env.step(action)
        curr_time += env.step_dt

        # get the current states
        odom = obs["policy"]['odom'][:, :7]      # [x, y, z, quat] per environment
        wheel_vel = obs["policy"]['vel'][:, :2]  # left, right wheel velocities
        arm_vel = obs["policy"]['vel'][:, 2:]
        arm_pos = obs["policy"]['pos']

        # control

        start_xy = odom[:, :2].cpu().numpy()
        start_th = euler_xyz_from_quat(odom[:, 3:7])[2].cpu().numpy()
        start = np.array([start_xy[0,0], start_xy[0,1], start_th[0]])

        v_start = controller.wheel2cmd(wheel_vel[0].cpu().numpy())
        goal = np.array([5.0, 5.0, 0])
        v_goal = np.array([0.0, 0.0])
        

        result = controller.inference(start, v_start, goal, v_goal, duration - curr_time, initial_estimate=result)

        velocities = controller.get_velocity(result)
        wl, wr = controller.cmd2wheel(velocities[1])
        
        action[:,0:2] = torch.tensor([[wl, wr]], device=env.device)

        # debug path
        if not args_cli.headless:
            debug_drawer.clear_lines()

            poses = controller.get_pose(result)
            point_list_1 = [(pose[0], pose[1], 0.0) for pose in poses[:-1]]
            point_list_2 = [(pose[0], pose[1], 0.0) for pose in poses[1:]]
            colors = [(0,1,0,1) for _ in range(len(poses)-1)]
            sizes = [10 for _ in range(len(poses)-1)]

            debug_drawer.draw_lines(point_list_1, point_list_2, colors, sizes)
                
        
        

       



    env.close()
    simulation_app.close()
 

if __name__ == "__main__":
    main()
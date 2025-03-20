# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# start the simulator
import time
import csv
import torch
import math
import torch.nn.functional as F
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.utils.math import euler_xyz_from_quat
from esther.env import EstherEnvCfg
print('---------------------------------',args_cli.headless)

if not args_cli.headless:
    from omni.isaac.debug_draw import _debug_draw
    debug_drawer = _debug_draw.acquire_debug_draw_interface()

def prescribed_path():
    R = 5.0 # radius, m
    T = 10.0 # total time, sec
        
    t = torch.linspace(0, T, 100)
    x = R * torch.sin(2 * torch.pi * t / T)
    y = R - R  *  torch.cos(2 * torch.pi * t / T)

    return torch.stack([t, x, y], dim=1)

def get_next_waypoint(stamped_path: torch.Tensor, t:float):
    
    # Find the indices of the two nearest points in time
    idx = torch.searchsorted(stamped_path[:, 0], t)
    
    # Handle edge cases where t is outside the path's time range
    if idx == 0:
        return stamped_path[0, 1:]  # Return the first waypoint
    if idx == len(stamped_path):
        return stamped_path[-1, 1:]  # Return the last waypoint
    
    # Get the two nearest points
    t0, x0, y0 = stamped_path[idx - 1]
    t1, x1, y1 = stamped_path[idx]
    
    # Perform linear interpolation
    alpha = (t - t0) / (t1 - t0)
    x = x0 + alpha * (x1 - x0)
    y = y0 + alpha * (y1 - y0)
    
    return torch.tensor([x, y])

def wait_awhile(env, action, duration= 3.0):
    num_steps = int(duration / env.step_dt)
    for _ in range(num_steps):
        env.step(action)

class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, error_fn = None):
        '''
        error_fn use to calculate the error given the current state and the desired state

        if not defined, the error is defined as the euclidean distance between the two states
        '''
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.prev_error = 0.0

        self.error_fn = lambda x, y: y-x if error_fn is None else error_fn(x, y)

    
    def compute_action(self, curr_states, des_states, dt):
        
        error = self.error_fn(curr_states, des_states)

        proportional = self.Kp * error
        self.integral += error * dt
        integral = self.Ki * self.integral
        
        derivative = self.Kd * (error - self.prev_error) / dt
        
        control_action = proportional + integral + derivative
        self.prev_error = error
        
        return control_action


    
class DiffController:
    def __init__(self, r_wheel=0.320, l_wheel=0.690, th_wheel=20/180*torch.pi, max_speed_wheel=20):
        '''
        params:
            r_wheel: radius of the wheel
            l_wheel: distance between the wheels
            th_wheel: angle of the wheel
        '''
        self.r_wheel = r_wheel
        self.l_wheel = l_wheel
        self.th_wheel = th_wheel
        self.max_speed_wheel = max_speed_wheel # rad/s

        self.v_controller = PIDController(Kp=3.0, Ki=0.1, Kd=0.001, error_fn = None)
        self.w_controller = PIDController(Kp=30.0, Ki=0.2, Kd=0.001, error_fn = None)

    def cmd_vel_to_wheel_vel(self, v, w):
        '''
        convert the linear velocity and angular velocity to the left and right wheel angular velocity
        '''
        w_l = (v - w *self.l_wheel/2 - w * self.r_wheel*math.sin(self.th_wheel)) / self.r_wheel
        w_r = (v + w *self.l_wheel/2 + w * self.r_wheel*math.sin(self.th_wheel)) / self.r_wheel

        # clip 
        norm = torch.sqrt(torch.tensor(w_l**2 + w_r**2))
        if norm > self.max_speed_wheel:
            w_l = w_l / norm * self.max_speed_wheel
            w_r = w_r / norm * self.max_speed_wheel

        return w_l, w_r
    
    def wheel_vel_to_cmd_vel(self, w_l, w_r):
        '''
        convert the left and right wheel angular velocity to the linear velocity and angular velocity
        '''
        v = (w_l + w_r) / 2 * self.r_wheel 
        w = (w_r - w_l) / (self.l_wheel + 2*self.r_wheel*math.sin(self.th_wheel)) * self.r_wheel

        return v, w
    

    def compute_action(self, odom, next_waypoint, dt):
        xy_curr = odom[0,0:2]
        theta_curr = euler_xyz_from_quat(odom[:, 3:7])[2]  
        xy_desired = next_waypoint[0]
        theta_desired = torch.atan2(xy_desired[1] - xy_curr[1], xy_desired[0] - xy_curr[0])

        print('xy_curr:', xy_curr)
        print('xy_desired:', xy_desired)
        print('theta_curr:', theta_curr)
        print('theta_desired:', theta_desired)

        def angle_diff(a, b):
            return (a - b + torch.pi) % (2 * torch.pi) - torch.pi # [-pi, pi]
            


        err_angle = angle_diff(theta_desired, theta_curr)
        if abs(err_angle) > math.pi/2:
            err_pos = -torch.linalg.norm(xy_desired - xy_curr)
        else:
            err_pos = torch.linalg.norm(xy_desired - xy_curr)

        print('err_pos:', err_pos)
        print('err_angle:', err_angle)

        cmd_v = self.v_controller.compute_action(0, err_pos, dt)
        cmd_w = self.w_controller.compute_action(0, angle_diff(theta_desired, theta_curr), dt)

        print('cmd_v:', cmd_v)
        print('cmd_w:', cmd_w)

        w_l, w_r = self.cmd_vel_to_wheel_vel(cmd_v, cmd_w)

        # clip
        w_l = torch.clip(w_l, -self.max_speed_wheel, self.max_speed_wheel)
        w_r = torch.clip(w_r, -self.max_speed_wheel, self.max_speed_wheel)

        return w_l, w_r

    

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
    stamped_path = prescribed_path()


    # draw prescribed path
    if not args_cli.headless:
        to_draw = torch.stack([stamped_path[:,1], stamped_path[:,2], torch.ones(len(stamped_path))*0.05], dim=1)
        debug_drawer.draw_lines_spline(to_draw.tolist(), (0,1,0,1), 5, False)

    # init controller
    controller = DiffController()

    wait_awhile(env, action, duration=3.0)

    # waypoint_csv = open('waypoint.csv', 'a', newline="")
    # waypoint_writer = csv.writer(waypoint_csv)
    # odom_csv = open('odom.csv', 'a', newline="")
    # odom_writer = csv.writer(odom_csv)

    action[:,0:2] = torch.tensor([10,4], device=env.device)
    while simulation_app.is_running():
        # wait for a while to stable the initial dynamics

        with torch.inference_mode():
            obs, _ = env.step(action)
            curr_time += env.step_dt



            # get the current states
            odom = obs["policy"]['odom'][:, :7]      # [x, y, z, quat] per environment
            wheel_vel = obs["policy"]['vel'][:, :2]  # left, right wheel velocities
            arm_vel = obs["policy"]['vel'][:, 2:]
            arm_pos = obs["policy"]['pos']

            # get the next waypoint (shape [num_envs, 3] or at least [num_envs, 2])
            next_waypoint = get_next_waypoint(stamped_path, curr_time)[None, ...].to(env.device)

            # compute the control action
            w_l, w_r = controller.compute_action(odom, next_waypoint, env.step_dt)
            # action[:,0:2] = torch.stack([w_l, w_r], dim=1)

            set_wheel  = [10, 4]
            action[:,0:2] = torch.tensor(set_wheel, device=env.device)

            # draw the next waypoint arg0: List[carb._carb.Float3], arg1: List[carb._carb.ColorRgba], arg2: List[float]
            # if args_cli contains --headless
            if not args_cli.headless:
                points = next_waypoint.cpu().tolist()
                points[0].append(0.06)
                colors = [[1,0,0,1]]
                size = [10.0]
                debug_drawer.draw_points(points, colors, size)
           

            # odom_writer.writerow(odom[0,:].cpu().tolist())
            # waypoint_writer.writerow(get_next_waypoint(stamped_path, curr_time - env.step_dt).cpu().tolist())

            print('odom:', odom)
            print('next waypoint:', next_waypoint)
            # print('wheel vel:', wheel_vel)
            v, w = controller.wheel_vel_to_cmd_vel(*set_wheel)
            w_l, w_r = controller.cmd_vel_to_wheel_vel(v, w)
            print('root vel:', torch.linalg.norm(obs["policy"]['odom'][0, 7:10]))
            print('kine vel:', v)
            print('root ang vel:', obs["policy"]['odom'][0, 10:13])
            print('kine ang vel:', w)
            print('wheel vel:', [w_l, w_r])
            # time.sleep(0.1)



    env.close()
    simulation_app.close()
    # waypoint_csv.close()
    # odom_csv.close()

if __name__ == "__main__":
    main()
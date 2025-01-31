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
import torch
import torch.nn.functional as F

from omni.isaac.lab.envs import ManagerBasedEnv
from esther.env import EstherEnvCfg

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


def main():
    """Main function."""
    # parse the arguments
    env_cfg = EstherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    curr_time = 0.0
    actions = torch.zeros_like(env.action_manager.action, device=env.device)
    stamped_path = prescribed_path()

    # For a true PD controller, track previous error (same shape as error).
    prev_error_xy = None

    # Example gains (tune to your scenario)
    Kp_dist = 1.0       # Proportional gain for distance
    Kp_head = 0      # Proportional gain for heading
    Kd_dist = 0.1       # Derivative gain for distance
    Kd_head = 0     # Derivative gain for heading

    # Baseline (distance) between the two wheels, needed to convert (v, w) -> (v_left, v_right)
    # If you have a known track width or half-baseline, adjust accordingly.
    baseline = 0.4

    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _ = env.step(actions)
            curr_time += env.step_dt

            # get the current states
            odom = obs["policy"]['odom'][:, :3]      # [x, y, theta] per environment
            wheel_vel = obs["policy"]['vel'][:, :2]  # left, right wheel velocities
            arm_vel = obs["policy"]['vel'][:, 2:]
            arm_pos = obs["policy"]['pos']

            # get the next waypoint (shape [num_envs, 3] or at least [num_envs, 2])
            next_waypoint = get_next_waypoint(stamped_path, curr_time)[None, ...].to(env.device)

            # calculate the position/heading error in the XY plane
            # ----------------------------------------------------
            # Position error in x and y (ignoring z for differential drive)
            print(f"Next waypoint: {next_waypoint}")
            print(f"Odom: {odom}")
            error_xy = next_waypoint[:, 0:2] - odom[:, 0:2]
            distance_error = torch.norm(error_xy, dim=-1)  # scalar distance to waypoint

            # Current heading (robot orientation) assumed to be odom[:, 2]
            current_heading = odom[:, 2]

            # Desired heading is the angle from current (x,y) to the waypoint
            desired_heading = torch.atan2(error_xy[:, 1], error_xy[:, 0])

            # Heading error: difference between desired_heading and current_heading
            heading_error = desired_heading - current_heading

            # Wrap heading error to [-pi, pi]
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

  

            # ----------------------------------------------------
            # Optional: Derivative term (PD), if desired
            # We only do derivative in the distance error & heading error.
            # If prev_error_xy is None, we have no derivative info yet.
            # ----------------------------------------------------
            dist_deriv = torch.zeros_like(distance_error)
            head_deriv = torch.zeros_like(heading_error)

            if prev_error_xy is not None:
                # previous distance error
                prev_distance_error = torch.norm(prev_error_xy, dim=-1)

                # distance derivative
                dist_deriv = (distance_error - prev_distance_error) / env.step_dt

                # heading derivative
                # We do the same trick for heading (but watch out for wrap-around).
                # A quick way is just to re-compute heading errors from the prev vector:
                # However, you might also have stored prev_heading_error explicitly.
                # Here, we assume same approach:
                #   heading_error(t) = desired_heading - current_heading
                #   heading_error(t-1) = desired_heading_prev - current_heading_prev
                # For simplicity, let's define heading derivative as:
                head_deriv = (heading_error - prev_heading_error) / env.step_dt

            # Store current for next iteration
            prev_error_xy = error_xy.clone()
            prev_heading_error = heading_error.clone()

            # ----------------------------------------------------
            # PD control
            # ----------------------------------------------------
            # Linear velocity command
            v_cmd = (Kp_dist * distance_error) + (Kd_dist * dist_deriv)
            # Angular velocity command
            w_cmd = (Kp_head * heading_error) + (Kd_head * head_deriv)

            # Convert (v, w) into left/right wheel velocities
            # v_left  = v - (w * baseline / 2)
            # v_right = v + (w * baseline / 2)
            # For vector operations, shape is [num_envs]
            left_vel = v_cmd - 0.5 * baseline * w_cmd
            right_vel = v_cmd + 0.5 * baseline * w_cmd

            # Now place these into the actions tensor
            # actions shape = [num_envs, 2] => (left_wheel, right_wheel)
            actions[:, 0] = left_vel
            actions[:, 1] = right_vel

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
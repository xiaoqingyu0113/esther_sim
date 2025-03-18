import gtsam
from gtsam.symbol_shorthand import X, U
from typing import List, Optional
import numpy as np


# factors for gtsam
def assign_jacobians(jacobians: List[np.ndarray],J: List[np.ndarray]):
     for i, JJ in enumerate(J):
          jacobians[i] = JJ

def compute_pose(x1, u1, t):
    """
    Compute the predicted pose after applying controls.
    x1: [x, y, theta] (initial pose)
    u1: [v_x, theta_dot] (control inputs)
    t: time duration
    Returns: [x, y, theta] (predicted pose)
    """
    x, y, theta = x1
    v_x, theta_dot = u1
    
    if abs(theta_dot) > 1e-6:  # Arc motion
        R = v_x / theta_dot
        delta_theta = theta_dot * t
        x_new = x - R * np.sin(theta) + R * np.sin(theta + delta_theta)
        y_new = y + R * np.cos(theta) - R * np.cos(theta + delta_theta)
        theta_new = theta + delta_theta
    else:  # Straight-line motion
        x_new = x + v_x * np.cos(theta) * t
        y_new = y + v_x * np.sin(theta) * t
        theta_new = theta
    # Normalize theta to [-pi, pi]
    # theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_new, y_new, theta_new])

def Jerr_compute_pose(x1, u1, x2, t):
    """
    Compute the Jacobians of the error function with respect to x1, u1, and x2.
    x1: [x, y, theta] (initial pose)
    u1: [v_x, theta_dot] (control inputs)
    x2: [x, y, theta] (final pose)
    t: time duration
    Returns: Jacobians (J_x1, J_u1, J_x2)
    """
    x, y, theta = x1
    v_x, theta_dot = u1
    
    if abs(theta_dot) > 1e-6:  # Arc motion
        R = v_x / theta_dot
        delta_theta = theta_dot * t
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta_dt = np.sin(theta + delta_theta)
        cos_theta_dt = np.cos(theta + delta_theta)
        
        # Jacobian w.r.t x1
        J_x1 = np.array([
            [1, 0, -R * (cos_theta - cos_theta_dt)],
            [0, 1, -R * (sin_theta_dt - sin_theta)],
            [0, 0, 1]
        ])
        
        # Jacobian w.r.t u1
        dR_dv_x = 1 / theta_dot
        dR_dtheta_dot = -v_x / (theta_dot ** 2)
        J_u1 = np.array([
            [dR_dv_x * (-sin_theta + sin_theta_dt), dR_dtheta_dot * (-sin_theta + sin_theta_dt) + R * t * cos_theta_dt],
            [dR_dv_x * (cos_theta - cos_theta_dt), dR_dtheta_dot * (cos_theta - cos_theta_dt) - R * t * sin_theta_dt],
            [0, t]
        ])
        
    else:  # Straight-line motion
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Jacobian w.r.t x1
        J_x1 = np.array([
            [1, 0, -v_x * t * sin_theta],
            [0, 1, v_x * t * cos_theta],
            [0, 0, 1]
        ])
        
        # Jacobian w.r.t u1
        J_u1 = np.array([
            [t * cos_theta, 0],
            [t * sin_theta, 0],
            [0, t]
        ])
    
    # Jacobian w.r.t x2
    J_x2 = np.eye(3)
    
    return -J_x1, -J_u1, J_x2

class DynFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel,x1_key, u1_key, x2_key, t1, t2):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            x1, u1, x2 =  values.atVector(x1_key), values.atVector(u1_key), values.atVector(x2_key)
            error = x2 - compute_pose(x1, u1, t2 - t1)
            if jacobians is not None:
                assign_jacobians(jacobians,Jerr_compute_pose(x1, u1, x2, t2 - t1))
            return error
        super().__init__(noiseModel, [x1_key, u1_key, x2_key], error_function) # may change to partial


class PriorFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, x_key, x_val):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            x = values.atVector(x_key)
            error = x - x_val
            if jacobians is not None:
                jacobians[0] = np.eye(3)
            return error
        super().__init__(noiseModel, [x_key], error_function) # may change to partial

class VPriorFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, x_key, x_val):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            x = values.atVector(x_key)
            error = x - x_val
            if jacobians is not None:
                jacobians[0] = np.eye(2)
            return error
        super().__init__(noiseModel, [x_key], error_function) # may change to partial

class VBetweenFactor(gtsam.CustomFactor):
    def __init__(self, noiseModel, x_key, y_key):
        def error_function(self,values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> float:
            x, y = values.atVector(x_key), values.atVector(y_key)
            error = y-x
            if jacobians is not None:
                jacobians[0] = -np.eye(2)
                jacobians[1] = np.eye(2)
            return error
        super().__init__(noiseModel, [x_key,y_key], error_function) # may change to partial
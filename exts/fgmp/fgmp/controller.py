import numpy as np
import gtsam
from gtsam.symbol_shorthand import  V, X
from fgmp.factors import PriorFactor, DynFactor, VPriorFactor, VBetweenFactor


class WheelOnlyController:
    def __init__(self, N = 50):
        self.N = N
        self.dyn_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1])*5e-4)
        self.v_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1])*1e5)
        self.v_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1])*1e1)
        self.x_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*5e-4)
        self.constraint_noise3 = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-4)
        self.constraint_noise2 = gtsam.noiseModel.Diagonal.Sigmas(np.ones(2)*1e-4)

    def cmd2wheel(self, cmd):
        '''
        cmd: [vx, vtheta]
        return: [vleft, vright]
        '''
        raise NotImplementedError    
    
    def wheel2cmd(self, wheel_vel):
        '''
        convert the left and right wheel angular velocity to the linear velocity and angular velocity
        '''
        raise NotImplementedError
    
    def get_velocity(self, result: gtsam.Values) -> np.ndarray:
        '''
        get the velocity from the result
        '''
        v = [result.atVector(V(i)) for i in range(self.N)]
        return np.array(v, dtype=float)

    def get_pose(self, result: gtsam.Values) -> np.ndarray:
        '''
        get the pose from the result
        '''
        x = [result.atVector(X(i)) for i in range(self.N)]
        return np.array(x, dtype=float)    

    def inference(self, start, v_start, goal, v_goal, duration, initial_estimate=None):
        '''
        start: [x, y, theta]
        v_start: [vx, theta_dot]
        goal: [x, y, theta]
        v_goal: [vx, theta_dot]
        duration: time spent to reach the goal
        initial_estimate: may reuse previous result
        '''
        N =  50
        t = np.linspace(0, duration, N)

        graph = gtsam.NonlinearFactorGraph()
        params = gtsam.LevenbergMarquardtParams()

        ## build factor graph
        dt = duration / (N-1)
        for i in range(N-1):
            graph.push_back(DynFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1])*5e-3), 
                                    X(i), 
                                    V(i), 
                                    X(i+1), 0, dt))
            
            graph.push_back(VPriorFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1])*5e0), 
                                        V(i), 
                                        np.array([0.0, 0.0])))
            
            # graph.push_back(PriorFactor( gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1])*5e2), 
            #                             X(i), 
            #                             (goal-start)/(N-1)*i + start))
            if i < N-2:
                graph.push_back(VBetweenFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1])*5e1),
                                                V(i),
                                                V(i+1)))

        graph.push_back(PriorFactor(gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1])*5e-4), 
                                    X(0), start))
        graph.push_back(PriorFactor( gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1])*1e-4), 
                                    X(N-1), goal))
        graph.push_back(VPriorFactor( gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1])*5e-4), 
                                    V(0), v_start))
        graph.push_back(VPriorFactor(gtsam.noiseModel.Diagonal.Sigmas(np.ones(2)*5e-4),
                                    V(N-1), v_goal))


        # add estimate
        initial_estimate = gtsam.Values()
        initial_estimate.insert(X(0), start)
        initial_estimate.insert(X(N-1), goal)
        initial_estimate.insert(V(0), v_start)
        initial_estimate.insert(V(N-1), v_goal)

        for i in range(1, N-1):
            initial_estimate.insert(X(i), (goal-start)/(N-1)*i + start)
            initial_estimate.insert(V(i), np.array([1.0, 0]))

        
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        # optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()

        return result
import gtsam
from gtsam.symbol_shorthand import V, X
import numpy as np
from fgmp.factors import PriorFactor, DynFactor, VPriorFactor, VBetweenFactor, compute_pose
import matplotlib.pyplot as plt
def gtsam_naive_path(start , v_start, goal, v_goal, duration, N):
    # N = 50
    # duration = 5

    # start = np.array([0, 0, 0])
    # goal = np.array([5, 5, 0])
    # v_start = np.array([0, 0])
    # v_goal = np.array([0, 0])
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
        
        # graph.push_back(PriorFactor( gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1])*5e1), 
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
        initial_estimate.insert(V(i), np.array([0, 0]))

    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    # optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    poses = np.array([result.atVector(X(i)) for i in range(N)], dtype=float)
    x = poses[:, 0]
    y = poses[:, 1]
    th = poses[:, 2]

    v = np.array([result.atVector(V(i)) for i in range(N)], dtype=float)

    error = np.array([poses[i+1] - compute_pose(poses[i], v[i+1], dt) for i in range(N-1)], dtype=float)
    print('error:', error)

    print(f"x end error = {goal} - {poses[-1]} = {goal - poses[-1]}")

    print("v = ", v)
    return np.stack((t, x, y, th)).T

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)

    start = np.array([0, 0, 0])
    v_start = np.array([0, 0])
    goal = np.array([5, 5, -np.pi])  
    v_goal = np.array([0, 0])
    stamped_path = gtsam_naive_path(start, v_start, goal, v_goal, 5.0, 50)


    print(stamped_path.shape)
    ax.plot(stamped_path[:,1], stamped_path[:,2] ,label='trajectory')
    ax.plot(start[0], start[1], 'ro', label='start')
    ax.plot(goal[0], goal[1], 'go', label='goal')
    ax.quiver(stamped_path[:,1], stamped_path[:,2], np.cos(stamped_path[:,3]), np.sin(stamped_path[:,3]), scale=40, color='r', width=0.005)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('trajectory')


    plt.show()
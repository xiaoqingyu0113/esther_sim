import numpy as np
from fgmp.controller import WheelOnlyController
import matplotlib.pyplot as plt



class EstherController(WheelOnlyController):
    def __init__(self, r_wheel=0.320, l_wheel=0.690, th_wheel=20/180*np.pi, N = 50):
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
        w_l = (v - w *self.l_wheel/2 - w * self.r_wheel*np.sin(self.th_wheel)) / self.r_wheel
        w_r = (v + w *self.l_wheel/2 + w * self.r_wheel*np.sin(self.th_wheel)) / self.r_wheel

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
        w = (w_r - w_l) / (self.l_wheel + 2*self.r_wheel*np.sin(self.th_wheel)) * self.r_wheel
        return v, w


controller = EstherController()

# start = np.array([1.11403875e-04 ,1.84215605e-06, 1.40647126e-05])
start = np.array([-0.0158763 ,  0.00099092 ,-0.04902387])
v_start = np.array([0.02187854766845703, 1.1024384196366346])
# v_start = np.array([0, 0])
goal = np.array([5, 5, 0])
v_goal = np.array([0, 0])
goal_reaching_duration = 10.0

fig = plt.figure()
ax = fig.add_subplot(111)

result = None
result = controller.inference(start, v_start, goal, v_goal, goal_reaching_duration, initial_estimate=result)

# plot the result


poses = controller.get_pose(result)
th = poses[:, 2]

print(th)
ax.plot(poses[:, 0], poses[:, 1], label='trajectory')
ax.plot(start[0], start[1], 'ro', label='start')
ax.plot(goal[0], goal[1], 'go', label='goal')
ax.quiver(poses[:, 0], poses[:, 1], np.cos(th), np.sin(th), scale=40, color='r', width=0.005)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('trajectory')


plt.show()
import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax):
    extents = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_range = np.ptp(extents, axis=1).max() / 2
    for ctr, axis in zip(centers, 'xyz'):
        getattr(ax, f'set_{axis}lim')(ctr - max_range, ctr + max_range)

def draw_tennis_court(ax):
    """
    Draws a 3D tennis court (lines only) on the provided axes.
    
    Coordinate system:
      - Origin (0,0,0) is the center of the baseline.
      - x-axis points into the court.
      - y-axis runs along the baseline.
      
    Court dimensions (in feet):
      - Overall court length: 78 ft.
      - Doubles court width: 36 ft (i.e. y = ±18).
      - Singles court width: 27 ft (i.e. y = ±13.5).
      - Net at x = 39 ft.
      - Service lines: 21 ft from the net (i.e. at x = 18 and x = 60).
      - Baseline center mark (on the near baseline at x = 0): 4 inches (≈0.333 ft) long.
    """
    # --- Define key dimensions (in feet) ---
    baseline_near = 0       # near baseline (at the origin)
    baseline_far  = 78      # far baseline (opposite end)
    net_x         = 39      # x coordinate of the net
    service_near  = 18      # service line on the server's (near) side
    service_far   = 60      # service line on the opponent's (far) side
    
    doubles_half_width = 18    # half-width for doubles (36 ft overall)
    singles_half_width = 13.5  # half-width for singles (27 ft overall)
    
    center_mark_length = 0.333  # 4 inches ~0.333 ft total length on the baseline
    center_mark_half   = center_mark_length / 2
    
    # --- Outer lines (doubles court) ---
    # Near baseline (vertical line at x = 0 from y = -18 to 18)
    # ax.plot([baseline_near, baseline_near],
    #         [-doubles_half_width, doubles_half_width],
    #         [0, 0], color='k', lw=2)
    
    # Far baseline (x = 78)
    # ax.plot([baseline_far, baseline_far],
    #         [-doubles_half_width, doubles_half_width],
    #         [0, 0], color='k', lw=2)
    
    # Left doubles sideline (y = -18)
    ax.plot([baseline_near, baseline_far],
            [-doubles_half_width, -doubles_half_width],
            [0, 0], color='k', lw=2)
    
    # Right doubles sideline (y = 18)
    ax.plot([baseline_near, baseline_far],
            [doubles_half_width, doubles_half_width],
            [0, 0], color='k', lw=2)
    
    # Net (vertical line at x = 39, dashed)
    ax.plot([net_x, net_x],
            [-doubles_half_width, doubles_half_width],
            [0, 0], color='k', lw=2, linestyle='--')
    
    # --- Inner lines (singles court lines and service lines) ---
    # Left singles sideline (y = -13.5)
    ax.plot([baseline_near, baseline_far],
            [-singles_half_width, -singles_half_width],
            [0, 0], color='k', lw=2)
    
    # Right singles sideline (y = 13.5)
    ax.plot([baseline_near, baseline_far],
            [singles_half_width, singles_half_width],
            [0, 0], color='k', lw=2)
    
    # Service line on the server's side (x = 18)
    ax.plot([service_near, service_near],
            [-singles_half_width, singles_half_width],
            [0, 0], color='k', lw=2)
    
    # Service line on the opponent's side (x = 60)
    ax.plot([service_far, service_far],
            [-singles_half_width, singles_half_width],
            [0, 0], color='k', lw=2)
    
    # Center service line in the server's half (from x = 18 to net, at y = 0)
    ax.plot([service_near, net_x],
            [0, 0],
            [0, 0], color='k', lw=2)
    
    # Center service line in the opponent's half (from net to x = 60, at y = 0)
    ax.plot([net_x, service_far],
            [0, 0],
            [0, 0], color='k', lw=2)
    
    # Center mark on the near baseline (short line at x = 0, centered at y = 0)
    ax.plot([baseline_near, baseline_near],
            [-center_mark_half, center_mark_half],
            [0, 0], color='k', lw=2)


class EstherRobot:
    '''
    This is a wheelchair robot with two-wheeled differential drive.
    The robot has a wheelbase of b, a radius of r and an arm length of arm.
    The robot arm sitted on the wheelchair, and has 3 dof.
    '''
    def __init__(self, b=2.0, r=1.0, arm=(1.5, 1.5), racket_r=0.270):
        self.b = b  # Wheelbase
        self.r = r  # Wheel radius
        self.arm = arm  # Arm lengths (assuming 2 segments for simplicity)
        self.racket_r = racket_r  # Racket radius

    def _get_initial_pose(self, ax: plt.Axes):
        '''
        compute initial pose of the robot, return 
        '''
        # arm joint
        th1 = 0
        th2 = 0
        th3 = 0

        # Draw the wheels
        th = np.linspace(0, 2 * np.pi, 20)
        wheel_left =  np.stack([self.r*np.cos(th),np.zeros_like(th) + self.b/2, self.r*np.sin(th) + self.r]).T
        wheel_right = np.stack([self.r*np.cos(th),np.zeros_like(th) - self.b/2, self.r*np.sin(th)+ self.r]).T

        ax.plot(*wheel_left.T, color='black')
        ax.plot(*wheel_right.T, color='black')

        # wheel link
        wheel_link = np.array([[0, self.b/2, self.r], 
                               [0, -self.b/2, self.r]])
        ax.plot(*wheel_link.T, color='black')

        # arm link 1
        arm_link1 = np.array([[0, 0, 0], 
                              [0, 0, self.arm[0]]]) + np.array([0, 0, self.r])
        ax.plot(*arm_link1.T, color='black')

        # # arm link 2


        R1 = np.array([[np.cos(th1), -np.sin(th1), 0], 
                       [np.sin(th1), np.cos(th1), 0], 
                          [0, 0, 1]])
        # along y
        R2 = np.array([[np.cos(th2), 0, np.sin(th2)], 
                       [0, 1, 0], 
                       [-np.sin(th2), 0, np.cos(th2)]])

        arm_link2_end = R1 @ np.array([0,0, self.arm[0]]) + R1@R2@np.array([0,0,self.arm[1]]) 
        arm_link2 = np.array([[0,0, self.arm[0]], 
                              arm_link2_end]) + np.array([0, 0, self.r])
        ax.plot(*arm_link2.T, color='black')

        # racket, rorate around z axis
        R3 = np.array([[np.cos(th3), -np.sin(th3), 0], 
                       [np.sin(th3), np.cos(th3), 0], 
                          [0, 0, 1]])
        
        racket = np.stack([self.racket_r*np.cos(th),np.zeros_like(th), self.racket_r*np.sin(th)+self.racket_r]).T
        racket = R1@R2@R3 @ racket.T + R1 @ np.array([[0],[0], [self.arm[0]]]) + R1@R2@np.array([[0],[0],[self.arm[1]]]) + np.array([[0], [0], [self.r]])
        racket = racket.T
        ax.plot(*racket.T, color='black')

# Example usage
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
robot = EstherRobot()
robot.draw_initial_pose(ax)
# draw_tennis_court(ax)
set_axes_equal(ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.show()

import matplotlib.pyplot as plt
import csv

import numpy as np
from scipy.spatial.transform import Rotation as R



# read csv
waypoint_path = 'data/debug/path_error/waypoint.csv'
odom_path = 'data/debug/path_error/odom.csv'

with open(waypoint_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = []
    for row in reader:
        data.append([float(x) for x in row])
    data = np.array(data)

with open(odom_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    odom = []
    for row in reader:
        odom.append([float(x) for x in row])
    odom = np.array(odom)
  
N = min(data.shape[0], odom.shape[0])
timespan = np.linspace(0, 10.0, N)
data = data[:N,:]
odom = odom[:N,:]

position_error = data[:,0:2] - odom[:,0:2]
position_error = np.linalg.norm(position_error, axis=1)

dy = odom[:,1] - data[:,1]
dx = odom[:,0] - data[:,0]
# compute the angle between the two vectors
angle = np.arctan2(dy, dx)
# convert to degrees
angle = np.degrees(angle)

# compute the euler angle from quaternion odom[:,3:7]
quat = odom[:,3:7]
euler_angles = R.from_quat(quat[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)  # (x, y, z, w) order

desired_angle = euler_angles[:,2] - angle

error_angle = (desired_angle - angle + np.pi) % (2 * np.pi) - np.pi # [-pi, pi]


# plot the data
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(121)
ax.plot(data[:,0], data[:,1], label='waypoint')
ax.plot(odom[:,0], odom[:,1], label='odom')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Path Following Error')
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.savefig('data/debug/path_error/path_following.png')

# plot the error
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(121)
ax.plot(timespan, position_error, label='position error')
ax.set_xlabel('timestep (seconds)')
ax.set_ylabel('position error (meters)')
ax.set_title('Position Error')
ax.legend()

ax = fig.add_subplot(122)
ax.set_ylim([-10,10])
ax.plot(timespan, error_angle, label='error angle')
ax.set_xlabel('timestep (seconds)')
ax.set_ylabel('angle (degrees)')
ax.set_title('Angle Error')

ax.legend()
plt.savefig('data/debug/path_error/error.png')
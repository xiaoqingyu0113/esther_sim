import numpy as np

from fgmp.factors import Jerr_compute_pose, compute_pose



x1= np.array([0.0, 0.0, 0.0])
u1= np.random.rand(2)
x2 = np.random.rand(3)
t = np.random.rand()


J_x1, J_u1, J_x2 = Jerr_compute_pose(x1, u1, x2, t)
print('J_x1:\n', J_x1)
print('J_u1:\n', J_u1)
print('J_x2:\n', J_x2)


J_num_x1 = np.zeros((3, 3))
J_num_u1 = np.zeros((3, 2))
J_num_x2 = np.zeros((3, 3))

delta = 1e-9
for i in range(3):
    x1_plus = x1.copy()
    x1_plus[i] += delta
    J_num_x1[:, i] = ( - compute_pose(x1_plus, u1, t)  + compute_pose(x1, u1, t)) / delta

for i in range(2):
    u1_plus = u1.copy()
    u1_plus[i] += delta
    J_num_u1[:, i] = (-compute_pose(x1, u1_plus, t) + compute_pose(x1, u1, t)) / delta

for i in range(3):
    x2_plus = x2.copy()
    x2_plus[i] += delta
    J_num_x2[:, i] = (x2_plus - x2) / delta

print('J_num_x1:\n', J_num_x1)
print('J_num_u1:\n', J_num_u1)
print('J_num_x2:\n', J_num_x2)
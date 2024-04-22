import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def rotate_left(lst, i=1):
    return np.hstack((lst[i:],lst[:i]))


# middle
h_s = 5.0    # high speed
m_s =4.0     # medium speed
l_s = 3.0    # low speed

x = np.array([-1.44, 1.22, 3.53, 5.04, 7.16, 5.0, 1.97, 0.94, 0.82, -1.4, -3.1, -3.0, -2.7])
y = np.array([-0.47,-0.55,-0.54,-0.43,  1.0, 2.5, 2.48, 3.96, 6.47, 8.55, 6.28, 2.94, 0.88])
z = np.array([  l_s,  m_s,  h_s,  m_s,  l_s,  m_s, m_s,  h_s,  m_s,  l_s,  m_s,  m_s,  l_s])


# due to the unsatisfactory effect at the junction of the beginning and end, try placing the junction in different segments.
for i in range(10):
    x = rotate_left(x)
    y = rotate_left(y)
    z = rotate_left(z)

# let the curve to be closed
if not (x[0], y[0], z[0]) == (x[-1], y[-1], z[-1]):
    x_ = np.append(x, x[0])
    y_ = np.append(y, y[0])
    z_ = np.append(z, z[0])
curve = [x_,y_,z_]

# fitting
tck, _ = splprep(curve, s=4)
print(tck)

# interpolate to some number of points
new_points = np.array(splev(np.linspace(0, 1, 100), tck))

# new_points[0,:] = new_points[0,:] - 0.3
# new_points[1,:]=new_points[1,:] -0.3

# visulazition
plt.figure()
plt.scatter(x, y, c=z, cmap='viridis', marker='o')
plt.plot(new_points[0], new_points[1], 'k-', alpha=0.5)
points = plt.scatter(new_points[0], new_points[1], c=new_points[2], cmap='viridis', s=10)
plt.colorbar(points, label='Z value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Curve with Z Value Represented by Color')
plt.show()

# add column yaw
new_points = new_points.T
points_length = new_points.shape[0]
yaws = np.zeros((points_length, 1))
for i in range(points_length + 1):
    curr_point = new_points[i % points_length, 0:2]
    prev_point = new_points[(i - 1) % points_length, 0:2]
    yaws[i % points_length] = np.arctan2(curr_point[1] - prev_point[1], curr_point[0] - prev_point[0])
new_points = np.hstack((new_points, yaws))
print(new_points.shape)

# save the result
csvfile = '/home/lucien/ESE6150/traj_map_middle_ls.csv'
np.savetxt(csvfile, new_points, delimiter=",")


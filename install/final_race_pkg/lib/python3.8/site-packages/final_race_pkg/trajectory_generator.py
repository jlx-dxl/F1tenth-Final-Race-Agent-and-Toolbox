import rclpy
from rclpy.node import Node
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        self.declare_parameter('high_speed', 0.55)
        self.declare_parameter('medium_speed', 0.0)
        self.declare_parameter('low_speed', 0.005)
        
        self.generate_single_trajectory()

    def rotate_left(self, lst, i=1):
        return np.hstack((lst[i:], lst[:i]))

    def generate_single_trajectory(self):
        h_s = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        m_s = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        l_s = self.get_parameter('lookahead_distance').get_parameter_value().double_value

        x = np.array([-1.44, 1.22, 3.53, 5.04, 7.16, 5.0, 1.97, 0.94, 0.82, -1.4, -3.1, -3.0, -2.7])
        y = np.array([-0.47,-0.55,-0.54,-0.43,  1.0, 2.5, 2.48, 3.96, 6.47, 8.55, 6.28, 2.94, 0.88])
        z = np.array([l_s, m_s, h_s, m_s, l_s, m_s, m_s, h_s, m_s, l_s, m_s, m_s, l_s])

        # Rotate points
        for i in range(10):
            x = self.rotate_left(x)
            y = self.rotate_left(y)
            z = self.rotate_left(z)

        # Close curve
        x_ = np.append(x, x[0])
        y_ = np.append(y, y[0])
        z_ = np.append(z, z[0])
        curve = [x_, y_, z_]

        # Fitting
        tck, _ = splprep(curve, s=4)
        new_points = np.array(splev(np.linspace(0, 1, 100), tck))

        self.get_logger().info('Trajectory generated')
        
    def determine_trajectory(self):
        return None
    
    def send_trajectory(self):
        return None

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
